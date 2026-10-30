"""SigLIP-Qwen VLM: frozen FP16 Qwen3-0.6B, FP32 MLP connector, optional FP32 LoRA on attention projections.

Visual tokens replace image placeholders inside the user turn, before the question:
    <|im_start|>user\n [visual tokens] \n question \n instruction <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
No-image inputs drop the visual tokens only; positions and masks are rebuilt from the shorter sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from task1.common import load_causal_lm, load_tokenizer
from task1.lora import LoRALinear
from task2.config import (ANSWER_TOKEN_CAP, CAPTION_PROMPT, CAPTION_TOKEN_CAP, GRID, LORA_ALPHA, LORA_DROPOUT,
                          LORA_RANK, LORA_TARGETS, MODEL_NAME, QUESTION_TOKEN_CAP, SHORT_ANSWER_SUFFIX, TEXT_DIM,
                          VISION_DIM)

IMAGE_PLACEHOLDER = "<|image_pad|>"
LOSS_CHUNK_TOKENS = 256


def pool_features(features, visual_tokens):
    """196 tokens as cached, or 49 tokens by averaging each 2 x 2 block of the 14 x 14 grid."""
    if visual_tokens == GRID * GRID:
        return features
    batch = features.shape[0]
    grid = features.reshape(batch, GRID, GRID, VISION_DIM).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid.float(), kernel_size=2)
    return pooled.flatten(2).transpose(1, 2)


class Connector(nn.Module):
    def __init__(self, visual_tokens):
        super().__init__()
        self.visual_tokens = visual_tokens
        self.fc1 = nn.Linear(VISION_DIM, TEXT_DIM)
        self.fc2 = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.register_buffer("output_scale", torch.ones(()))

    def forward(self, features):
        hidden = pool_features(features, self.visual_tokens).float()
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.fc2(F.gelu(self.fc1(hidden))) * self.output_scale


class Encoder:
    """Turns examples into token IDs and label masks; content caps are applied before the chat wrapper."""

    def __init__(self, tokenizer, visual_tokens):
        self.tokenizer = tokenizer
        self.visual_tokens = visual_tokens
        self.placeholder_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        self.end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.user_open = self.ids("<|im_start|>user\n")
        self.newline = self.ids("\n")
        self.assistant_open = self.ids("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.caption_prompt = self.ids(CAPTION_PROMPT)

    def ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def prompt_ids(self, example, with_image=True):
        if example["kind"] == "caption":
            text_ids = self.caption_prompt
        else:
            question = self.tokenizer.decode(self.ids(example["question"])[:QUESTION_TOKEN_CAP])
            text_ids = self.ids(question + "\n" + SHORT_ANSWER_SUFFIX)
        visual = [self.placeholder_id] * self.visual_tokens if with_image else []
        return self.user_open + visual + self.newline + text_ids + self.assistant_open

    def target_ids(self, example):
        cap = CAPTION_TOKEN_CAP if example["kind"] == "caption" else ANSWER_TOKEN_CAP
        return self.ids(example["target"])[:cap] + [self.end_id]

    def chat_template_ids(self, example):
        """Reference tokenization through Qwen's own chat template (used by the interface check)."""
        text = CAPTION_PROMPT if example["kind"] == "caption" else example["question"] + "\n" + SHORT_ANSWER_SUFFIX
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": IMAGE_PLACEHOLDER * self.visual_tokens + "\n" + text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.ids(rendered)


def collate_training(encoder, examples, with_image=True, include_end=True):
    """Right-padded batch. Labels cover target tokens (and the end-of-turn token when include_end)."""
    sequences, label_masks = [], []
    for example in examples:
        prompt = encoder.prompt_ids(example, with_image)
        target = encoder.target_ids(example)
        supervised = [True] * len(target)
        if not include_end:
            supervised[-1] = False
        sequences.append(prompt + target)
        label_masks.append([False] * len(prompt) + supervised)
    length = max(len(sequence) for sequence in sequences)
    pad_id = encoder.tokenizer.pad_token_id
    input_ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
    for row, (sequence, mask) in enumerate(zip(sequences, label_masks)):
        input_ids[row, :len(sequence)] = torch.tensor(sequence)
        attention_mask[row, :len(sequence)] = 1
        labels[row, :len(sequence)] = torch.where(torch.tensor(mask), torch.tensor(sequence), torch.tensor(-100))
    return input_ids, labels, attention_mask


def collate_generation(encoder, examples, with_image=True):
    prompts = [encoder.prompt_ids(example, with_image) for example in examples]
    length = max(len(prompt) for prompt in prompts)
    pad_id = encoder.tokenizer.pad_token_id
    input_ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        input_ids[row, length - len(prompt):] = torch.tensor(prompt)
        attention_mask[row, length - len(prompt):] = 1
    return input_ids, attention_mask


def chunk_token_losses(hidden, targets, lm_head):
    return F.cross_entropy(lm_head(hidden).float(), targets, reduction="none")


class VLM(nn.Module):
    def __init__(self, visual_tokens, device):
        super().__init__()
        self.tokenizer = load_tokenizer(MODEL_NAME)
        self.encoder = Encoder(self.tokenizer, visual_tokens)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.language_model = load_causal_lm(MODEL_NAME, dtype, device)
        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)
        self.language_model.config.use_cache = False
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.connector = Connector(visual_tokens).to(device)
        self.adapters = {}
        self.device = device

    def enable_lora(self):
        for layer_index, layer in enumerate(self.language_model.model.layers):
            for target in LORA_TARGETS:
                module = getattr(layer.self_attn, target)
                adapter = LoRALinear(module, LORA_RANK, LORA_ALPHA, dropout=LORA_DROPOUT)
                setattr(layer.self_attn, target, adapter)
                self.adapters[f"model.layers.{layer_index}.self_attn.{target}"] = adapter

    def trainable_parameters(self):
        groups = {"connector": [parameter for parameter in self.connector.parameters()]}
        if self.adapters:
            groups["lora"] = [parameter for adapter in self.adapters.values()
                              for parameter in (adapter.lora_A, adapter.lora_B)]
        return groups

    def adapter_state(self):
        return {name: {"lora_A": adapter.lora_A.detach().cpu(), "lora_B": adapter.lora_B.detach().cpu()}
                for name, adapter in self.adapters.items()}

    @torch.no_grad()
    def load_adapter_state(self, state):
        if state and not self.adapters:
            self.enable_lora()
        for name, tensors in state.items():
            self.adapters[name].lora_A.copy_(tensors["lora_A"])
            self.adapters[name].lora_B.copy_(tensors["lora_B"])

    def embed(self, input_ids, features):
        embeddings = self.language_model.get_input_embeddings()(input_ids)
        visual_positions = input_ids == self.encoder.placeholder_id
        if features is None or not visual_positions.any():
            return embeddings
        visual = self.connector(features.to(self.device)).to(embeddings.dtype)
        assert visual_positions.sum().item() == visual.shape[0] * visual.shape[1], "visual token count mismatch"
        return embeddings.masked_scatter(visual_positions.unsqueeze(-1), visual)

    def example_losses(self, input_ids, labels, attention_mask, features):
        """Mean token NLL over each example's supervised tokens (shape: batch)."""
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        attention_mask = attention_mask.to(self.device)
        embeddings = self.embed(input_ids, features)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        hidden = self.language_model.model(inputs_embeds=embeddings, attention_mask=attention_mask,
                                           position_ids=position_ids).last_hidden_state
        shifted_targets = labels[:, 1:]
        supervised = shifted_targets != -100
        selected_hidden = hidden[:, :-1][supervised]
        selected_targets = shifted_targets[supervised]
        token_losses = []
        for start in range(0, selected_targets.numel(), LOSS_CHUNK_TOKENS):
            end = start + LOSS_CHUNK_TOKENS
            if self.training:
                token_losses.append(checkpoint(chunk_token_losses, selected_hidden[start:end],
                                               selected_targets[start:end], self.language_model.lm_head,
                                               use_reentrant=False))
            else:
                token_losses.append(chunk_token_losses(selected_hidden[start:end], selected_targets[start:end],
                                                       self.language_model.lm_head))
        token_losses = torch.cat(token_losses)
        example_index = torch.arange(input_ids.shape[0], device=self.device).unsqueeze(1).expand_as(supervised)
        example_index = example_index[supervised]
        sums = torch.zeros(input_ids.shape[0], device=self.device).index_add(0, example_index, token_losses)
        counts = supervised.sum(dim=1).clamp(min=1)
        return sums / counts

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, features, max_new_tokens):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        embeddings = self.embed(input_ids, features)
        output = self.language_model.generate(inputs_embeds=embeddings, attention_mask=attention_mask,
                                              max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                              eos_token_id=self.encoder.end_id,
                                              pad_token_id=self.tokenizer.pad_token_id, temperature=None,
                                              top_p=None, top_k=None)
        texts = []
        for row in output:
            tokens = row.tolist()
            if self.encoder.end_id in tokens:
                tokens = tokens[:tokens.index(self.encoder.end_id)]
            texts.append(self.tokenizer.decode(tokens, skip_special_tokens=True).strip())
        return texts
