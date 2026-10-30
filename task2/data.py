"""Task 2 data: COCO captions, annotation-derived grounding pairs and VQAv2 complementary pairs.

    python -m task2.data            (downloads annotations, selects all splits, downloads the images)

Selections use seed 6304. Training uses train2014 only; development and final-test sets come from val2014
and are image-disjoint from one another (each image is used by at most one evaluation example).
"""

import argparse
import io
import json
import re
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from task1.common import load_json, save_json
from task2.config import (CAPTION_DEV_IMAGES, CAPTION_TRAIN_IMAGES, CAPTIONS_PER_TRAIN_IMAGE, COCO_ANNOTATIONS_URL,
                          COCO_IMAGE_URL, COUNT_RANGE, GROUNDING_DEV_PAIRS, GROUNDING_POOL_PAIRS,
                          GROUNDING_TEST_PAIRS, MIN_INSTANCE_AREA_FRACTION, SEED, VQA_DEV_PAIRS, VQA_FILES,
                          VQA_MIN_AGREEMENT, VQA_TEST_PAIRS, VQA_TRAIN_PAIRS, VQA_URL, data_root, results_root)
from task2.vqa_eval import normalize_answer

IRREGULAR_PLURALS = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis", "scissors": "scissors",
                     "knife": "knives", "skateboard": "skateboards"}


def plural(name):
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    if name.endswith(("s", "sh", "ch", "x")):
        return name + "es"
    return name + "s"


def with_article(name):
    if name in ("skis", "scissors"):
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def presence_question(name):
    return f"Does this image contain {with_article(name)}?"


def counting_question(name):
    return f"How many {plural(name)} can you see?"


def image_key(split, image_id):
    return f"{split}/{image_id}"


def image_path(key):
    split, image_id = key.split("/")
    return data_root() / "images" / split / f"COCO_{split}_{int(image_id):012d}.jpg"


def download(url, destination, attempts=5):
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(temporary, "wb") as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
            temporary.replace(destination)
            return destination
        except Exception as error:
            if attempt == attempts - 1:
                raise
            print(f"retry {url}: {error}", flush=True)
            time.sleep(2 + 3 * attempt)
    return destination


def zip_json(archive_path, member_suffix):
    with zipfile.ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(member_suffix))
        with archive.open(member) as handle:
            return json.load(io.TextIOWrapper(handle, encoding="utf-8"))


def raw_root():
    return data_root() / "raw"


def load_coco(split):
    archive = download(COCO_ANNOTATIONS_URL, raw_root() / "annotations_trainval2014.zip")
    captions = zip_json(archive, f"captions_{split}.json")
    instances = zip_json(archive, f"instances_{split}.json")
    return captions, instances


def load_vqa(split):
    questions_name, annotations_name, pairs_name = VQA_FILES[split]
    questions = zip_json(download(VQA_URL.format(name=questions_name), raw_root() / f"{questions_name}.zip"), ".json")
    annotations = zip_json(download(VQA_URL.format(name=annotations_name), raw_root() / f"{annotations_name}.zip"),
                           ".json")
    pairs = zip_json(download(VQA_URL.format(name=pairs_name), raw_root() / f"{pairs_name}.zip"), ".json")
    return questions["questions"], annotations["annotations"], pairs


class CocoIndex:
    """Per-image captions and per-category instance lists for one COCO split."""

    def __init__(self, split, captions, instances):
        self.split = split
        self.category_names = {category["id"]: category["name"] for category in instances["categories"]}
        self.image_area = {image["id"]: image["width"] * image["height"] for image in instances["images"]}
        self.captions = defaultdict(list)
        for annotation in sorted(captions["annotations"], key=lambda row: row["id"]):
            self.captions[annotation["image_id"]].append(" ".join(annotation["caption"].split()))
        self.instances = defaultdict(lambda: defaultdict(list))
        for annotation in instances["annotations"]:
            self.instances[annotation["image_id"]][annotation["category_id"]].append(annotation)
        self.mention_patterns = {}
        for category_id, name in self.category_names.items():
            words = sorted({name, plural(name)}, key=len, reverse=True)
            self.mention_patterns[category_id] = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")

    def clean_positive(self, image_id, category_id):
        annotations = self.instances[image_id].get(category_id, [])
        minimum = MIN_INSTANCE_AREA_FRACTION * self.image_area[image_id]
        return any(not row["iscrowd"] and row["area"] >= minimum for row in annotations)

    def clean_negative(self, image_id, category_id):
        """No annotated instance and no caption mentioning the category (COCO annotations can omit objects)."""
        if self.instances[image_id].get(category_id):
            return False
        pattern = self.mention_patterns[category_id]
        return not any(pattern.search(caption.lower()) for caption in self.captions[image_id])

    def clean_count(self, image_id, category_id):
        """Instance count when it is unambiguous: no crowd regions and every instance clearly visible."""
        annotations = self.instances[image_id].get(category_id, [])
        if not annotations or any(row["iscrowd"] for row in annotations):
            return None
        minimum = MIN_INSTANCE_AREA_FRACTION * self.image_area[image_id]
        if any(row["area"] < minimum for row in annotations):
            return None
        count = len(annotations)
        if not COUNT_RANGE[0] <= count <= COUNT_RANGE[1]:
            return None
        return count


def grounding_example(kind, coco, image_id, category_id, answer, pair_id, side):
    name = coco.category_names[category_id]
    question = presence_question(name) if kind == "presence" else counting_question(name)
    return {"uid": f"{pair_id}:{side}", "kind": kind, "image": image_key(coco.split, image_id), "question": question,
            "target": answer, "canonical": normalize_answer(answer), "category": name, "pair_id": pair_id}


class GroundingCandidates:
    """Per-category image lists that satisfy the label-quality filters, restricted to the allowed images."""

    def __init__(self, coco, allowed_images):
        allowed = sorted(allowed_images)
        self.coco = coco
        self.positives = defaultdict(list)
        self.negatives = defaultdict(list)
        self.by_count = defaultdict(lambda: defaultdict(list))
        for category_id in sorted(coco.category_names):
            for image_id in allowed:
                if coco.clean_positive(image_id, category_id):
                    self.positives[category_id].append(image_id)
                elif coco.clean_negative(image_id, category_id):
                    self.negatives[category_id].append(image_id)
                count = coco.clean_count(image_id, category_id)
                if count is not None:
                    self.by_count[category_id][count].append(image_id)
        self.presence_categories = [category for category in sorted(coco.category_names)
                                    if self.positives[category] and self.negatives[category]]
        self.counting_categories = [category for category in sorted(coco.category_names)
                                    if len(self.by_count[category]) >= 2]

    def draw_presence(self, rng):
        category = self.presence_categories[rng.integers(len(self.presence_categories))]
        positive = self.positives[category][rng.integers(len(self.positives[category]))]
        negative = self.negatives[category][rng.integers(len(self.negatives[category]))]
        if rng.random() < 0.5:
            return category, (positive, "yes"), (negative, "no")
        return category, (negative, "no"), (positive, "yes")

    def draw_counting(self, rng):
        category = self.counting_categories[rng.integers(len(self.counting_categories))]
        counts = sorted(self.by_count[category])
        first, second = rng.choice(counts, size=2, replace=False)
        first_image = self.by_count[category][first][rng.integers(len(self.by_count[category][first]))]
        second_image = self.by_count[category][second][rng.integers(len(self.by_count[category][second]))]
        return category, (first_image, str(first)), (second_image, str(second))

    def make_pair(self, kind, rng, prefix, index):
        if kind == "presence":
            category, first, second = self.draw_presence(rng)
        else:
            category, first, second = self.draw_counting(rng)
        pair_id = f"{prefix}-{kind}-{index:05d}"
        examples = [grounding_example(kind, self.coco, image_id, category, answer, pair_id, side)
                    for side, (image_id, answer) in zip("ab", (first, second))]
        return {"pair_id": pair_id, "kind": kind, "examples": examples}


def vqa_candidates(split, questions, annotations, pairs):
    """Complementary pairs whose normalized consensus answers differ, each with at least 8/10 agreement."""
    question_by_id = {row["question_id"]: row for row in questions}
    annotation_by_id = {row["question_id"]: row for row in annotations}
    candidates = []
    for first_id, second_id in pairs:
        examples = []
        for question_id in (first_id, second_id):
            annotation = annotation_by_id[question_id]
            raw_answers = [row["answer"] for row in annotation["answers"]]
            normalized = [normalize_answer(answer) for answer in raw_answers]
            consensus, agreement = Counter(normalized).most_common(1)[0]
            if agreement < VQA_MIN_AGREEMENT:
                break
            group = Counter(raw for raw, norm in zip(raw_answers, normalized) if norm == consensus)
            target = max(group, key=lambda raw: (group[raw], -raw_answers.index(raw)))
            examples.append({"uid": f"vqa-{question_id}", "kind": "vqa",
                             "image": image_key(f"{split}2014", annotation["image_id"]),
                             "question": question_by_id[question_id]["question"].strip(), "target": target,
                             "canonical": consensus, "answers": raw_answers,
                             "answer_type": annotation["answer_type"], "question_id": question_id})
        if len(examples) != 2:
            continue
        first, second = examples
        if first["canonical"] == second["canonical"] or first["image"] == second["image"]:
            continue
        if normalize_answer(first["question"]) != normalize_answer(second["question"]):
            continue
        if first["answer_type"] != second["answer_type"]:
            continue
        pair_id = f"vqa-{first_id}-{second_id}"
        for example in examples:
            example["pair_id"] = pair_id
        candidates.append({"pair_id": pair_id, "kind": "vqa", "answer_type": first["answer_type"],
                           "examples": examples})
    return candidates


def pair_images(pair):
    return [example["image"] for example in pair["examples"]]


def select_disjoint(candidates, quotas, used_images, rng, type_key):
    """Seeded selection by type; every chosen image is new to the evaluation pool (then marked used)."""
    order = rng.permutation(len(candidates))
    remaining = dict(quotas)
    chosen = []
    for index in order:
        pair = candidates[index]
        kind = pair[type_key]
        if remaining.get(kind, 0) == 0:
            continue
        images = pair_images(pair)
        if any(image in used_images for image in images):
            continue
        used_images.update(images)
        chosen.append(pair)
        remaining[kind] -= 1
        if not any(remaining.values()):
            break
    if any(remaining.values()):
        raise RuntimeError(f"not enough candidates for {quotas}: missing {remaining}")
    return chosen


def select_grounding_eval(candidates, quotas, used_images, rng, prefix):
    chosen = []
    for kind, quota in quotas.items():
        attempts = 0
        while sum(pair["kind"] == kind for pair in chosen) < quota:
            attempts += 1
            if attempts > 200_000:
                raise RuntimeError(f"cannot find disjoint {kind} pairs")
            pair = candidates.make_pair(kind, rng, prefix, len(chosen))
            images = pair_images(pair)
            if images[0] == images[1] or any(image in used_images for image in images):
                continue
            used_images.update(images)
            chosen.append(pair)
    return chosen


def build_selection():
    rng = np.random.default_rng(SEED)
    started = time.time()

    print("loading val2014", flush=True)
    val_captions, val_instances = load_coco("val2014")
    val_coco = CocoIndex("val2014", val_captions, val_instances)
    del val_captions, val_instances
    val_questions, val_annotations, val_pairs = load_vqa("val")
    val_vqa = vqa_candidates("val", val_questions, val_annotations, val_pairs)
    del val_questions, val_annotations, val_pairs
    print(f"val VQA candidate pairs: {len(val_vqa)} {Counter(pair['answer_type'] for pair in val_vqa)}", flush=True)

    used_eval_images = set()
    vqa_test = select_disjoint(val_vqa, VQA_TEST_PAIRS, used_eval_images, rng, "answer_type")
    vqa_dev = select_disjoint(val_vqa, VQA_DEV_PAIRS, used_eval_images, rng, "answer_type")

    val_image_ids = sorted(val_coco.image_area)
    remaining_val = [image_id for image_id in val_image_ids
                     if image_key("val2014", image_id) not in used_eval_images]
    grounding_pool_size = min(len(remaining_val), 8000)
    grounding_val_images = rng.choice(remaining_val, size=grounding_pool_size, replace=False).tolist()
    val_grounding = GroundingCandidates(val_coco, grounding_val_images)
    grounding_test = select_grounding_eval(val_grounding, GROUNDING_TEST_PAIRS, used_eval_images, rng, "gtest")
    grounding_dev = select_grounding_eval(val_grounding, GROUNDING_DEV_PAIRS, used_eval_images, rng, "gdev")

    caption_dev = []
    captioned_val = [image_id for image_id in val_image_ids
                     if val_coco.captions[image_id] and image_key("val2014", image_id) not in used_eval_images]
    for image_id in rng.choice(captioned_val, size=CAPTION_DEV_IMAGES, replace=False).tolist():
        captions = val_coco.captions[image_id]
        key = image_key("val2014", image_id)
        used_eval_images.add(key)
        caption_dev.append({"uid": f"capdev-{image_id}", "kind": "caption", "image": key,
                            "target": captions[rng.integers(len(captions))]})
    del val_coco, val_grounding

    print("loading train2014", flush=True)
    train_captions, train_instances = load_coco("train2014")
    train_coco = CocoIndex("train2014", train_captions, train_instances)
    del train_captions, train_instances
    train_questions, train_annotations, train_pairs = load_vqa("train")
    train_vqa_candidates = vqa_candidates("train", train_questions, train_annotations, train_pairs)
    del train_questions, train_annotations, train_pairs
    print(f"train VQA candidate pairs: {len(train_vqa_candidates)}", flush=True)
    vqa_train = select_disjoint(train_vqa_candidates, VQA_TRAIN_PAIRS, set(), rng, "answer_type")

    eligible = [image_id for image_id in sorted(train_coco.image_area)
                if len(train_coco.captions[image_id]) >= CAPTIONS_PER_TRAIN_IMAGE]
    caption_images = rng.choice(eligible, size=CAPTION_TRAIN_IMAGES, replace=False).tolist()
    caption_train = []
    for image_id in caption_images:
        captions = train_coco.captions[image_id]
        for caption_index in sorted(rng.choice(len(captions), size=CAPTIONS_PER_TRAIN_IMAGE, replace=False)):
            caption_train.append({"uid": f"cap-{image_id}-{caption_index}", "kind": "caption",
                                  "image": image_key("train2014", image_id), "target": captions[caption_index]})

    vqa_train_images = {int(image.split("/")[1]) for pair in vqa_train for image in pair_images(pair)}
    pool_images = sorted(set(caption_images) | vqa_train_images)
    train_grounding = GroundingCandidates(train_coco, pool_images)
    pools = {}
    for kind, size in GROUNDING_POOL_PAIRS.items():
        pools[kind] = [train_grounding.make_pair(kind, rng, "gtrain", index) for index in range(size)]
        pools[kind] = [pair for pair in pools[kind] if pair["examples"][0]["image"] != pair["examples"][1]["image"]]

    manifest = {
        "train": {"caption": caption_train, "presence": pools["presence"], "counting": pools["counting"],
                  "vqa": vqa_train},
        "dev": {"caption": caption_dev, "grounding": grounding_dev, "vqa": vqa_dev},
        "test": {"grounding": grounding_test, "vqa": vqa_test},
        "info": {
            "seed": SEED,
            "selection_seconds": time.time() - started,
            "grounding_train_pool_images": len(pool_images),
            "presence_categories_train": len(train_grounding.presence_categories),
            "counting_categories_train": len(train_grounding.counting_categories),
            "val_vqa_candidates": len(val_vqa),
            "train_vqa_candidates": len(train_vqa_candidates),
            "min_instance_area_fraction": MIN_INSTANCE_AREA_FRACTION,
        },
    }
    check_isolation(manifest)
    return manifest


def manifest_images(manifest):
    images = set()
    for split in ("train", "dev", "test"):
        for items in manifest[split].values():
            for item in items:
                for example in item.get("examples", [item]):
                    images.add(example["image"])
    return sorted(images)


def split_images(manifest, split, component=None):
    images = set()
    for name, items in manifest[split].items():
        if component is not None and name != component:
            continue
        for item in items:
            for example in item.get("examples", [item]):
                images.add(example["image"])
    return images


def check_isolation(manifest):
    train_images = split_images(manifest, "train")
    evaluation_sets = [split_images(manifest, split, component)
                       for split in ("dev", "test") for component in manifest[split]]
    assert all(image.startswith("train2014/") for image in train_images)
    for index, first in enumerate(evaluation_sets):
        assert all(image.startswith("val2014/") for image in first)
        assert not first & train_images
        for second in evaluation_sets[index + 1:]:
            assert not first & second, "evaluation sets share images"
    for split in ("dev", "test"):
        for component in ("grounding", "vqa"):
            images = [image for pair in manifest[split].get(component, []) for image in pair_images(pair)]
            assert len(images) == len(set(images)), f"{split}/{component} reuses an image"


def selection_summary(manifest):
    def type_counts(pairs, key):
        return dict(Counter(pair[key] for pair in pairs))

    return {
        "train": {"caption_examples": len(manifest["train"]["caption"]),
                  "caption_images": len({row["image"] for row in manifest["train"]["caption"]}),
                  "presence_pool_pairs": len(manifest["train"]["presence"]),
                  "counting_pool_pairs": len(manifest["train"]["counting"]),
                  "vqa_pairs": type_counts(manifest["train"]["vqa"], "answer_type")},
        "dev": {"caption_images": len(manifest["dev"]["caption"]),
                "grounding_pairs": type_counts(manifest["dev"]["grounding"], "kind"),
                "vqa_pairs": type_counts(manifest["dev"]["vqa"], "answer_type")},
        "test": {"grounding_pairs": type_counts(manifest["test"]["grounding"], "kind"),
                 "vqa_pairs": type_counts(manifest["test"]["vqa"], "answer_type")},
        "answer_distribution": {
            f"{split}/{component}": dict(Counter(example["canonical"] for pair in manifest[split][component]
                                                 for example in pair["examples"]).most_common(12))
            for split, component in (("train", "presence"), ("train", "counting"), ("dev", "grounding"),
                                     ("test", "grounding"), ("test", "vqa"))},
        "unique_images": len(manifest_images(manifest)),
        "info": manifest["info"],
    }


def save_split_ids(manifest):
    """Image, question and pair IDs (committed to the repo) for every selected split."""
    splits = results_root() / "splits"
    save_json(splits / "caption_train.json", [[row["image"], row["uid"]] for row in manifest["train"]["caption"]])
    save_json(splits / "caption_dev.json", [[row["image"], row["uid"]] for row in manifest["dev"]["caption"]])
    for split, component in (("train", "vqa"), ("dev", "vqa"), ("test", "vqa"), ("dev", "grounding"),
                             ("test", "grounding")):
        save_json(splits / f"{component}_{split}.json", manifest[split][component])
    for kind in ("presence", "counting"):
        save_json(splits / f"{kind}_train_pool.json",
                  [[pair["pair_id"], pair["examples"][0]["category"]]
                   + [[example["image"], example["target"]] for example in pair["examples"]]
                   for pair in manifest["train"][kind]])
    save_json(splits / "summary.json", selection_summary(manifest))


def download_images(images, workers=32):
    started = time.time()

    def fetch(key):
        split, image_id = key.split("/")
        url = COCO_IMAGE_URL.format(split=split, image_id=int(image_id))
        return download(url, image_path(key)).stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes = list(pool.map(fetch, images))
    return {"images": len(images), "bytes": int(sum(sizes)), "seconds": time.time() - started}


def manifest_path():
    return data_root() / "manifest.json"


def load_manifest():
    return load_json(manifest_path())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    if manifest_path().exists():
        manifest = load_manifest()
        print("manifest exists, reusing it", flush=True)
    else:
        manifest = build_selection()
        save_json(manifest_path(), manifest)
        save_split_ids(manifest)
    print(json.dumps(selection_summary(manifest), indent=1), flush=True)
    if not args.skip_images:
        report = download_images(manifest_images(manifest))
        save_json(data_root() / "download_info.json", report)
        print(report, flush=True)


if __name__ == "__main__":
    main()
