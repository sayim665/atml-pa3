"""CPU smoke test of the Task 2 pipeline on a synthetic manifest with random images (no downloads).

    ATML_SMOKE_TEST=1 python -m task2.smoke
"""

import numpy as np

from task1.common import save_json
from task2.config import SMOKE_TEST, data_root
from task2.data import check_isolation, image_path, manifest_path


def synthetic_manifest():
    rng = np.random.default_rng(0)
    counter = iter(range(1, 10_000))

    def image(split):
        return f"{split}/{next(counter)}"

    def pair(kind, split, index, answers, question):
        pair_id = f"{split}-{kind}-{index}"
        examples = []
        for side, answer in zip("ab", answers):
            example = {"uid": f"{pair_id}:{side}", "kind": kind, "image": image(split), "question": question,
                       "target": answer, "canonical": answer, "pair_id": pair_id}
            if kind == "vqa":
                example.update(answers=[answer] * 8 + ["other"] * 2, answer_type=["yes/no", "number", "other"][index % 3])
            examples.append(example)
        record = {"pair_id": pair_id, "kind": kind, "examples": examples}
        if kind == "vqa":
            record["answer_type"] = examples[0]["answer_type"]
        return record

    caption_train = [{"uid": f"cap-{index}", "kind": "caption", "image": image("train2014"),
                      "target": "A small dog runs across a grassy field."} for index in range(12)]
    manifest = {
        "train": {"caption": caption_train,
                  "presence": [pair("presence", "train2014", index, ("yes", "no"), "Does this image contain a dog?")
                               for index in range(6)],
                  "counting": [pair("counting", "train2014", index, ("2", "3"), "How many dogs can you see?")
                               for index in range(6)],
                  "vqa": [pair("vqa", "train2014", index, ("red", "blue"), "What color is the car?")
                          for index in range(6)]},
        "dev": {"caption": [{"uid": f"capdev-{index}", "kind": "caption", "image": image("val2014"),
                             "target": "A cat sleeping on a couch."} for index in range(3)],
                "grounding": [pair("presence", "val2014", 0, ("yes", "no"), "Does this image contain a cat?"),
                              pair("counting", "val2014", 1, ("1", "4"), "How many cats can you see?")],
                "vqa": [pair("vqa", "val2014", index, ("yes", "no"), "Is it raining?") for index in range(3)]},
        "test": {"grounding": [pair("presence", "val2014", 10 + index, ("yes", "no"), "Does this image contain a cat?")
                               for index in range(3)],
                 "vqa": [pair("vqa", "val2014", 20 + index, ("2", "3"), "How many people are there?")
                         for index in range(3)]},
        "info": {"synthetic": True},
    }
    check_isolation(manifest)
    return manifest, rng


def main():
    assert SMOKE_TEST, "set ATML_SMOKE_TEST=1"
    from PIL import Image

    from task2.data import manifest_images

    manifest, rng = synthetic_manifest()
    save_json(manifest_path(), manifest)
    for key in manifest_images(manifest):
        path = image_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
    print(f"synthetic manifest with {len(manifest_images(manifest))} images in {data_root()}")


if __name__ == "__main__":
    main()
