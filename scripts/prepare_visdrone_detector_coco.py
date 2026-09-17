#!/usr/bin/env python3
"""
Reconstruct the frozen VisDrone detector dataset used by UAR-MOT.

Eligibility:
    score == 1
    category in {1, 4, 5, 6, 9}

track_id is NOT an eligibility condition.

VisDrone -> contiguous COCO categories:
    1 -> 1 pedestrian
    4 -> 2 car
    5 -> 3 van
    6 -> 4 truck
    9 -> 5 bus

Output:
    /content/UAR-MOT_detector_data/
        train2017/
        val2017/
        annotations/
            instances_train2017.json
            instances_val2017.json

Images are renamed:
    <sequence>__<original_filename>

Boundary-crossing boxes are clipped deterministically.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


DEFAULT_PROJECT = Path("/content/UAR-MOT")

DEFAULT_SOURCE = Path(
    "/content/drive/MyDrive/UAR-MOT/datasets/"
    "VisDrone/VisDrone2019-MOT-train"
)

DEFAULT_OUTPUT = Path("/content/UAR-MOT_detector_data")

RAW_TO_COCO = {
    1: (1, "pedestrian"),
    4: (2, "car"),
    5: (3, "van"),
    6: (4, "truck"),
    9: (5, "bus"),
}

EXPECTED = {
    "train": {
        "sequences": 27,
        "images": 13067,
        "annotations": 475877,
        "classes": {
            1: 123298,
            2: 302866,
            3: 26662,
            4: 18192,
            5: 4859,
        },
    },
    "val": {
        "sequences": 5,
        "images": 2104,
        "annotations": 51776,
        "classes": {
            1: 15154,
            2: 33715,
            3: 1729,
            4: 1017,
            5: 161,
        },
    },
    "combined": {
        "sequences": 32,
        "images": 15171,
        "annotations": 527653,
        "raw_rows": 757678,
        "inside": 527637,
        "cross_boundary": 16,
        "fully_outside": 0,
        "invalid_geometry": 0,
        "track_id_zero": 3392,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate source without creating output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )

    return parser.parse_args()


def read_manifest(path):
    if not path.is_file():
        raise RuntimeError(f"Manifest missing: {path}")

    values = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(values) != len(set(values)):
        raise RuntimeError(
            f"Duplicate sequence in manifest: {path}"
        )

    return values


def get_image_files(sequence_dir):
    return sorted(
        [
            p for p in sequence_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in {".jpg", ".jpeg", ".png"}
            )
        ],
        key=lambda p: int(p.stem),
    )


def load_sequence_images(sequence_dir):
    images = get_image_files(sequence_dir)
    frame_info = {}

    for path in images:
        try:
            frame = int(path.stem)
        except ValueError as exc:
            raise RuntimeError(
                f"Non-numeric frame filename: {path}"
            ) from exc

        if frame in frame_info:
            raise RuntimeError(
                f"Duplicate frame {frame} in {sequence_dir}"
            )

        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        height, width = image.shape[:2]

        frame_info[frame] = (
            path,
            int(width),
            int(height),
        )

    return images, frame_info


def parse_row(annotation_file, line_number, line):
    fields = line.split(",")

    if len(fields) != 10:
        raise RuntimeError(
            f"{annotation_file}:{line_number}: "
            f"expected 10 columns, got {len(fields)}"
        )

    try:
        values = [float(x) for x in fields]
    except ValueError as exc:
        raise RuntimeError(
            f"{annotation_file}:{line_number}: "
            "non-numeric annotation"
        ) from exc

    return {
        "frame": int(values[0]),
        "track_id": int(values[1]),
        "x": float(values[2]),
        "y": float(values[3]),
        "width": float(values[4]),
        "height": float(values[5]),
        "score": int(values[6]),
        "category": int(values[7]),
    }


def clip_box(x, y, width, height, image_width, image_height):
    if width <= 0 or height <= 0:
        return "invalid_geometry", None

    x2 = x + width
    y2 = y + height

    if (
        x2 <= 0
        or y2 <= 0
        or x >= image_width
        or y >= image_height
    ):
        return "fully_outside", None

    crosses = (
        x < 0
        or y < 0
        or x2 > image_width
        or y2 > image_height
    )

    x1c = max(0.0, x)
    y1c = max(0.0, y)
    x2c = min(float(image_width), x2)
    y2c = min(float(image_height), y2)

    wc = x2c - x1c
    hc = y2c - y1c

    if wc <= 0 or hc <= 0:
        raise RuntimeError(
            "Clipping produced non-positive geometry."
        )

    return (
        "cross_boundary" if crosses else "inside",
        [x1c, y1c, wc, hc],
    )


def categories():
    return [
        {
            "id": coco_id,
            "name": name,
            "supercategory": "object",
        }
        for _, (coco_id, name)
        in sorted(RAW_TO_COCO.items())
    ]


def process_split(
    split_name,
    sequences,
    source_root,
    output_root,
    audit_only,
    global_stats,
):
    sequence_root = source_root / "sequences"
    annotation_root = source_root / "annotations"

    coco_images = []
    coco_annotations = []
    class_counts = Counter()

    image_id = 1
    annotation_id = 1

    output_image_dir = output_root / (
        "train2017"
        if split_name == "train"
        else "val2017"
    )

    for sequence in sequences:
        sequence_dir = sequence_root / sequence
        annotation_file = (
            annotation_root / f"{sequence}.txt"
        )

        if not sequence_dir.is_dir():
            raise RuntimeError(
                f"Missing sequence directory: {sequence_dir}"
            )

        if not annotation_file.is_file():
            raise RuntimeError(
                f"Missing annotation file: {annotation_file}"
            )

        images, frame_info = load_sequence_images(
            sequence_dir
        )

        frame_to_image_id = {}

        for source_image in images:
            frame = int(source_image.stem)
            _, width, height = frame_info[frame]

            destination_name = (
                f"{sequence}__{source_image.name}"
            )

            frame_to_image_id[frame] = image_id

            coco_images.append(
                {
                    "id": image_id,
                    "file_name": destination_name,
                    "width": width,
                    "height": height,
                }
            )

            if not audit_only:
                shutil.copy2(
                    source_image,
                    output_image_dir / destination_name,
                )

            image_id += 1

        with annotation_file.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, raw_line in enumerate(
                handle,
                start=1,
            ):
                line = raw_line.strip()

                if not line:
                    continue

                global_stats["raw_rows"] += 1

                row = parse_row(
                    annotation_file,
                    line_number,
                    line,
                )

                if not (
                    row["score"] == 1
                    and row["category"] in RAW_TO_COCO
                ):
                    continue

                global_stats["eligible"] += 1

                if row["track_id"] == 0:
                    global_stats["track_id_zero"] += 1

                frame = row["frame"]

                if frame not in frame_info:
                    raise RuntimeError(
                        f"{annotation_file}:{line_number}: "
                        f"missing frame {frame}"
                    )

                _, image_width, image_height = (
                    frame_info[frame]
                )

                state, bbox = clip_box(
                    row["x"],
                    row["y"],
                    row["width"],
                    row["height"],
                    image_width,
                    image_height,
                )

                global_stats[state] += 1

                if bbox is None:
                    continue

                coco_category, _ = RAW_TO_COCO[
                    row["category"]
                ]

                class_counts[coco_category] += 1

                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id":
                            frame_to_image_id[frame],
                        "category_id":
                            coco_category,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )

                annotation_id += 1

    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories(),
    }, class_counts


def require(label, actual, expected):
    if actual != expected:
        raise RuntimeError(
            f"{label}: expected {expected}, got {actual}"
        )

    if isinstance(actual, int):
        print(
            f"{label:30s}: {actual:,}  PASS"
        )
    else:
        print(
            f"{label:30s}: {actual}  PASS"
        )


def validate_split(name, sequences, coco, counts):
    expected = EXPECTED[name]

    print(f"\n=== {name.upper()} validation ===")

    require(
        "Sequences",
        len(sequences),
        expected["sequences"],
    )
    require(
        "Images",
        len(coco["images"]),
        expected["images"],
    )
    require(
        "Annotations",
        len(coco["annotations"]),
        expected["annotations"],
    )

    for class_id in range(1, 6):
        require(
            f"Class {class_id}",
            counts[class_id],
            expected["classes"][class_id],
        )

    image_ids = [
        x["id"] for x in coco["images"]
    ]
    annotation_ids = [
        x["id"] for x in coco["annotations"]
    ]
    filenames = [
        x["file_name"] for x in coco["images"]
    ]

    require(
        "Unique image IDs",
        len(set(image_ids)),
        len(image_ids),
    )
    require(
        "Unique annotation IDs",
        len(set(annotation_ids)),
        len(annotation_ids),
    )
    require(
        "Unique filenames",
        len(set(filenames)),
        len(filenames),
    )


def validate_combined(
    train_sequences,
    val_sequences,
    train_coco,
    val_coco,
    stats,
):
    expected = EXPECTED["combined"]

    print("\n=== COMBINED validation ===")

    require(
        "Sequences",
        len(train_sequences) + len(val_sequences),
        expected["sequences"],
    )

    overlap = set(train_sequences) & set(val_sequences)

    if overlap:
        raise RuntimeError(
            "Train/validation sequence overlap."
        )

    print(
        "Train/validation overlap      : 0  PASS"
    )

    require(
        "Images",
        len(train_coco["images"])
        + len(val_coco["images"]),
        expected["images"],
    )
    require(
        "Annotations",
        len(train_coco["annotations"])
        + len(val_coco["annotations"]),
        expected["annotations"],
    )

    for key in [
        "raw_rows",
        "inside",
        "cross_boundary",
        "fully_outside",
        "invalid_geometry",
        "track_id_zero",
    ]:
        require(
            key,
            stats[key],
            expected[key],
        )

    geometry_total = sum(
        stats[key]
        for key in [
            "inside",
            "cross_boundary",
            "fully_outside",
            "invalid_geometry",
        ]
    )

    require(
        "Geometry accounting",
        geometry_total,
        stats["eligible"],
    )

    require(
        "Eligible annotations",
        stats["eligible"],
        expected["annotations"],
    )


def write_json(path, data):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def main():
    args = parse_args()

    project_root = args.project_root.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    print("=== VisDrone detector COCO reconstruction ===")
    print("Project :", project_root)
    print("Source  :", source_root)
    print("Output  :", output_root)
    print(
        "Mode    :",
        "AUDIT ONLY"
        if args.audit_only
        else "RECONSTRUCT",
    )

    if not (source_root / "sequences").is_dir():
        raise RuntimeError(
            "Source sequences directory missing."
        )

    if not (source_root / "annotations").is_dir():
        raise RuntimeError(
            "Source annotations directory missing."
        )

    split_root = project_root / "data" / "splits"

    train_sequences = read_manifest(
        split_root / "visdrone_detector_train.txt"
    )

    val_sequences = read_manifest(
        split_root / "visdrone_detector_val.txt"
    )

    if set(train_sequences) & set(val_sequences):
        raise RuntimeError(
            "Detector train/validation manifests overlap."
        )

    if not args.audit_only:
        if output_root.exists():
            if not args.force:
                raise RuntimeError(
                    f"Output already exists: {output_root}. "
                    "Use --force only if intentional."
                )

            shutil.rmtree(output_root)

        (output_root / "train2017").mkdir(
            parents=True
        )
        (output_root / "val2017").mkdir(
            parents=True
        )
        (output_root / "annotations").mkdir(
            parents=True
        )

    stats = Counter()

    print("\nProcessing frozen TRAIN split...")

    train_coco, train_counts = process_split(
        "train",
        train_sequences,
        source_root,
        output_root,
        args.audit_only,
        stats,
    )

    print("Processing frozen VAL split...")

    val_coco, val_counts = process_split(
        "val",
        val_sequences,
        source_root,
        output_root,
        args.audit_only,
        stats,
    )

    validate_split(
        "train",
        train_sequences,
        train_coco,
        train_counts,
    )

    validate_split(
        "val",
        val_sequences,
        val_coco,
        val_counts,
    )

    validate_combined(
        train_sequences,
        val_sequences,
        train_coco,
        val_coco,
        stats,
    )

    if args.audit_only:
        print(
            "\nSTATUS: PASS — source data reproduces "
            "the frozen detector COCO dataset exactly."
        )
        print("No output files were created.")
        return

    annotation_dir = output_root / "annotations"

    train_json = (
        annotation_dir / "instances_train2017.json"
    )
    val_json = (
        annotation_dir / "instances_val2017.json"
    )

    print("\nWriting COCO JSON...")

    write_json(train_json, train_coco)
    write_json(val_json, val_coco)

    print("\nCreated:", output_root)
    print(train_json)
    print(val_json)

    print(
        "\nSTATUS: PASS — detector dataset "
        "reconstructed and validated."
    )


if __name__ == "__main__":
    main()
