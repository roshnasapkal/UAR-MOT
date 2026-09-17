#!/usr/bin/env python3

"""
Safely apply the UAR-MOT YOLOX 0.1.0 detector-training
compatibility patch.

The patch makes exactly two intended training changes:

1. Effective MosaicDetection TrainTransform label capacity:
   120 -> 768.

2. Run the SimOTA BCE probability/loss calculation explicitly
   in FP32 with CUDA autocast disabled for compatibility with
   modern PyTorch AMP.

This script verifies the pinned upstream commit, source hashes,
patch hash, patch applicability, and resulting source hashes.

It is intentionally idempotent:
- pristine expected checkout -> apply and verify;
- already correctly patched checkout -> verify and exit;
- any other source state -> refuse to continue.
"""

from pathlib import Path
import argparse
import hashlib
import subprocess
import sys


EXPECTED_YOLOX_COMMIT = (
    "e1052df71842031413f6030723c3607b839c80ce"
)

EXPECTED_PATCH_SHA256 = (
    "8f9e20b3a10ad74b02c56099f4cad95862f0cc83df58ea639924629dc0b4b663"
)

PRISTINE_HASHES = {
    "yolox/exp/yolox_base.py":
        "768de1ab024e9ec682b2a03b4204c59aee36baa0e7908049900b277eb06f85c1",
    "yolox/models/yolo_head.py":
        "1666ece85236b71ae5e13bdc416f04483ececfcbef655f048b0fcc5ffdb99c09",
}

PATCHED_REQUIRED_TEXT = {
    "yolox/exp/yolox_base.py": [
        "max_labels=768,",
    ],
    "yolox/models/yolo_head.py": [
        "with torch.cuda.amp.autocast(enabled=False):",
        "obj_preds_.float()",
        "F.binary_cross_entropy",
    ],
}

PATCHED_FORBIDDEN_TEXT = {
    "yolox/exp/yolox_base.py": [
        "                max_labels=120,",
    ],
}


def run(cmd, cwd=None, check=True):
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def git_value(root, *args):
    return run(
        ["git", "-C", str(root), *args]
    ).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(yolox_root):
    result = {}
    for rel in PRISTINE_HASHES:
        path = yolox_root / rel
        if not path.is_file():
            raise RuntimeError(
                f"Expected source file missing: {path}"
            )
        result[rel] = sha256(path)
    return result


def verify_patched_content(yolox_root):
    for rel, fragments in PATCHED_REQUIRED_TEXT.items():
        text = (
            yolox_root / rel
        ).read_text(encoding="utf-8")

        for fragment in fragments:
            count = text.count(fragment)
            if count != 1:
                raise RuntimeError(
                    f"Expected exactly one patched fragment "
                    f"{fragment!r} in {rel}; found {count}."
                )

    for rel, fragments in PATCHED_FORBIDDEN_TEXT.items():
        text = (
            yolox_root / rel
        ).read_text(encoding="utf-8")

        for fragment in fragments:
            if fragment in text:
                raise RuntimeError(
                    f"Pristine fragment still present after "
                    f"patching in {rel}: {fragment!r}"
                )


def changed_files(yolox_root):
    # The approved compatibility patch modifies only
    # already-tracked YOLOX source files. Using Git's
    # name-only diff avoids fragile porcelain slicing.
    output = git_value(
        yolox_root,
        "diff",
        "--name-only",
    )

    return sorted(
        line.strip()
        for line in output.splitlines()
        if line.strip()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolox-root",
        default="/content/YOLOX-detector",
    )
    parser.add_argument(
        "--patch",
        default=(
            "/content/UAR-MOT/patches/"
            "yolox_0.1.0_visdrone_training_compat.patch"
        ),
    )

    args = parser.parse_args()

    yolox_root = Path(args.yolox_root).resolve()
    patch_file = Path(args.patch).resolve()

    print(
        "=== UAR-MOT YOLOX training compatibility ==="
    )
    print("YOLOX root:", yolox_root)
    print("Patch     :", patch_file)

    if not (yolox_root / ".git").exists():
        raise RuntimeError(
            "YOLOX root is not a Git checkout."
        )

    if not patch_file.is_file():
        raise RuntimeError(
            f"Patch file missing: {patch_file}"
        )

    head = git_value(
        yolox_root,
        "rev-parse",
        "HEAD",
    )

    print("YOLOX HEAD:", head)

    if head != EXPECTED_YOLOX_COMMIT:
        raise RuntimeError(
            "YOLOX commit does not match the pinned "
            "training checkout."
        )

    patch_sha = sha256(patch_file)
    print("Patch SHA256:", patch_sha)

    if patch_sha != EXPECTED_PATCH_SHA256:
        raise RuntimeError(
            "Compatibility patch hash mismatch."
        )

    before_hashes = file_hashes(yolox_root)
    files_before = changed_files(yolox_root)

    print("\nCurrent source hashes:")
    for rel, value in before_hashes.items():
        print(f"  {rel}: {value}")

    print(
        "Current changed files:",
        files_before if files_before else "none",
    )

    pristine = (
        before_hashes == PRISTINE_HASHES
        and files_before == []
    )

    expected_changed = sorted(
        PRISTINE_HASHES.keys()
    )

    # ------------------------------------------------
    # Already-patched state
    # ------------------------------------------------

    if not pristine:
        if files_before != expected_changed:
            raise RuntimeError(
                "YOLOX is neither pristine nor in the "
                "expected two-file patched state."
            )

        reverse_check = run(
            [
                "git",
                "-C",
                str(yolox_root),
                "apply",
                "--reverse",
                "--check",
                str(patch_file),
            ],
            check=False,
        )

        if reverse_check.returncode != 0:
            raise RuntimeError(
                "YOLOX has two changed files, but they "
                "do not exactly correspond to the "
                "approved compatibility patch."
            )

        verify_patched_content(yolox_root)

        print(
            "\nSTATUS: already correctly patched — "
            "verification PASS."
        )
        return 0

    # ------------------------------------------------
    # Pristine state: check then apply
    # ------------------------------------------------

    check_result = run(
        [
            "git",
            "-C",
            str(yolox_root),
            "apply",
            "--check",
            str(patch_file),
        ],
        check=False,
    )

    if check_result.returncode != 0:
        print(check_result.stdout)
        print(check_result.stderr, file=sys.stderr)
        raise RuntimeError(
            "Patch does not cleanly apply to pristine "
            "pinned YOLOX."
        )

    apply_result = run(
        [
            "git",
            "-C",
            str(yolox_root),
            "apply",
            str(patch_file),
        ],
        check=False,
    )

    if apply_result.returncode != 0:
        print(apply_result.stdout)
        print(apply_result.stderr, file=sys.stderr)
        raise RuntimeError(
            "Patch application failed."
        )

    files_after = changed_files(yolox_root)

    if files_after != expected_changed:
        raise RuntimeError(
            "Unexpected files changed after patching: "
            f"{files_after}"
        )

    reverse_check = run(
        [
            "git",
            "-C",
            str(yolox_root),
            "apply",
            "--reverse",
            "--check",
            str(patch_file),
        ],
        check=False,
    )

    if reverse_check.returncode != 0:
        raise RuntimeError(
            "Post-application reverse verification failed."
        )

    verify_patched_content(yolox_root)

    after_hashes = file_hashes(yolox_root)

    print("\nPatched source hashes:")
    for rel, value in after_hashes.items():
        print(f"  {rel}: {value}")

    print("Changed files:", files_after)

    print(
        "\nSTATUS: compatibility patch applied and "
        "verified successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
