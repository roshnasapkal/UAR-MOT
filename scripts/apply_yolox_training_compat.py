#!/usr/bin/env python3

"""
Safely apply the UAR-MOT compatibility patches to the pinned
YOLOX 0.1.0 detector-training checkout.

The compatibility layer makes exactly three intended changes:

1. yolox/exp/yolox_base.py
   Effective MosaicDetection TrainTransform label capacity:
   120 -> 768.

2. yolox/models/yolo_head.py
   Run the SimOTA BCE probability/loss calculation explicitly
   in FP32 with CUDA autocast disabled for compatibility with
   modern PyTorch AMP.

3. yolox/utils/ema.py
   Make Apex optional in is_parallel(). Standard PyTorch
   DataParallel and DistributedDataParallel remain supported;
   Apex DDP is included when Apex is installed.

The script verifies:
- exact pinned upstream YOLOX commit;
- exact hashes of both approved patch files;
- exact pristine hashes of all three target source files;
- exact patched hashes of all three target source files;
- exact changed-file set;
- semantic fragments;
- reverse applicability of both patches.

It is intentionally idempotent:
- pristine expected checkout -> apply both patches and verify;
- exact combined patched checkout -> verify and exit;
- partial, altered, or otherwise unexpected state -> refuse.
"""

from pathlib import Path
import argparse
import hashlib
import subprocess
import sys


EXPECTED_YOLOX_COMMIT = (
    "e1052df71842031413f6030723c3607b839c80ce"
)


PATCH_SPECS = (
    {
        "name": "training compatibility",
        "default_path": (
            "/content/UAR-MOT/patches/"
            "yolox_0.1.0_visdrone_training_compat.patch"
        ),
        "sha256": (
            "8f9e20b3a10ad74b02c56099f4cad958"
            "62f0cc83df58ea639924629dc0b4b663"
        ),
    },
    {
        "name": "optional-Apex EMA compatibility",
        "default_path": (
            "/content/UAR-MOT/patches/"
            "yolox_0.1.0_optional_apex_ema_compat.patch"
        ),
        "sha256": (
            "6cc716a29613a07d80b67ca64914659f"
            "cb8d113dc023f74084b277a41410f7f5"
        ),
    },
)


PRISTINE_HASHES = {
    "yolox/exp/yolox_base.py": (
        "768de1ab024e9ec682b2a03b4204c59a"
        "ee36baa0e7908049900b277eb06f85c1"
    ),
    "yolox/models/yolo_head.py": (
        "1666ece85236b71ae5e13bdc416f04483"
        "ececfcbef655f048b0fcc5ffdb99c09"
    ),
    "yolox/utils/ema.py": (
        "3e82e19000b4ab0f88d2f94a4b425cc"
        "20323fba84e0700e885a5638d0ce3eebb"
    ),
}


PATCHED_HASHES = {
    "yolox/exp/yolox_base.py": (
        "95a1aa94938f39f42bb1449efd90aefa"
        "174aca23718cd09ce7b189055ee2c473"
    ),
    "yolox/models/yolo_head.py": (
        "c8e1b6335a3b1f472b578d556c7cb363"
        "585ac86029894ee18c8d434f12e073ea"
    ),
    "yolox/utils/ema.py": (
        "88692a29551ca24eed37c19d3986ad46"
        "f9b458cec378a9da35689aceb777563c"
    ),
}


PATCHED_REQUIRED_TEXT = {
    "yolox/exp/yolox_base.py": (
        "max_labels=768,",
    ),
    "yolox/models/yolo_head.py": (
        "with torch.cuda.amp.autocast(enabled=False):",
        "cls_preds_.float()",
        "obj_preds_.float()",
        "F.binary_cross_entropy",
    ),
    "yolox/utils/ema.py": (
        "nn.parallel.DataParallel",
        "nn.parallel.DistributedDataParallel",
        "try:\n        import apex",
        "except ImportError:",
        "apex.parallel.distributed.DistributedDataParallel",
    ),
}


PATCHED_FORBIDDEN_TEXT = {
    "yolox/exp/yolox_base.py": (
        "                max_labels=120,",
    ),
    "yolox/utils/ema.py": (
        '    import apex\n\n    parallel_type = (',
    ),
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


def source_hashes(yolox_root):
    result = {}

    for rel in PRISTINE_HASHES:
        path = yolox_root / rel

        if not path.is_file():
            raise RuntimeError(
                f"Expected source file missing: {path}"
            )

        result[rel] = sha256(path)

    return result


def changed_files(yolox_root):
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


def verify_patch_files(patches):
    for spec, path in zip(
        PATCH_SPECS,
        patches,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Patch file missing: {path}"
            )

        actual = sha256(path)

        print(
            f"{spec['name']} patch SHA256: {actual}"
        )

        if actual != spec["sha256"]:
            raise RuntimeError(
                f"{spec['name']} patch hash mismatch."
            )


def verify_patched_content(yolox_root):
    for rel, fragments in (
        PATCHED_REQUIRED_TEXT.items()
    ):
        text = (
            yolox_root / rel
        ).read_text(
            encoding="utf-8"
        )

        for fragment in fragments:
            count = text.count(fragment)

            if count != 1:
                raise RuntimeError(
                    f"Expected exactly one patched fragment "
                    f"{fragment!r} in {rel}; found {count}."
                )

    for rel, fragments in (
        PATCHED_FORBIDDEN_TEXT.items()
    ):
        text = (
            yolox_root / rel
        ).read_text(
            encoding="utf-8"
        )

        for fragment in fragments:
            if fragment in text:
                raise RuntimeError(
                    f"Forbidden pristine fragment remains "
                    f"in {rel}: {fragment!r}"
                )


def verify_reverse_checks(
    yolox_root,
    patches,
):
    for spec, path in zip(
        PATCH_SPECS,
        patches,
    ):
        result = run(
            [
                "git",
                "-C",
                str(yolox_root),
                "apply",
                "--reverse",
                "--check",
                str(path),
            ],
            check=False,
        )

        if result.returncode != 0:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(
                    result.stderr,
                    file=sys.stderr,
                )

            raise RuntimeError(
                f"{spec['name']} patch does not "
                f"reverse-check against patched state."
            )


def verify_exact_patched_state(
    yolox_root,
    patches,
):
    expected_changed = sorted(
        PATCHED_HASHES.keys()
    )

    files = changed_files(
        yolox_root
    )

    hashes = source_hashes(
        yolox_root
    )

    if files != expected_changed:
        raise RuntimeError(
            "Unexpected changed-file set in patched "
            f"state: {files}"
        )

    if hashes != PATCHED_HASHES:
        print("Expected patched hashes:")
        for rel, value in PATCHED_HASHES.items():
            print(f"  {rel}: {value}")

        print("Actual source hashes:")
        for rel, value in hashes.items():
            print(f"  {rel}: {value}")

        raise RuntimeError(
            "Patched source hashes do not exactly match "
            "the approved three-file state."
        )

    verify_patched_content(
        yolox_root
    )

    verify_reverse_checks(
        yolox_root,
        patches,
    )

    return hashes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--yolox-root",
        default="/content/YOLOX-detector",
    )

    parser.add_argument(
        "--training-patch",
        default=PATCH_SPECS[0]["default_path"],
    )

    parser.add_argument(
        "--ema-patch",
        default=PATCH_SPECS[1]["default_path"],
    )

    args = parser.parse_args()

    yolox_root = Path(
        args.yolox_root
    ).resolve()

    patches = (
        Path(
            args.training_patch
        ).resolve(),
        Path(
            args.ema_patch
        ).resolve(),
    )

    print(
        "=== UAR-MOT YOLOX training compatibility ==="
    )
    print("YOLOX root:", yolox_root)

    for spec, path in zip(
        PATCH_SPECS,
        patches,
    ):
        print(
            f"{spec['name']} patch:",
            path,
        )

    if not (
        yolox_root / ".git"
    ).exists():
        raise RuntimeError(
            "YOLOX root is not a Git checkout."
        )

    head = git_value(
        yolox_root,
        "rev-parse",
        "HEAD",
    )

    print("YOLOX HEAD:", head)

    if head != EXPECTED_YOLOX_COMMIT:
        raise RuntimeError(
            "YOLOX commit does not match the "
            "pinned detector-training checkout."
        )

    verify_patch_files(
        patches
    )

    before_hashes = source_hashes(
        yolox_root
    )

    files_before = changed_files(
        yolox_root
    )

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

    patched = (
        before_hashes == PATCHED_HASHES
        and files_before
        == sorted(PATCHED_HASHES.keys())
    )

    # ----------------------------------------------
    # Exact already-patched state
    # ----------------------------------------------

    if patched:
        verify_exact_patched_state(
            yolox_root,
            patches,
        )

        print(
            "\nSTATUS: already exactly patched — "
            "three-file verification PASS."
        )

        return 0

    # ----------------------------------------------
    # Reject every unexpected or partial state
    # ----------------------------------------------

    if not pristine:
        raise RuntimeError(
            "YOLOX is neither pristine nor in the "
            "exact approved combined patched state. "
            "Refusing partial or unknown source state."
        )

    # ----------------------------------------------
    # Pristine state:
    # pre-check BOTH patches before changing anything
    # ----------------------------------------------

    print(
        "\nPristine pinned checkout verified."
    )

    for spec, path in zip(
        PATCH_SPECS,
        patches,
    ):
        check_result = run(
            [
                "git",
                "-C",
                str(yolox_root),
                "apply",
                "--check",
                str(path),
            ],
            check=False,
        )

        if check_result.returncode != 0:
            if check_result.stdout:
                print(check_result.stdout)

            if check_result.stderr:
                print(
                    check_result.stderr,
                    file=sys.stderr,
                )

            raise RuntimeError(
                f"{spec['name']} patch does not "
                f"cleanly apply to pristine checkout."
            )

    # ----------------------------------------------
    # Apply both approved patches in fixed order
    # ----------------------------------------------

    for spec, path in zip(
        PATCH_SPECS,
        patches,
    ):
        result = run(
            [
                "git",
                "-C",
                str(yolox_root),
                "apply",
                str(path),
            ],
            check=False,
        )

        if result.returncode != 0:
            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(
                    result.stderr,
                    file=sys.stderr,
                )

            raise RuntimeError(
                f"Failed while applying "
                f"{spec['name']} patch."
            )

        print(
            f"Applied: {spec['name']}"
        )

    # ----------------------------------------------
    # Exact post-application verification
    # ----------------------------------------------

    after_hashes = verify_exact_patched_state(
        yolox_root,
        patches,
    )

    print("\nPatched source hashes:")

    for rel, value in after_hashes.items():
        print(f"  {rel}: {value}")

    print(
        "Changed files:",
        changed_files(yolox_root),
    )

    print(
        "\nSTATUS: both compatibility patches "
        "applied; exact three-file verification PASS."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
