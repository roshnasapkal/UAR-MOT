#!/usr/bin/env python3
"""
Restore and verify the exact upstream YOLOX checkout used for
UAR-MOT detector training.

This script intentionally:
- does not install Python packages;
- does not modify the upstream YOLOX source;
- does not prepare datasets;
- does not download detector weights.

It only restores/verifies the pinned upstream Git checkout.
"""

from pathlib import Path
import argparse
import shutil
import subprocess
import sys


YOLOX_REPOSITORY = (
    "https://github.com/Megvii-BaseDetection/YOLOX.git"
)

YOLOX_COMMIT = (
    "e1052df71842031413f6030723c3607b839c80ce"
)

DEFAULT_DESTINATION = Path("/content/YOLOX-detector")


def run(cmd, cwd=None, capture=True):
    """Run a command and fail immediately on error."""

    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=capture,
    )

    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        raise RuntimeError(
            "Command failed: " + " ".join(map(str, cmd))
        )

    return result


def git_output(destination, *args):
    """Return stripped stdout from git -C destination."""

    return run(
        [
            "git",
            "-C",
            str(destination),
            *args,
        ]
    ).stdout.strip()


def verify_checkout(destination):
    """Verify exact commit, required files, and clean tree."""

    actual_commit = git_output(
        destination,
        "rev-parse",
        "HEAD",
    )

    if actual_commit != YOLOX_COMMIT:
        raise RuntimeError(
            "YOLOX commit mismatch:\n"
            f"  expected: {YOLOX_COMMIT}\n"
            f"  actual:   {actual_commit}"
        )

    required = [
        destination / "yolox" / "__init__.py",
        destination
        / "yolox" / "data" / "datasets" / "coco.py",
        destination
        / "yolox" / "exp" / "yolox_base.py",
        destination
        / "yolox" / "models" / "yolo_head.py",
        destination / "tools" / "train.py",
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Required YOLOX files are missing:\n"
            + "\n".join(missing)
        )

    status = git_output(
        destination,
        "status",
        "--porcelain",
    )

    if status:
        raise RuntimeError(
            "YOLOX working tree is not clean:\n"
            + status
        )

    return actual_commit


def restore(destination, force=False):

    destination = destination.resolve()

    # --------------------------------------------------
    # Existing destination
    # --------------------------------------------------

    if destination.exists():

        git_dir = destination / ".git"

        if git_dir.exists() and not force:
            print(
                "Existing YOLOX checkout found; "
                "verifying without modifying it."
            )

            commit = verify_checkout(destination)

            print("YOLOX checkout already valid.")
            print("Commit:", commit)
            return

        if not force:
            raise RuntimeError(
                f"Destination already exists but cannot be "
                f"accepted safely: {destination}\n"
                "Use --force only if you intentionally want "
                "the ephemeral destination replaced."
            )

        print(
            "Removing existing destination:",
            destination,
        )

        shutil.rmtree(destination)

    # --------------------------------------------------
    # Clone official upstream
    # --------------------------------------------------

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Cloning official YOLOX...")
    print("Repository:", YOLOX_REPOSITORY)
    print("Destination:", destination)

    run(
        [
            "git",
            "clone",
            YOLOX_REPOSITORY,
            str(destination),
        ],
        capture=False,
    )

    # --------------------------------------------------
    # Checkout exact pinned commit
    # --------------------------------------------------

    run(
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--detach",
            YOLOX_COMMIT,
        ],
        capture=False,
    )

    commit = verify_checkout(destination)

    print()
    print("YOLOX checkout restored and verified.")
    print("Commit:", commit)
    print("Working tree: clean")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=(
            "Ephemeral YOLOX checkout destination "
            "(default: /content/YOLOX-detector)"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace an existing destination before "
            "restoring the pinned checkout."
        ),
    )

    args = parser.parse_args()

    restore(
        args.destination,
        force=args.force,
    )


if __name__ == "__main__":
    main()
