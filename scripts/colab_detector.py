#!/usr/bin/env python3
"""Colab orchestration only; the versioned experiment remains authoritative."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import apply_yolox_training_compat as compat

PROJECT = Path(__file__).resolve().parents[1]
BASE = "6916af887a28af48f5b282612728ade5176ebf90"
YOLOX = Path("/content/YOLOX-detector")
DATA = Path("/content/UAR-MOT_detector_data")
DRIVE = Path("/content/drive/MyDrive/UAR-MOT")
EXP = "yolox_m_visdrone_5class"
INIT = "661a5b1615caa300fc394d2571708cff62017f658f1032990fb5094040650ebe"
DATA_SPECS = {
    "train2017": (13067, 475877, "b7440b82335269b0d7fcddeb761ba24cc60debe8b1c3b92604c8deeda8aae6d2"),
    "val2017": (2104, 51776, "20f1445fdd4b225d4da7e9b735c5a87c2e7024dfc4207746a5d55533d8bdd370"),
}
# Keep Colab's existing torch/torchvision/CUDA stack. No tracker-only packages.
DEPENDENCIES = {
    "cv2": "opencv-python", "numpy": "numpy", "PIL": "Pillow",
    "loguru": "loguru", "tqdm": "tqdm", "thop": "thop",
    "tabulate": "tabulate", "tensorboard": "tensorboard",
    "pycocotools": "pycocotools", "psutil": "psutil",
    "matplotlib": "matplotlib", "ninja": "ninja",
    "setuptools": "setuptools", "packaging": "packaging",
}


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def frozen_project():
    """Refuse changes to scientific inputs relative to the verified baseline."""
    paths = ["configs/detector_training.yaml", "exps/" + EXP + ".py",
             "scripts/prepare_visdrone_detector_coco.py",
             "scripts/bootstrap_yolox_detector.py",
             "scripts/apply_yolox_training_compat.py"]
    paths += [p.relative_to(PROJECT).as_posix()
              for folder in ("data/splits", "patches")
              for p in sorted((PROJECT / folder).iterdir()) if p.is_file()]
    for name in paths:
        original = subprocess.check_output(["git", "-C", str(PROJECT), "show", f"{BASE}:{name}"])
        actual = (PROJECT / name).read_bytes()
        if actual.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
            raise RuntimeError(f"Frozen input differs from baseline: {name}")


def patches():
    return tuple(PROJECT / "patches" / Path(s["default_path"]).name
                 for s in compat.PATCH_SPECS)


def verify_source():
    if compat.git_value(YOLOX, "rev-parse", "HEAD") != compat.EXPECTED_YOLOX_COMMIT:
        raise RuntimeError("YOLOX commit mismatch")
    compat.verify_patch_files(patches())
    compat.verify_exact_patched_state(YOLOX, patches())
    # The original verifier checks unstaged changes; also reject staged changes.
    if compat.git_value(YOLOX, "diff", "--cached", "--name-only"):
        raise RuntimeError("Unexpected staged YOLOX changes")
    if compat.git_value(YOLOX, "ls-files", "--others", "--exclude-standard"):
        raise RuntimeError("Unexpected untracked YOLOX files")


def verify_data(root=DATA):
    for split, (images, annotations, digest) in DATA_SPECS.items():
        path = root / "annotations" / f"instances_{split}.json"
        if compat.sha256(path) != digest:
            raise RuntimeError(f"Annotation SHA-256 mismatch: {path}")
        coco = json.loads(path.read_text(encoding="utf-8"))
        if (len(coco["images"]), len(coco["annotations"])) != (images, annotations):
            raise RuntimeError(f"COCO counts mismatch: {split}")
        for image in coco["images"]:
            file = root / split / image["file_name"]
            if not file.is_file() or file.stat().st_size == 0:
                raise RuntimeError(f"Missing/empty image: {file}")
        if sum(p.is_file() for p in (root / split).iterdir()) != images:
            raise RuntimeError(f"Image directory count mismatch: {split}")


def verify_init():
    checkpoint = DRIVE / "checkpoints" / "yolox_m_visdrone_5class_init_seed42.pth"
    if compat.sha256(checkpoint) != INIT:
        raise RuntimeError(f"Initialization SHA-256 mismatch: {checkpoint}")
    return checkpoint


def environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(YOLOX) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def runtime_check(gpu=False):
    # Import the actual training/evaluation path; do not construct a model or train.
    code = """
import torch, torchvision
from yolox.core import Trainer
from yolox.exp import get_exp
from yolox.evaluators import COCOEvaluator
from yolox import _C
from yolox.layers import COCOeval_opt
import cv2, numpy, PIL, loguru, tqdm, thop, tabulate, tensorboard, pycocotools, psutil, matplotlib
assert hasattr(torch.amp, 'GradScaler'), 'Native torch.amp.GradScaler required'
exp = get_exp(EXPERIMENT, None)
assert exp.seed == 42 and exp.max_epoch == 100
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Training imports and compiled COCO evaluator: PASS (no training launched)')
if REQUIRE_GPU:
    assert torch.cuda.is_available(), 'Select a Colab GPU runtime first'
""".replace("EXPERIMENT", repr(str(PROJECT / "exps" / (EXP + ".py"))))
    code = code.replace("REQUIRE_GPU", repr(gpu))
    run([sys.executable, "-c", code], cwd=YOLOX, env=environment(), timeout=60)


def install_dependencies():
    # Fail before pip if Colab's core stack is absent or broken.
    run([sys.executable, "-c", "import torch, torchvision; assert hasattr(torch.amp, 'GradScaler')"])
    missing = [package for module, package in DEPENDENCIES.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        from importlib.metadata import version
        with tempfile.TemporaryDirectory() as temporary:
            constraints = Path(temporary) / "constraints.txt"
            constraints.write_text("\n".join(f"{p}=={version(p)}" for p in ("torch", "torchvision", "numpy")), encoding="utf-8")
            run([sys.executable, "-m", "pip", "install", "-c", constraints, *missing])
    # Upstream has a standard COCO fallback, but the validated backend is not
    # recorded. Retain the existing compiled-backend requirement rather than
    # silently choosing a different evaluation implementation.
    probe = subprocess.run([sys.executable, "-c", "import torch; from yolox import _C"],
                           cwd=YOLOX, env=environment(), capture_output=True)
    if probe.returncode:
        run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=YOLOX, env=environment())
    run([sys.executable, "-m", "pip", "freeze"])


def require_colab():
    if sys.platform != "linux" or not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Run setup/train in Linux Colab after mounting Google Drive")
    if PROJECT != Path("/content/UAR-MOT"):
        raise RuntimeError("Expected repository at /content/UAR-MOT")


def setup(verify_only=False):
    frozen_project()
    verify_init()
    if verify_only:
        verify_source()
        verify_data()
        return
    if not YOLOX.exists():
        run([sys.executable, PROJECT / "scripts/bootstrap_yolox_detector.py"])
    command = [sys.executable, PROJECT / "scripts/apply_yolox_training_compat.py"]
    for flag, patch in zip(("--training-patch", "--ema-patch", "--native-amp-patch"), patches()):
        command.extend([flag, patch])
    run(command)
    verify_source()
    install_dependencies()
    runtime_check()
    if not DATA.exists():
        run([sys.executable, PROJECT / "scripts/prepare_visdrone_detector_coco.py"])
    # Existing incomplete assets are never silently deleted or rebuilt.
    verify_data()


def training_command(checkpoint):
    return [sys.executable, "-u", str(PROJECT / "scripts/yolox_train_entry.py"),
            "-f", str(PROJECT / "exps" / (EXP + ".py")),
            "-d", "1", "-b", "8", "--fp16", "-c", str(checkpoint),
            "output_dir", str(DRIVE / "outputs/detector")]


def train():
    setup(verify_only=True)
    runtime_check(gpu=True)
    output = DRIVE / "outputs/detector" / EXP
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Training output is nonempty; refusing to overwrite: {output}")
    command = training_command(verify_init())
    print("Starting frozen 100-epoch training; verbose output goes to Drive.", flush=True)
    print("Command:", json.dumps(command), flush=True)
    # Foreground execution preserves signals/exit status. No automatic resume:
    # upstream does not restore all RNG/data-loader states across interruptions.
    run(command, cwd=YOLOX, env=environment())


def status():
    checks = (("Frozen inputs", frozen_project), ("Patched YOLOX", verify_source),
              ("COCO hashes/counts/images", verify_data), ("Initialization", verify_init))
    good = True
    for label, check in checks:
        try:
            # Suppress verbose patch checks for compact notebook output.
            with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
                check()
            print(label + ": PASS")
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            good = False
            print(f"{label}: NOT READY ({exc})")
    print("Asset verification:", "PASS" if good else "FAIL")
    try:
        result = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                     capture_output=True, text=True)
        print("GPU:", result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        print("GPU: unavailable")
    # Capture import diagnostics separately from asset verification.
    try:
        readiness = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "readiness", "--require-gpu"],
            capture_output=True, text=True, timeout=75,
        )
        ready = readiness.returncode == 0
        detail = (readiness.stderr or readiness.stdout).strip().splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        ready, detail = False, [str(exc)]
    print("Training readiness (imports, compiled evaluator, CUDA):", "PASS" if ready else "FAIL")
    if not ready:
        print("  " + (detail[-1] if detail else "Readiness probe failed"))
    output = DRIVE / "outputs/detector" / EXP
    print("Checkpoints / YOLOX logs / TensorBoard:", output)
    for path in sorted(output.glob("*.pth")):
        stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        print(f"  {path.name}: {path.stat().st_size:,} bytes; modified {stamp}")
    logs = sorted((DRIVE / "logs/detector").glob("*.log"))
    print("Latest launcher log:", logs[-1] if logs else "none")
    if logs:
        # Seek only the tail; training logs may be large.
        with logs[-1].open("rb") as handle:
            handle.seek(max(0, logs[-1].stat().st_size - 4096))
            print("\n".join(handle.read().decode("utf-8", errors="replace").splitlines()[-5:]))
    print("Checkpoint files are progress artifacts; status does not deserialize or rank them.")
    return 0 if good and ready else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "train", "status", "readiness"))
    parser.add_argument("--verify-only", action="store_true", help="Read-only setup audit; no installs, patches, or dataset writes")
    parser.add_argument("--require-gpu", action="store_true", help="Require CUDA in the readiness probe")
    args = parser.parse_args()
    if args.verify_only and args.command != "setup":
        parser.error("--verify-only is valid only with setup")
    if args.require_gpu and args.command != "readiness":
        parser.error("--require-gpu is valid only with readiness")
    if args.command == "readiness":
        runtime_check(gpu=args.require_gpu)
        return 0
    if args.command == "status":
        return status()
    require_colab()
    if args.verify_only:
        setup(True)
        print("Asset verification: PASS (no assets changed; training readiness not checked)")
        return 0
    folder = DRIVE / "logs/detector"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + args.command + ".log")
    print("Verbose log:", log, flush=True)
    # Re-enter in a child so native/subprocess output and tracebacks share one log.
    with log.open("x", encoding="utf-8") as handle:
        command = [sys.executable, "-u", "-c",
                   "import colab_detector as c; " + ("c.setup()" if args.command == "setup" else "c.train()")]
        result = subprocess.run(command, cwd=PROJECT / "scripts", stdout=handle, stderr=subprocess.STDOUT)
    print(f"{args.command}: {'PASS' if result.returncode == 0 else 'FAILED'}; log: {log}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
