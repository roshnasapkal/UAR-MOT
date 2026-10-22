# UAR-MOT
Official implementation of Uncertainty-Guided Association Recovery for UAV Multi-Object Tracking (UAR-MOT).

## Frozen detector training in Google Colab

Choose **Runtime → Change runtime type → GPU**. These commands automate the
existing detector protocol; they do not tune the experiment. Run each code block
in a separate Colab cell. Nothing starts training until step 4.

1. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Place this repository (branch `major-revision`, including the automation changes)
at `/content/UAR-MOT`. For a fresh runtime, after these changes have been made
available on your remote branch:

```python
!git clone --branch major-revision https://github.com/roshnasapkal/UAR-MOT.git /content/UAR-MOT
```

Before setup, Drive must already contain the extracted **VisDrone2019-MOT-train**
dataset at
`MyDrive/UAR-MOT/datasets/VisDrone/VisDrone2019-MOT-train/`
with `sequences/<sequence>/<frame>.jpg` and `annotations/<sequence>.txt`, and the
verified initialization checkpoint at
`MyDrive/UAR-MOT/checkpoints/yolox_m_visdrone_5class_init_seed42.pth`.
The automation does not download datasets or weights. Do not substitute official
VisDrone validation/test data. If the automation is only local, clone the repository
in Colab, then upload only `.gitattributes`, `README.md`, `tests/test_colab_detector.py`,
`scripts/colab_detector.py` and `scripts/yolox_train_entry.py` into that clone.
Keep the frozen patches from the Linux clone; do not upload Windows-converted patches.

The rule `patches/*.patch text eol=lf` in `.gitattributes` preserves LF patch bytes
on future Git checkouts, including Windows. Git already stores these frozen patches
as LF; system `core.autocrlf=true` caused this Windows checkout to contain CRLF.
Adding the rule does not rewrite existing working files. Do not edit patch contents,
normalize them in the verifier, or change expected hashes. Their frozen SHA-256 values are:

| Patch | SHA-256 |
| --- | --- |
| `yolox_0.1.0_visdrone_training_compat.patch` | `8f9e20b3a10ad74b02c56099f4cad95862f0cc83df58ea639924629dc0b4b663` |
| `yolox_0.1.0_optional_apex_ema_compat.patch` | `6cc716a29613a07d80b67ca64914659fcb8d113dc023f74084b277a41410f7f5` |
| `yolox_0.1.0_native_amp_trainer_compat.patch` | `c3d12431346d41d63de908749bac0cd2e05e3311dd8ae0ceb21e48a30ab50f7e` |

3. Prepare the runtime (one command):

```python
!python /content/UAR-MOT/scripts/colab_detector.py setup
```

This restores the pinned YOLOX commit, applies only the three approved patches,
verifies all four patched source hashes, installs missing training dependencies,
and builds the upstream COCO evaluation extension. Colab's existing PyTorch,
torchvision and NumPy versions are preserved. The complete installed package list
is recorded in the setup log; see [ENVIRONMENT.md](ENVIRONMENT.md) for the recorded
research environment. Setup fails on incompatible imports rather than replacing
the CUDA stack. It reconstructs the dataset using the existing frozen manifests
and converter, then checks both exact JSON hashes, counts and image presence.
Already prepared datasets are audited and reused.
Image auditing checks file presence, nonzero size and counts; it does not decode
every image or establish full image-content integrity. Annotation JSON hashes are
verified exactly.

The pinned evaluator supports an upstream standard COCO fallback, so extension
compilation is not intrinsically required by upstream. This automation retains its
compiled evaluator requirement because the validated backend was not recorded;
it does not select a fallback or change checkpoint-selection metrics. Compilation
against the current Colab toolchain remains an untested compatibility risk.

To **only verify** existing assets, without installation, patching or reconstruction:

```python
!python /content/UAR-MOT/scripts/colab_detector.py setup --verify-only
```

Asset verification also checks the initialization hash. It does not check Python
imports, extension compatibility or GPU availability. Run the separate lightweight
readiness probe for those checks, without constructing a model or starting training:

```python
!python /content/UAR-MOT/scripts/colab_detector.py readiness
```

This imports the training dependencies and compiled COCO evaluator, loads the
experiment settings, and reports CUDA availability. To require CUDA as well:

```python
!python /content/UAR-MOT/scripts/colab_detector.py readiness --require-gpu
```

Neither readiness command installs dependencies, builds extensions or prepares data.
Readiness does not replace asset verification or a real GPU execution test.
Asset verification does not need the original Drive dataset once reconstruction
is complete.
Missing or inconsistent assets cause a failure. An incomplete dataset or unexpected
YOLOX tree is never automatically deleted: inspect the setup log, then explicitly
rename/remove only the affected ephemeral `/content` directory and rerun setup.
Never remove the Drive source data or persistent training outputs to repair setup.

4. Start training **only when ready**:

```python
!python /content/UAR-MOT/scripts/colab_detector.py train
```

The launcher rechecks assets and CUDA availability before invoking the unchanged
experiment with one GPU, batch 8 and FP16. All other scientific settings come from
the frozen experiment and pinned patched source: seed 42, 100 epochs, validation
interval 10, and checkpoint selection by detector-validation COCO AP@[0.50:0.95].
YOLOX's existing no-augmentation schedule may validate every epoch in its final
phase; the launcher preserves that behavior.

Verbose stdout/stderr is saved to
`MyDrive/UAR-MOT/logs/detector/<UTC timestamp>_train.log`.
YOLOX checkpoints (`latest_ckpt.pth`, `best_ckpt.pth`, and schedule checkpoints),
training logs and TensorBoard files are written directly under
`MyDrive/UAR-MOT/outputs/detector/yolox_m_visdrone_5class/`.
Notebook output stays compact; open the Drive log to inspect detailed progress.
The launcher runs in the foreground. A small entry adapter reuses the pinned CLI,
experiment loading and launch routine but invokes the training function underneath
Loguru's exception-suppressing decorator, so exceptions propagate as failures. It
does not alter upstream files, training logic, seeds or evaluation behavior.
Do not launch two training cells at once.

5. Check progress from another cell or a reconnected runtime:

```python
!python /content/UAR-MOT/scripts/colab_detector.py status
```

Status reports asset verification separately from training readiness (Python imports,
compiled evaluator and CUDA), plus checkpoint sizes/modification times,
log paths and the latest log tail. It does not load or score checkpoint files;
use the log's epoch/AP records to interpret progress. It returns nonzero when asset
checks or training readiness fail. A complete image-presence audit can take a short time.

Colab runtime resets erase `/content`; mount Drive and rerun setup after reconnecting
to a fresh runtime. Drive logs/checkpoints persist, but abrupt interruption can leave
an incomplete checkpoint. The launcher refuses nonempty training outputs and does
not automatically resume: upstream checkpoints do not restore every RNG/data-loader
state, so exact interrupted-run reproducibility is not claimed. Preserve prior
outputs and decide how to handle an interrupted experiment before starting again.

The wrapper checks scientific inputs against baseline commit
`6916af887a28af48f5b282612728ade5176ebf90`. Keep that Git history available.
No frozen configuration, split, calibration rule or evaluation rule is changed.
This automation has CPU-only unit coverage; it has not been GPU-tested in Colab.

Local lightweight tests (no GPU packages or dataset required):

```text
python -m unittest discover -s tests -v
```
