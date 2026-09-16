# UAR-MOT Experimental Environment

## Compute Environment

- Platform: Google Colab
- GPU: NVIDIA Tesla T4
- GPU memory: approximately 15 GB

## Core Software

- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- NumPy: 2.1.3
- OpenCV: 5.0.0

## ByteTrack

- Repository: FoundationVision/ByteTrack
- Commit: d1bf0191adff59bc8fcfeaa0b33d3d1642552a99

## ByteTrack Dependencies

- lap: 0.5.13
- filterpy: 1.4.5
- motmetrics: 1.4.0
- thop: 0.1.1-2209072238
- loguru: 0.7.3
- cython_bbox: 0.1.5

## Notes

ByteTrack core tracker imports were verified successfully:

- yolox.tracker.byte_tracker.BYTETracker
- yolox.tracker.matching

The upstream ByteTrack requirements file was not installed wholesale.
Only dependencies required for the tracking pipeline were installed in
the existing Google Colab environment to avoid unnecessarily replacing
the working PyTorch/CUDA stack.

Detector, detector checkpoint, preprocessing, tracking thresholds,
dataset splits, calibration protocol, and TrackEval settings will be
recorded separately before final evaluation.
