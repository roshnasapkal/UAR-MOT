# VisDrone MOT Split Protocol for UAR-MOT

## Purpose

This document defines the frozen VisDrone2019-MOT training-data partition
used for UAR-MOT uncertainty calibration and parameter development.

The partition was defined before UAR-MOT uncertainty calibration,
recovery-parameter tuning, or final evaluation.

## Source Dataset

Dataset: VisDrone2019-MOT training set

Number of training sequences: 56

Master sequence manifest:

data/splits/visdrone_mot_train_all.txt

Sequence-level statistics:

data/splits/visdrone_mot_train_sequence_stats.csv

## Partition

The 56 official training sequences are partitioned into three
non-overlapping subsets:

- Calibration: 12 sequences
- Development: 12 sequences
- Remaining training: 32 sequences

Sequence lists:

- data/splits/visdrone_mot_calibration.txt
- data/splits/visdrone_mot_development.txt
- data/splits/visdrone_mot_train_remaining.txt

The three subsets are mutually exclusive and their union contains all
56 training sequences.

## Permitted Use

### Calibration subset

The calibration subset is used only to estimate and calibrate uncertainty
quantities required by UAR-MOT, including the detection-localization,
motion-prediction, and camera-motion uncertainty components.

It must not be used for final performance reporting.

### Development subset

The development subset is used for model-selection decisions and
UAR-MOT parameter selection, including recovery/stability thresholds
such as tau_psi, tau_G, kappa_a, and other tunable recovery parameters.

It must not be used for final performance reporting.

### Remaining-training subset

The remaining 32 sequences are reserved for training-related operations
if required by the experimental pipeline.

They are not used to select UAR-MOT uncertainty-calibration parameters
or recovery thresholds.

### Official VisDrone validation/evaluation data

Official VisDrone validation/evaluation sequences are kept outside this
training-set partition and must not be used for uncertainty calibration
or UAR-MOT parameter selection.

They are reserved for controlled evaluation after the UAR-MOT protocol
and parameters have been frozen.

## Deterministic Split Construction

The split was generated using only sequence-level characteristics from
the official VisDrone2019-MOT training annotations.

No UAR-MOT tracking accuracy, stability margin, recovery success,
validation-set result, or test-set result was used to construct the
partition.

Four sequence characteristics were used:

1. objects_per_frame
2. median_bbox_area
3. occlusion_rate
4. frames

The following variables were transformed with log1p before
standardization:

- objects_per_frame
- median_bbox_area
- frames

All four variables were then standardized using StandardScaler.

K-means clustering was performed with:

- n_clusters = 12
- random_state = 42
- n_init = 50

Within each cluster, sequences were ordered by Euclidean distance to
the corresponding cluster center, with sequence name used as the
deterministic tie-breaker.

Selection rule:

- closest sequence to cluster center -> calibration
- second-closest sequence -> development
- all other sequences -> remaining training

This produces exactly one calibration and one development sequence
from each of the 12 clusters.

## Split Audit

The resulting partition contains:

- Calibration: 12 sequences
- Development: 12 sequences
- Remaining training: 32 sequences
- Pairwise overlap: 0 sequences
- Total unique sequences: 56

Before freezing the split, distributional coverage was checked using
sequence length, object density, bounding-box scale, occlusion,
truncation, and representation of the five VisDrone MOT evaluation
classes:

- pedestrian
- car
- van
- truck
- bus

Both calibration and development subsets contain examples of all five
MOT evaluation classes across multiple sequences.

No subsequent manual sequence substitutions are permitted based on
UAR-MOT experimental performance.

## Leakage-Control Rule

After this protocol is committed, the calibration/development sequence
membership is treated as frozen.

Final evaluation results must not be used to revise:

- sequence membership,
- uncertainty calibration,
- tau_psi,
- tau_G,
- kappa_a,
- recovery-selection rules,
- or other UAR-MOT tunable parameters.

Any later protocol change must be explicitly versioned and documented
before rerunning final evaluation.
