# PulseGuard-FL-Agentic-Federated-Multimodal-Deep-Learning-
PulseGuard-FL is an agentic federated multimodal learning framework for wearable activity recognition that jointly handles missing-sensor recovery, client scheduling, uncertainty calibration, and selective inference. It improves robustness, reliability, and participation fairness under varying sensor availability.

## Reproducible experimental structure:

- Public datasets: MHEALTH and UCI Human Activity Recognition Using Smartphones.
- Five subject-disjoint folds.
- Per fold: one test fold, the next fold for calibration, remaining three folds for federated training.
- Seeds: 17, 31, 47.
- Missing-modality probabilities: 0, 0.25, 0.50, 0.75 per modality.
- If all modalities are dropped for an example, one randomly chosen modality is retained.
- Six methods: FedAvg, FedProx, MaskDrop-FL, Prototype-FL, Recovery-FL, PulseGuard-FL.
- Classifier: 16 global rounds, four local optimization steps per selected client, AdamW.
- Denoiser: six global rounds.
- FedProx coefficient: 0.003.
- Recovery router: direct + class-prototype + denoising candidates.
- Router: simplex grid step 0.25; patterns with fewer than eight calibration examples use the global mixture.
- Temperature scaling on calibration subjects.
- Acceptance threshold: calibration-confidence 20th percentile, targeting about 80% coverage.
- ECE uses 12 bins.
- Client scheduling uses uncertainty, representation novelty, staleness and simulated sensor reliability.
- Participation is constrained lexicographically by cumulative selection count.

## Dataset dimensions stated in the paper

### MHEALTH
- 10 subjects
- 12 activities
- 2,670 non-overlapping windows in the paper's processed artifact
- 276 standardized engineered features
- 3 modalities:
  - chest/ECG: 60 features
  - ankle: 108 features
  - wrist: 108 features

### UCI-HAR
- 30 subjects
- 6 activities
- 10,299 windows
- 72 engineered features in the paper's local processed artifact
- 2 modalities:
  - accelerometer: 36 features
  - gyroscope: 36 features

## Critical reproducibility limitation

- the exact MHEALTH feature-engineering/window construction used to obtain exactly 2,670 × 276;
- the exact 72-feature UCI-HAR selection/construction;
- the exact numerical client-score weights in Eq. (15);
- the client selection budget Q;
- exact hidden-layer widths/latent dimension;
- learning rates, batch sizes, weight decay, and some reliability-simulation details.


## Getting the public datasets

UCI IDs:
- MHEALTH: 319
- Human Activity Recognition Using Smartphones: 240
