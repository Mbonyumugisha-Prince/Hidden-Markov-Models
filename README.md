# Modeling Human Activity States Using Hidden Markov Models

This project recognizes **standing, walking, jumping, and still** from smartphone accelerometer and gyroscope recordings. A Gaussian Hidden Markov Model (HMM) uses features extracted from short sensor windows to infer the activity sequence over time.

The [project notebook](notebook/notebook.ipynb) contains data preprocessing, feature extraction, an HMM implemented from scratch, training, visualizations, and evaluation on two unseen recordings. The recordings were collected using Sensor Logger on an iPhone 13.

## Project structure

```text
Hidden-Markov-Models/
├── README.md
├── notebook/
│   └── notebook.ipynb     # Complete analysis and HMM implementation
├── zips/
│   ├── standing/          # Raw training recordings from Sensor Logger
│   ├── walking/
│   ├── jumping/
│   └── still/
├── dataset/
│   ├── standing/          # Resampled and labelled training CSV files
│   ├── walking/
│   ├── jumping/
│   └── still/
└── test_zips/             # Two unseen recordings containing all four activities
```

## Setup and execution

Use Python 3 with NumPy, pandas, SciPy, Matplotlib, and seaborn. JupyterLab provides an interface for running the notebook.

From the project root, create and activate a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy matplotlib seaborn jupyterlab ipykernel
python -m jupyterlab notebook/notebook.ipynb
```

Alternatively, open `notebook/notebook.ipynb` in VS Code with its Python and Jupyter extensions and select the `.venv` Python interpreter as the notebook kernel.

**Restart the kernel and run all cells from top to bottom.** Training creates `act_idx`, `hmm`, and `STATE2ACT`, which the evaluation cells require. The notebook supports execution with the working directory set to either the project root or `notebook/`.

Keep the raw ZIP files in their existing folders. The notebook reads them directly and regenerates the corresponding CSV files in `dataset/` when its preprocessing cell runs. Each recording archive must contain `Accelerometer.csv` and `Gyroscope.csv`.

## Dataset

The training dataset contains 72 recordings:

| Activity | Recordings | Total duration before trimming (seconds) |
|---|---:|---:|
| Standing | 20 | 213.9 |
| Walking | 12 | 125.7 |
| Jumping | 20 | 212.6 |
| Still | 20 | 212.8 |

Each exported CSV contains:

- `timestamp_ns`: sensor timestamp in nanoseconds.
- `time_s`: elapsed time in seconds relative to the aligned recording start.
- `acc_x`, `acc_y`, `acc_z`: accelerometer channels.
- `gyro_x`, `gyro_y`, `gyro_z`: gyroscope channels.
- `label`: activity name.
- `recording`: recording identifier, such as `still_04`.

Sensor timestamps are sorted and duplicates removed before both sensors are interpolated onto a shared **100 Hz** time grid. Measured training accelerometer rates average approximately 100.4–100.5 Hz across the activities.

The first and last second of each training recording are excluded during feature extraction to reduce start/stop handling artifacts. **The exported CSVs retain those seconds:** trimming currently applies only to the data used for training features. Saved recording durations range from about 9.65 to 11.50 seconds.

## Features and preprocessing

The notebook uses **1-second windows (100 samples)** with a **0.5-second step (50% overlap)**, producing 1,137 training windows from the current recordings.

Each window has 13 features:

| Domain | Features | Count |
|---|---|---:|
| Time | Acceleration magnitude mean and standard deviation | 2 |
| Time | Accelerometer and gyroscope signal magnitude area | 2 |
| Time | Gyroscope magnitude standard deviation | 1 |
| Time | Accelerometer axis correlations: xy, xz, yz | 3 |
| Frequency | Dominant frequencies of acceleration and gyroscope magnitudes | 2 |
| Frequency | Spectral energies of acceleration and gyroscope magnitudes | 2 |
| Frequency | Acceleration spectral entropy | 1 |

Frequency features use the FFT of centered, Hann-windowed magnitude signals. Selected magnitude, variability, and energy features receive a log transform. All features are then z-score normalized using the training mean and standard deviation. The same training statistics are applied to unseen recordings.

## HMM implementation

The model has four hidden states, one for each activity, and a 13-dimensional observation vector per window. It learns a transition matrix, initial state probabilities, and Gaussian emission means and diagonal variances.

Training constructs 60 synthetic sessions by joining 4–7 existing training recordings per session. Activities are sampled with equal probability before choosing a recording, reducing the effect of having fewer walking recordings. These sessions reuse collected recordings; they are not additional independent recordings.

Labelled training statistics initialize the model. **Baum–Welch**, using forward–backward calculations in log space, refines its parameters. Training stops when the absolute change in total log-likelihood falls below `0.001`, with a safety limit of 300 iterations. The current saved run converged after 21 iterations.

**Viterbi decoding** finds the most probable state sequence using both sensor features and transition probabilities. Learned states are matched to activity labels using training data only. The HMM algorithms are implemented with NumPy, with SciPy utilities for log-sum-exp and state-to-activity assignment.

## Evaluation and current results

Two separate recordings in `test_zips/` are held out from normalization fitting, training, and state matching. Each contains all four activities. Manual activity intervals are defined in `TEST_SEGMENTS` in chronological file order. Only windows fully inside a labelled interval contribute to the metrics; windows crossing boundaries or unlabelled gaps are excluded.

The current saved evaluation reports **94.4% overall accuracy on 179 scored windows**:

| Activity | Scored windows | Sensitivity | Specificity | One-versus-rest accuracy |
|---|---:|---:|---:|---:|
| Standing | 48 | 0.979 | 0.969 | 0.972 |
| Walking | 65 | 1.000 | 0.947 | 0.966 |
| Jumping | 32 | 0.844 | 1.000 | 0.972 |
| Still | 34 | 0.882 | 1.000 | 0.978 |

Sensitivity measures the proportion of an activity's windows correctly detected. Specificity measures the proportion of other activities' windows correctly rejected. Per-activity accuracy treats each activity against all others; overall accuracy measures correct activity predictions across all scored windows. Window counts are not counts of independent recordings.

These values reflect the notebook's saved numerical output. Its written reflection still contains older figures, including 95.5% overall accuracy, and needs to be reconciled with that output.

## Visualizations

Running the notebook produces sample accelerometer and gyroscope plots, a training convergence curve, a transition matrix heatmap, a heatmap of Gaussian emission means, labelled test signals, a test confusion matrix, and predicted activity sequences alongside their ground-truth labels.

## Limitations and possible improvements

The data comes from one participant and one phone. Two test recordings with overlapping windows provide limited evidence of generalization to other people, devices, or phone positions. Manual test intervals exclude transition boundaries from scoring, so the reported accuracy does not measure performance across every transition.

Transitions learned from concatenated recordings reflect the synthetic session construction rather than measured everyday activity patterns. Further work could include more participants and phone positions, continuous recordings with natural transitions, additional features, and richer emission models.
