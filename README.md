# FT11 FER2013 experiment

Compare an MLP, the original small CNN, and ResNet18 on one duplicate-aware split. The majority predictor is a reference. The notebook and command line call the same Python pipeline.

## Methods

Lecture 2 and Tutorials 1–2 support the MLP, convolution, pooling, cross-entropy, backpropagation, and gradient updates. Lecture 2, page 63, supports residual-network ideas. Exact model sizes and ResNet18 are FT11 choices. All models start with random weights. Data augmentation is outside this baseline comparison.

The CNN keeps its v1 design: valid 5×5 convolutions with channels 1→6→16→32, max pooling, a 64-unit dense layer, and seven outputs. It has 24,115 trainable parameters.

The dataset source is [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013), version `msambare/fer2013/versions/1`. Check its access terms. The dataset paper is Goodfellow et al., [“Challenges in Representation Learning: A report on three machine learning contests”](https://arxiv.org/abs/1307.0414) (2013), DOI [10.1007/978-3-642-42051-1_16](https://doi.org/10.1007/978-3-642-42051-1_16).

Course sources: CSCI323 Lecture 2, pages 39–59 and 63; Tutorial 1, pages 28–45; Tutorial 2, pages 3–10. The project instruction files set report, experiment, contribution, presentation, and Q&A requirements.

## Local setup

Open this repository in VS Code. Create and activate `.venv`, then install the pinned local packages:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.13 -m venv .venv` and activate `.venv\Scripts\Activate.ps1`. Local requirements pin PyTorch and torchvision without a CPU-only package source.

## Data

KaggleHub saves the versioned image folders to the configured default path. The download stays under `data/fer2013` and is reused on later runs.

```bash
python -c 'import kagglehub; print(kagglehub.dataset_download("msambare/fer2013/versions/1", output_dir="data/fer2013"))'
python -m src.fer.data
```

The shared manifest records source files, class counts, preprocessing, split settings, exact-image hashes, duplicate groups, and exclusions. It converts images to grayscale, resizes them to 48×48 with bilinear interpolation, and scales pixels by 255. Same-label training duplicates stay together. Training copies of images in source test are excluded. Conflicting training labels are listed and excluded. Source-test rows remain in test. No source image or label is changed.

Raw images and manifests are local. The manifest and run outputs are under ignored `outputs/`. The saved split fingerprint for the current version is `76f42aec6b90ee1b1656d521a4a13656b32c69147631625b683ff0860aa9efc7`.

## Train and compare

Run a bounded pipeline check:

```bash
python -m src.fer.train --model all --epochs 1 --max-train-records 512 --max-validation-records 512
```

Run the configured budget after the team agrees on the settings:

```bash
python -m src.fer.train --model all
```

The current settings use Adam, learning rate 0.001, batch size 64, up to 10 epochs, and early stopping after three epochs without validation macro-F1 improvement. A tie uses lower validation loss. Training selects checkpoints with validation data. It never scores source test.

Each run saves its settings, split and subset fingerprints, package versions, model parameter counts, training time, validation metrics, class metrics, confusion matrices, training history, curves, and checkpoints under `outputs/runs/<run-id>/`. Selected small evidence from the existing pilot and full-budget run is in [results/](results/README.md). Both folders label their metrics as validation results.

Export a saved run to one ZIP with its checkpoints, configuration, metrics, training history, and split manifest. The bundle does not include raw images or `.venv`:

```bash
python scripts/export_run_bundle.py --run-dir "outputs/runs/<run-id>"
```

The Colab notebook runs the same export helper at the end and downloads the ZIP before the runtime stops.

## Frozen evaluation

Final test scoring is separate from training. First review validation results and freeze the model and settings. Then choose an existing run and score its saved checkpoints:

```bash
python -m src.fer.evaluate --checkpoint-dir outputs/runs/<run-id> --split test --confirm-final-test
```

This command loads the existing checkpoints. It does not create an optimizer, update weights, select a checkpoint, or train a model. It writes to `outputs/evaluations/`. The source test set has not been evaluated in the saved runs. Do not tune after viewing final test results.

## Colab notebook

Open [notebooks/FT11_Option1_Baselines.ipynb](notebooks/FT11_Option1_Baselines.ipynb) in VS Code or Colab. Colab clones the repository URL configured in the notebook and installs dependencies without replacing Colab's PyTorch or torchvision packages. Push the reviewed code before starting a new Colab runtime. The notebook defaults to a bounded pilot; change `RUN_MODE` to `full` to use the configured budget. Notebook Run All does not score source test.

The actual GitHub remote currently configured for this repository is `https://github.com/AlexBearBear0319/ft11-facial-expression-recognition.git`. If the team changes the remote, update `REPO_URL` in the notebook to the real repository URL. Colab execution has not been verified in an authenticated runtime.

## Historical work and course deliverables

The original v1 notebook and its artifacts are preserved locally under `local_archive/notebooks/results/v1/`; their checksums are in `local_archive/archive_checksums.txt`. The archive is excluded from Git. The pilot and full-budget validation evidence are separate under `results/`.

## Course deliverables

The project requires a report, Python code with a GitHub link, PDF slides, a presentation video with all six members, Q&A, and report and presentation contribution records. Cite the dataset and course sources. Record actual GenAI support in the report. Confirm deadlines, presentation duration, and the six-member contribution format with current Moodle or the tutor. Detailed FT11 planning and task records are kept in the parent `Group Assignment` folder; they are not part of this repository.

OpenAI Codex assisted with this experiment code and its verification. The team must review the code and record this support in its report under current course guidance.

## Synthetic smoke check

```bash
python scripts/smoke_check.py
```

It creates temporary synthetic images, checks split repeatability and duplicate handling, trains each model for one epoch, reloads the saved checkpoints, checks frozen validation predictions, and runs final evaluation on synthetic test images only.
