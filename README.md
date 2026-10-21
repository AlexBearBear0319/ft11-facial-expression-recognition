# FT11 Facial Expression Recognition

FT11's CSCI323 group project. The current plan compares a majority-class baseline, an MLP, a small CNN, and one controlled CNN augmentation test. The team must agree on the dataset and settings before full training.

## Course and assignment sources

- Course concepts: Lecture 2 and Tutorials 1-2. The project applies feedforward networks, CNNs, convolution, shared filters, pooling, classification loss, backpropagation, and gradient-based training.
- Assignment requirements: use the project brief, report and presentation guides, current Moodle instructions, and the FT11 Notion plan. These set the deliverables and deadlines.
- Project choices: FER2013, package versions, exact network sizes, image transforms, evaluation metrics, and run limits are proposals. The course material does not prescribe them. Record team decisions and evidence in Notion.

## Team links

- [FT11 Project Plan](https://app.notion.com/p/3fe192ff496081f9a995d9ebdc306a02)
- [FT11 Task Tracker](https://app.notion.com/p/3dcc8a3bb49b44689976edf497e98130)
- [FT11 Progress and Decisions](https://app.notion.com/p/3fe192ff4960819ab5e5f6b2f9e80cc6)

## Setup

Use the Python version and package versions that FT11 confirms after the shared setup check. The dependency list is provisional. Record the tested versions before full training.

```bash
python -m venv .venv
```

Activate the environment, then install the listed packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation can differ by operating system and hardware. Use the official PyTorch install selector if the default install does not match the team environment.

## Project layout

```text
configs/    agreed model, split, and run settings
data/       local datasets; never commit raw data
notebooks/  short exploration and demonstration notebooks
results/    selected small tables and figures for the report
src/        reusable data, model, and evaluation code
```

Do not commit passwords, access tokens, raw datasets, virtual environments, caches, or model checkpoints. Store large files outside GitHub and document how the team can obtain them. Check dataset access terms before use.

## Team workflow

1. Check the FT11 Notion tracker before starting work. Agree on a real owner with the team.
2. Create a branch for one task, for example `t05-data-audit`.
3. Make a small change and include the task ID in the commit message, for example `T05: add data checks`.
4. Open a pull request into `main`. Ask one teammate to review it.
5. Merge after the reviewer confirms that the change works and the evidence is saved.
6. Update the Notion task with its status, evidence, blocker, and next action.

Keep `main` as the shared working version. Do not add results or contribution claims until the team has checked them.

## Current state

This is a starter repository. It has no trained model, dataset, result, or agreed team owner yet. The first shared setup task is to confirm that another member can install the dependencies and run a small example.
