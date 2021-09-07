# Introduction

Training and inferencing code for MICCAI2021 competition: GAMMA, subtask 1 (image classification)

# Environment

```
python==3.8.8
torch==1.7.0
timm==0.4.12
pytorch_lightning==1.4.2
cuda==10.2
```

# Data directory structure

- data

    - train

        - images (renamed)

        - train.csv (renamed)

    - test

        - images

# How to run

## NOTE: 

**These are not the full codes for competition, but are some code templates for hyper-parameter tuning.**

- `python Solver_IMG.py` for training with fundus images

- `python Solver_OCT.py` for training with OCT images

- training configs were set in `args`

- `Inferencer_*.py` for inferencing
