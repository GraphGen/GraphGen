# GraphGen: A Scalable Approach to Domain-agnostic Labeled Graph Generation

This repository is the official PyTorch implementation of GraphGen, a generative graph model using auto-regressive model.

Most of the code has been adapted from [GraphRNN](https://github.com/snap-stanford/GraphRNN)

## Installation

We recommend [anaconda](https://www.anaconda.com/distribution/) distribution for Python and other packages. The code has been tested over [PyTorch 1.2.0](https://pytorch.org/) version with Python 3.7.0.

Pytorch and pip installation in conda. Change cuda version as per your GPU hardware support.

```bash
conda install pip pytorch=1.2.0 torchvision cudatoolkit=10.1 -c pytorch
```

Then install the other dependencies.

```bash
pip install -r requirements.txt
```

[Boost](https://www.boost.org/) and [OpenMP](https://www.openmp.org/) are required for compling C++ binaries. Run `build.sh` script in the project's root directory.

```bash
./build.sh
```

## Test run

```bash
python3 main.py
```

## Code description

1. `main.py` is the main script file, and specific arguments are set in `args.py`.
2. `train.py` includes training iterations framework and calls `model.py`, `data.py` and generative algorithm specific training files.
3. `preprocess.py` and `util.py` contain preprocessing and utility functions.
4. `process_dataset.py` reads graphs from various formats.

GraphGen:

- `dfs_code.cpp` calculates the minimum DFS code required by GraphGen. It is adapted from [kaviniitm](https://github.com/kaviniitm/DFSCode). `dfs_wrapper.py` is a python wrapper for the cpp file.
- `model.py` and `data.py` contain the model and DataLoader class respectively.
- `train_dfscode_rnn.py` contains the core loss evaluation and generation algorithm for GraphGen

For baseline models:

- We extend DeepGMG model for labeled graphs based on the [DGL (Deep Graph Library)](https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgmg). DeepGMG specific files are contained in `dgmg` folder
- We extended DeepGMG model for labeled graphs based upon [GraphRNN](https://github.com/snap-stanford/GraphRNN). GraphRNN specfic code is contained in `data.py`, `model.py` and `train_dfscode_rnn.py`

Parameter setting:

- All the input arguments and hyper parameters setting are included in `args.py`.
- **Set `args.note` to specify which generative algorithm (GraphGen, GraphRNN or DeepGMG) to run.**
- For example, `args.device` controls which device (GPU) is used to train the model, and `args.graph_type` specifies which dataset is used to train the generative model.
- See the documentation in `args.py` for more detailed descriptions of all fields.

## Outputs

There are several different types of outputs, each saved into a different directory under a path prefix. The path prefix is set at `args.dir_input`. Suppose that this field is set to `./`:

- `./tensorboard` contains tensorboard event objects which can be used to view training and validation graphs in real time.
- `./model_save` stores the model checkpoints
- `./temp` stores all the temporary files generated during training and evaluation.

## Evaluation

- The evaluation is done in `evaluate.py`, where user can choose which model to evaluate. Change the `ArgsEvaluate` class fields accordingly.
- We use [GraphRNN](https://github.com/snap-stanford/GraphRNN) implementation for structural metrics.
- [NSPDK](https://dtai.cs.kuleuven.be/software/nspdk) is evaluated using [EDeN](https://github.com/fabriziocosta/EDeN) python package.
- `isomorph.cpp` and `unique.cpp` contain C++ function call to boost subgraph isomorphism algorithm to evaluate novelty and uniqueness.

To evaluate, run

```bash
python3 evaluate.py
```
