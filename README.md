# COD3S
This repository houses the cod3 for 
 - Nathaniel Weir, Joao Sedoc, and Benjamin Van Durme (2020): 
   COD3S: Diverse Generation with Discrete Semantic Signatures. In _Proceedings of EMNLP_.
## Installation
1. This repository uses [conda](https://docs.conda.io/en/latest/miniconda.html) to manage packages and [Ducttape](https://github.com/jhclark/ducttape) to manage intermediate results 
of the experiment pipeline. Follow the latter's [quickstart guide](https://github.com/jhclark/ducttape#quick-start) to add ducttape to your path. 

2. `git clone git@github.com:nweir127/COD3S.git`
3. `cd COD3S && conda env create -f environment.yml && conda activate cod3s`
4. edit `tapes/main.tape` to point the `COD3S` package to your local repo
5. edit `tapes/submitters.tape` to point to your local `conda.sh` init script.


## Running Experiments
To run a task, run `ducttape tapes/main.tape -p <TASK>`, where task is one of the following:

| Task           | Description                                                      |
|----------------|------------------------------------------------------------------|
| `download_data`| downloads the training and evaluation data              |
| `compute_cod3s`  | Computes LSH signatures of all training and evaluation data      |
| `train`   |              |

Note that tasks will automatically run the other tasks required by their dependencies (`train` will run `download_data` etc).
