# Machine-Unlearning-x-Continual-Learning
Master's Thesis for DROP TABLE

## Abstract
This repository contains the code for the thesis "Machine Unlearning in Continual Learning". The goal of this thesis is to investigate the relationship between machine unlearning and continual learning, and to develop methods that can effectively unlearn data in a continual learning setting. The thesis also explores the implications of machine unlearning for privacy and security in machine learning systems.
The thesis is divided into three main parts:
1. **Introduction**: This section provides an overview of the problem of machine unlearning and its relevance to continual learning. It also discusses the motivation for the research and the main contributions of the thesis.
2. **Related Work**: This section reviews the existing literature on machine unlearning and continual learning, and identifies the gaps in the current research that this thesis aims to address.
3. **Methodology**: This section describes the methods developed in this thesis for machine unlearning in continual learning. It includes a detailed description of the algorithms, experiments, and evaluation metrics used in the research.
4. **Results**: This section presents the results of the experiments conducted in this thesis, and discusses the implications of the findings for machine unlearning and continual learning.
5. **Conclusion**: This section summarizes the main findings of the thesis, and discusses the implications of the research for future work in machine unlearning and continual learning.
6. **References**: This section lists the references cited in the thesis.

## Installation
This repository works by default on DCS systems as long as you have a batch compute access set up.
You will have to change some string directories in the code to make it work on your local machine.

They are in the following file(s):
- `./negGemGradSalun.py`

## Running a test
To run a test, use the provided sbatch scripts. The example jobs provided should work out of the box.
The arguments that can be passed to the python scripts are:
- ` --unlearn_mem_strength ` 0.6 : Strength of the unlearning memory
- ` --unlearn_batch_size ` 10 : Batch size for unlearning
- ` --average_over_n_runs ` 3 : Number of runs to average over
- ` --salun ` 1 : Use salun or not
- ` --salun_strength ` 0.2 : Strength of the salun, we use the top percentage quartile for salun. What this means is that if this 
                             value is set to 0.2 for example, it means the largest absolute magnitude top 20% of the weights are used for salun and the rest are set to 0.
- ` --rum ` 1 : Use rum or not
- ` --rum_split ` 0.1 : Determines how much of the memories are filled with `most` or `least` memorized samples for RUM.
- ` --rum_memorization ` most : Determines if we want to use the most or least memorized samples for RUM. `a` means randomly selected samples.

There are other arguments that can be passed to the scripts in `./negGem/args.py`, but we do not have to change them for now.
Additionally, we can change the unlearning algorithm used, in line `274` of `negGemGradSalun.py` we can change the unlearning algorithm used. The default is `neggem`, but we also have access to `neggrad` where we can specify an alpha value.
The default is `0.9` but we can change this to be whatever we want.

## Continual Learning

### GEM

The GEM algorithm is a continual learning algorithm that uses a memory of past tasks to prevent catastrophic forgetting. The algorithm works by storing a small subset of the training data from each task in memory, and using this memory to compute a gradient that is orthogonal to the gradients of the current task. This prevents the model from forgetting the previous tasks while learning the current task.

We apply the following constraint to the projection of the gradient to reduce the risk of catastrophic forgetting as per the original GEM paper:
$$
\begin{equation}
\begin{aligned}
\min_{\theta} \mathcal{L}_{t}(\theta) + \lambda \sum_{i=1}^{k} \max(0, \langle g_{t}, g_{i} \rangle - \epsilon)
\end{aligned}
\end{equation}
$$
where $\mathcal{L}_{t}(\theta)$ is the loss of the current task, $g_{t}$ is the gradient of the current task, $g_{i}$ is the gradient of the previous tasks, $\lambda$ is a hyperparameter that controls the strength of the constraint, and $\epsilon$ is a small positive constant that prevents the constraint from being too strict.
