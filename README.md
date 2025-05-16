# Machine-Unlearning-x-Continual-Learning
Master's Thesis for DROP TABLE

## Abstract
This repository contains the code for the thesis "Machine Unlearning in Continual Learning". The goal of this thesis is to investigate the relationship between machine unlearning and continual learning, and to develop methods that can effectively unlearn data in a continual learning setting. The thesis also explores the implications of machine unlearning for privacy and security in machine learning systems.
The thesis is divided into three main parts:
1. **Introduction**: This section provides a brief overview on how to run this experiment along with some installation instructions.
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

# 1.) Introduction

## Running a Test

To run an experiment, use the provided `sbatch` scripts or execute a command directly via terminal. The repository includes sample job scripts that should work out of the box once dependencies are installed and the dataset is downloaded.

### Example Command

```bash
time python3.12 negGemGradSalun.py \
  --algorithm neggem \
  --alpha 0.9 \
  --number_of_gpus 3 \
  --learn_mem_strength 0.5 \
  --learn_batch_size 10 \
  --unlearn_mem_strength 0.6 \
  --unlearn_batch_size 10 \
  --average_over_n_runs 3 \
  --salun 1 \
  --salun_strength 0.2 \
  --mem_learning_buffer 1 \
  --learning_buffer_split 0.2 \
  --learning_buffer_type least \
  --mem_unlearning_buffer 1 \
  --unlearning_buffer_split 0.2 \
  --unlearning_buffer_type most
```

### Argument Descriptions

| Argument | Description |
|----------|-------------|
| `--algorithm` | Defines which unlearning algorithm to use. Options: `neggem`, `negagem`, `RL-GEM`, `RL-AGEM`, `ALT-NEGGEM`, `neggrad`. |
| `--alpha` | Projection parameter for `neggrad` and related algorithms. |
| `--number_of_gpus` | Number of GPUs to use in parallel (e.g., for multi-run averaging). |
| `--learn_mem_strength` | GEM margin used during continual learning. |
| `--learn_batch_size` | Mini-batch size used during continual learning. |
| `--unlearn_mem_strength` | Margin for projection in unlearning algorithms. |
| `--unlearn_batch_size` | Batch size for unlearning updates. |
| `--average_over_n_runs` | Number of runs to average results over. |
| `--salun` | Toggle SaLUn on/off (`1` for enabled, `0` for disabled). |
| `--salun_strength` | Fraction of gradient elements retained by magnitude (e.g., `0.2` keeps top 20%). |
| `--mem_learning_buffer` | Enables memory buffer based on memorisation for learning. |
| `--learning_buffer_split` | Portion of learn buffer filled using memorisation scores; rest is random. |
| `--learning_buffer_type` | Score type used for selection: `least`, `most`, or `random`. |
| `--mem_unlearning_buffer` | Enables memory buffer based on memorisation for unlearning. |
| `--unlearning_buffer_split` | Portion of unlearn buffer filled using memorisation scores; rest is random. |
| `--unlearning_buffer_type` | Score type used for selection: `most`, `least`, or `random`. |

### Additional Notes

- All hyperparameters are defined in `negGem/args.py`. Only the arguments listed above are required for standard experiments.
- Class/task order is shuffled internally per run to simulate continual learning in realistic task sequences.
- Results are saved under `Results_*/`, including plots and CSV logs for metrics.

# 2.) Related Work

## Continual Learning

### GEM
The GEM algorithm is a continual learning algorithm that uses a memory of past tasks to prevent catastrophic forgetting. The algorithm works by storing a small subset of the training data from each task in memory, and using this memory to compute a gradient that is orthogonal to the gradients of the current task. This prevents the model from forgetting the previous tasks while learning the current task.

The original paper for GEM is "Gradient Episodic Memory for Continual Learning" by Lopez-Paz and Ranzato (2017). The paper can be found [here](https://arxiv.org/abs/1706.08840).

### AGEM
The Averaged Gradient Episodic Memory (AGEM) algorithm is a variant of the GEM algorithm that uses an average of the gradients from the memory to compute the orthogonal gradient. This allows the model to learn from the memory more effectively, and prevents catastrophic forgetting. This was found to be significantly faster and more efficient than the original GEM algorithm but not as performant is certain specific scenarios.

The original paper for AGEM is "Averaged Gradient Episodic Memory for Continual Learning" by Chaudhry et al. (2018). The paper can be found [here](https://arxiv.org/abs/1812.00420).

## Unlearning

### Negative Gradient Unlearning
Otherwise known as neggrad, this is a method for unlearning data from a model by computing the negative gradient of the loss function with respect to the model parameters. This allows the model to "unlearn" the data by moving the parameters in the opposite direction of the gradient, effectively removing the influence of the data from the model.

The paper that discusses this method was found in quite a few of the paper references here, but 
from the SCRUB paper (we don't actually use SCRUB here), we can see that this is a method that is used to unlearn data from a model. The paper can be found [here](https://arxiv.org/abs/2302.09880). 

### Refined Unlearning Meta Algorithm (RUM)
This is effectively where we target the most or least memorized samples in the memory and unlearn them. This is done by using
precomputed memorization scores to determine which samples are the most and least memorized.

The source for memorization scores for the CIFAR-100 can be found from the paper "What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation" by Vitaly Feldman and Chiyuan Zhang the download for the memorization scores 
is [here](https://pluskid.github.io/influence-memorization/#cifar100-dl)

The paper that discusses this method is "What makes unlearning hard and what to do about it" by Kairan Zhao, Meghdad Kurmanji, George-Octavian Bărbulescu, Eleni Triantafillou, Peter Triantafillou (2023). The paper can be found [here](https://arxiv.org/abs/2406.01257).

### SalUn
This is a method for unlearning data from a model by using the saliency of the data to determine which samples to unlearn. This allows the model to "unlearn" the data by removing the influence of the most salient samples from the model.
The paper that discusses this method is "SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation" by Chongyu Fan et al. (2024). The paper can be found [here](https://arxiv.org/abs/2310.12508).

# 3.) Methodology

It has been proven that we are able to continually learn tasks without the risk of catastrophic forgetting.
From papers such as "Gradient Episodic Memory for Continual Learning" by Lopez-Paz and Ranzato (2017) and "Averaged Gradient Episodic Memory for Continual Learning" by Chaudhry et al. (2018), we can see that we are able to learn many tasks by using a small
memory buffer alongside constraints on the gradient of the model to retain accuracy on the previous tasks.

Our question is: Can we apply these same constraints in the opposite direction to continually unlearn tasks using
a subset of the memory buffer?

First of all, let us define a task: A task in the case of CIFAR100 for our experiment is being able to classify between 5 of the classes out of 100 classes from CIFAR100.
This means that we are able to learn 20 tasks in total, and we can unlearn any of the tasks at any time.

The main idea is to use the memory buffer to compute the gradient of the model with respect to the task that we want to unlearn, and then use this gradient to update the model parameters in the opposite direction. This allows us to "unlearn" the task by removing its influence from the model. However we apply the same constraints as GEM and AGEM to ensure that we do not forget the other tasks while unlearning the task.

The main experiment being ran here is to:

1. Train a model on the CIFAR100 dataset using the GEM or AGEM algorithms continually learning 20 randomized tasks.
2. Unlearn all tasks backwards from 20 to 2 (we retain on 1) using a combination of NegGrad+, NegGem, RUM and SalUn.
3. Evaluate the model on the tasks after unlearning to see if we are able to retain accuracy on the tasks that we have not unlearned.

There are more experiments to be ran with respect to ordering of tasks, but this is the main experiment that we are running here.
We want to ensure that we are able to unlearn the tasks without forgetting the other tasks, and task 1 in this case
is the task that has been retained the longest through 20 rounds of continual learning and 19 rounds of continual unlearning.

# 4.) Results

### Our own continual learning results for the continuum of 20 tasks from CIFAR 100 running on AGEM and GEM variants:

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373020483477438697/cifar100CLcomp.png?ex=6828e478&is=682792f8&hm=80d52ddc1f0a7068c3ef0610620a1197c5b9925c0c6dfc585ba74e4dd9ef8f28&=&format=webp&quality=lossless&width=1247&height=391 "Our own continual learning results for the continuum of 20 tasks from CIFAR 100 running on AGEM and GEM variants")

### MIA results proving that a continual learner can maintain model utility whilst reducing risk of MIA AUC for samples of the the entire train set. Only the buffer samples (~10% of the original train data) become memorised protecting the rest of the train data:

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021328344612934/miaAGEM.png?ex=6828e542&is=682793c2&hm=a8135853fdd334736293cb4ea6cceca31ce5eca338381e1ff54b4c126c0287a4&=&format=webp&quality=lossless&width=1540&height=880 "MIA results proving that a continual learner can maintain model utility whilst reducing risk of MIA AUC for samples of the the entire train set. Only the buffer samples (~10% of the original train data) become memorised protecting the rest of the train data")

### A4 Test Set Task Accuracies:

![Alt text](https://cdn.discordapp.com/attachments/884782265866014760/1373016794071175260/ALL_TASK_ACCURACIES.png?ex=6828e109&is=68278f89&hm=19536202c550eaf56dc708140f548e24fb9b545470491c15027d2680acfdb4e1& "NegGEM on task sequence A4, all test accuracies shown")

### A4 CL-TOW values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373023267379613880/tow_scores_a4.png?ex=6828e710&is=68279590&hm=3c1f165fb85b0b18ee4fd5fb017cac6dde8c911d34d3f428c359d34723e5ba7e&=&format=webp&quality=lossless&width=2254&height=1128 "A4 CL-TOW values across all algorithms")

### A4 $CLU_{mix}$ values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021355313856675/CLUA4.png?ex=6828e548&is=682793c8&hm=255c288d00562d7094ec264c86aae4e073931f6504343b7f883343f337d8aff6&=&format=webp&quality=lossless&width=2254&height=1128 "A4 CLUmix values across all algorithms")

### B4 $CLU_{mix}$ values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021368236511373/CLUB4.png?ex=6828e54b&is=682793cb&hm=26dcc999d126fb59a4dd964b5bbf1135ace6becc734dcd3f44990bc56bc44aaa&=&format=webp&quality=lossless&width=2254&height=1128 "B4 CLUmix values across all algorithms")

### C4 $CLU_{mix}$ values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021404508852395/CLUD4.png?ex=6828e554&is=682793d4&hm=b7c0386d9ffac48461b855d34744393c5b33f9b9c91f9e346dcfc509371613b1&=&format=webp&quality=lossless&width=2254&height=1128 "C4v2 CLUmix values across all algorithms")

### C4v2 $CLU_{mix}$ values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021376449089708/CLUc4v2.png?ex=6828e54d&is=682793cd&hm=088cbaf3934f8606fb71ef8918ab66fea0eee0d232994da20bc323d8f91de408&=&format=webp&quality=lossless&width=2254&height=1128 "C4 CLUmix values across all algorithms")

### D4 $CLU_{mix}$ values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021388452925531/CLUC4.png?ex=6828e550&is=682793d0&hm=49585efd9afbca994a00b2020adcd9ceea1b736329eeb0c9971975fc12c9c93f&=&format=webp&quality=lossless&width=2254&height=1128 "D4 CLUmix values across all algorithms")

### E4 $CLU_{mix}$ values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021415011385475/CLUE4.png?ex=6828e556&is=682793d6&hm=f835dd0661839095017b581e034b99a49671828ac9c1fc930d494832c59a7fa8&=&format=webp&quality=lossless&width=2254&height=1128 "E4 CLUmix values across all algorithms")

### F4 $CLU_{mix}$ values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021428143882311/CLUF4.png?ex=6828e559&is=682793d9&hm=d0b9acb8c0dbcedf42aaf9d488b58c328041a37f679c3a4a7f9a9a60b66eba6a&=&format=webp&quality=lossless&width=2254&height=1128 "F4 CLUmix values across all algorithms")

### G4 $CLU_{mix}$ values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021439036358727/CLUG4.png?ex=6828e55c&is=682793dc&hm=8a8b5615e9fcbf752689ac4c8498e830f7e454e644489a25561d0605d3af933b&=&format=webp&quality=lossless&width=2254&height=1128 "G4 CLUmix values across all algorithms")

### H4 $CLU_{mix}$ values across all algorithms

![Alt text](https://media.discordapp.net/attachments/891721126340853761/1373021451065626674/CLUH4.png?ex=6828e55f&is=682793df&hm=dd39392485697ea162d2c93ca143b3c53147c5439e658362f55756b084e13109&=&format=webp&quality=lossless&width=2254&height=1128 "H4 CLUmix values across all algorithms")


# 5.) Conclusion

Across all experiments, one conclusion becomes unavoidable:
learning and unlearning must be designed together, not separately.

Successful unlearning is not just about cancelling out past gradients or deleting memory buffers—it depends crucially on the structures laid down during initial learning.
Algorithms that enforced structured learning updates (such as GEM-style constraints) created conditions where unlearning could be performed selectively, precisely, and efficiently, preserving the ability to learn new tasks and protecting privacy.

By contrast, methods that treated learning and unlearning as independent phases—first memorising freely, then applying naive erasure techniques—struggled with residual memorisation, poor future learning capacity, and unstable forgetting.

Thus, our results show that continual unlearning systems must adopt a holistic design philosophy: the way a model is trained must explicitly prepare it for the possibility of future unlearning.
Learning and unlearning are two sides of the same coin, and only by crafting them to work synergistically can we achieve models that learn continuously, forget selectively, and do so with efficiency, robustness, and privacy.

# 6.) References
- David Lopez-Paz, Marc'Aurelio Ranzato. Gradient Episodic Memory for Continual Learning. In NeurIPS, 2017. [here](https://arxiv.org/abs/1706.08840)
- Chaudhry, A., Rohrbach, M., Akata, Z., Schmid, C., & Tuytelaars, T. (2018). Averaged Gradient Episodic Memory for Continual Learning. In European Conference on Computer Vision (ECCV) (pp. 633-648). Springer. [here](https://arxiv.org/abs/1812.00420)
- Feldman, V., & Zhang, C. (2021). What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation. In International Conference on Learning Representations (ICLR). [here](https://arxiv.org/abs/2009.07832)
- Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, Eleni Triantafillou (2023). Towards Unbounded Machine Unlearning. In NeurIPS. [here](https://arxiv.org/abs/2302.09880)
- Vitaly Feldman and Chiyuan Zhang (2021). What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation. [here](https://arxiv.org/abs/2008.03703)
- Kairan Zhao, Meghdad Kurmanji, George-Octavian Bărbulescu, Eleni Triantafillou, Peter Triantafillou (2023). What makes unlearning hard and what to do about it. [here](https://arxiv.org/abs/2406.01257)
- Chongyu Fan, Yujun Shen, Zhaoyang Lv, Yujia Zhang, Jiajun Wu, Zexiang Xu (2024). SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation. [here](https://arxiv.org/abs/2310.12508)

# 7.) Extra notes on distributed computing

## Using DCS
The code is set up to work with DCS systems. We specifically target the `gecko` nodes
to distribute multiple model across multiple GPUs. The single GPU `falcon` nodes
take 1hr 5 minutes to train the average of 1 run using batch size 10 which was the recommended batch size from the original GEM paper.
Hence, average of 3 runs would take 3hr 15 minutes to train on a single GPU.
The `gecko` nodes however gives us access to 3 GPUs at the same time. I have found a way to use torch
multiprocessing libraries to train multiple models concurrently. This way the average of 3 runs
on `gecko` takes the same time as the average of 1 run on `gecko`. This means that the average
of 3 runs on `gecko` takes 1hr 25 minutes to train.

You can use the single node version of the code by using the old branch:
`housekeeping_rum_memorization`
But because we need averaged results fast, please use this branch if 3 GPUs are available.
