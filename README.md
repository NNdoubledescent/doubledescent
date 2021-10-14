# Multi-scale Feature Learning Dynamics

This repository contains the official implementation of:
> Multi-scale Feature Learning Dynamics: Insights for Double Descent

## Reproducibility
To ensure reproducibility, we publish the code, saved logs, and expected results of every experiment.
We claim that all figures presented in the manuscript can be reproduced using the following requirements:
```
Python 3.7.10
PyTorch 1.4.0
torchvision 0.5.0
tqdm
matplotlib 3.4.3
```

### ResNet experiments on CIFAR-10
ResNet experiments on CIFAR-10 took 12000 GPU hours on Nvidia V100. The code to manage experiments using the  ```slurm``` resource management tool is provided in the README available in the ```ResNet_experiments``` folder.

### To reproduce each figure of the manuscript
```python fig1.py```:
 The generalization error as the training time proceeds. (top): The case where only the fast-learning feature or slow-learning feature are trained. (bottom): The case where both features are trained with \kappa=100.
![fig](/expected_results/fig1.png)

```python fig2_ab.py```:
Heat-map of empirical generalization error (0-1 classification error) for the ResNet-18 trained on CIFAR-10 with $15 % label noise. The X-axis denotes the regularization strength, and Y-axis represents the training time.
![fig](/expected_results/fig2_ab.png)

```python fig2_cd.py```:
The same plot with the analytical results of the teacher-student. We observe a qualitative comparison between the ResNet-18 results and our analytical results.
![fig](/expected_results/fig2_cd.png)


```python fig3.py```:
Left: Phase diagram of the generalization error as a function of R(t) and Q(t). The trajectories describe the evolution of R(t) and Q(t) as training proceeds. Each trajectory corresponds to a different  $\kappa$, the condition number of the modulation matrix where it describes the ratio of the rates at which two sets of features are learned.
Right: The corresponding generalization curves for different plotted over the training time axis.
![fig](/expected_results/fig3.png)


## Extra experiments

### Match between theory and experiments
Here, we validate our analytical results by comparing the following three methods:
```
1. Emperical gradient descent
2. Analytical results - the exact general case (Eq. 9 substituted into Eq. 6):
3. Analytical results - the approximate fast-slow case (Eqs. 12, 14 substituted into Eq. 6):
```

``` python extra_experiments/emp_vs_analytic.py```
![fig](/extra_experiments/emp_vs_analytic.png)

### Previous experiments with different setups

We also provide further experiments where we vary the following variables:

```
n: number of training examples
d: number of total dimensions
p: number of fast learning dimensions
```

```Four variants of fig1```
![fig](/extra_experiments/variants_of_fig1.png)


```Four variants of fig3```
![fig](/extra_experiments/variants_of_fig3.png)


## Interactive notebook
Finally, to try different setups, please check out the following anonymous colab notebook: [Link](https://colab.research.google.com/drive/10UHRBnIa2V8uwBWXd5W_-ZhKKSh2OPy7?usp=sharing)
