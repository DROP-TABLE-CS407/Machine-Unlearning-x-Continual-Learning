# Machine-Unlearning-x-Continual-Learning
Master's Thesis for DROP TABLE

# cifar.py

Utilised to split the cifar-10 dataset into something akin to cifar-5.
The general idea we want to implement here is to split the images into different partitions with splits of 2 general classes to see how unlearning algorithms perform depending
on how intertwined the datasets are.

For example possible splitting(s) of cifar-5 could be randomised sets of:

5 birds : 0 animals
4 birds : 1 animal
3 birds : 2 animals
2 birds : 3 animals
....

# Github + VSCode quick set up guide

(if you are sshing through DCS)

## 1.) Install the Github VSCode extension 

## 2.) Ensure you are signed into your Github account with access to this repo

## 3.) run "git clone https://github.com/DROP-TABLE-CS407/Machine-Unlearning-x-Continual-Learning.git" in the directory of your choice

## 4.) run "python3.9 cifar.py"

(if this doesn't work or you are doing this in your own venv make sure you have the 'numpy' dependency installed 'pip3.x install numpy')