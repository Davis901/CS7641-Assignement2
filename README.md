# CS7641-Assignement2
Github:[ https://github.com/Davis901/CS7641-Assignement2/edit/main/](https://github.com/Davis901/CS7641-Assignement2.git)


This project consists of two parts: Randomized Optimizationi and Tuning Neural Networks.
It takes significant amount of time to run due to probelm complexity and loops in some parts.

Below is copy of required library:
import six
import sys
import time
import itertools
import pandas as pd
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive as mlrose
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve

List: Time, pandas, mlrose_hiive, sklearn,preprocessing, sklearn.model, numpy, sklearn.metrics, matplotlib, sklearn.model_selection
Rest of modules are for depedencies.

# Part 1: Randomized Optimization
- randomized_optimization.ipynb
- -> performed optimization with three different problems
- randomized_optimization_hyperparameter_tune.ipynb
- --> Tuning process of hyperparameters for algorithms with loss curve.

# Part 2: Tuning Neural Networks
- neural_networks.ipynb
- -> Perform tuning with optimization algorithm
- nerual_networks_hyperparameter_tune.ipynb
- --> Hyperparameter tuning process 

  
