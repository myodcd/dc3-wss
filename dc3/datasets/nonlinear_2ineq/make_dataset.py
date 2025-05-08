import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import Problem_Non_Linear_2Ineq

os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float32)

num_var = 2

np.random.seed(17)

num_examples = [200, 500, 1000, 10000, 30000]

for examples in num_examples:

    x = np.random.uniform(0, 4, (examples, num_var))

    problem = Problem_Non_Linear_2Ineq(x)

    
    if os.name == 'nt':  # Windows    
        file_path = ".\\random_nonlinear_2ineq_dataset_ex{}".format(examples)
    else:  # Linux/Unix/MacOS
        file_path = "./random_nonlinear_2ineq_dataset_ex{}".format(examples)

    with open(file_path, 'wb') as f:
        pickle.dump(problem, f)