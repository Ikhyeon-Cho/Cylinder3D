"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: config_tools.py
Date: 2024/11/2 18:50
"""

import os
import random
import torch
import numpy as np


def seed_all(seed):
    '''
    Set seeds for training reproducibility
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)