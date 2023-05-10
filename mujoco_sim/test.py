
from sim_failsafe import *
from A2C_torch import *
from tqdm import tqdm
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

## J1 = kp:397.5799999998996, ki:197.48000000002722, kd:207.32000000003237

INIT_GAINS = [[872.83, 0.57, 98.54], [979.37, 0.57, 27.95], [935.89, 0.15, 7.91], [983.7, 0.0, 86.24], [812.12, 0.29, 61.15], [284.39, 0.36, 6.98]]
pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]


simOnce(render=True,plot=True,pid_controllers=pid_controllers)