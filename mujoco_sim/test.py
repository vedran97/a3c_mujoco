
from sim_v import *
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

INIT_GAINS = [[397.579999,197.4800000,207.320000],[800,0,100],[400,0,70],[200,0,30],[80,10,30],[50,0,10]]
pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]


simOnce(render=True,plot=True,pid_controllers=pid_controllers)