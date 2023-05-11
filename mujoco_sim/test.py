
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

## J1= 4772.724609375,9.8755606359191e-07,8.397676467895508


INIT_GAINS = [[4859.5810546875,1.00020770332776e-06,8.67470932006836],
              [24239.791015625,9.863824743661098e-07,70.85529327392578],
              [400,0,70],
              [560.6026611328125,10,11.472661018371582],
              [80,10,30],
              [50,0,10]]
pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]


simOnce(render=True,plot=True,pid_controllers=pid_controllers)