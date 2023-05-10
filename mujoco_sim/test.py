
from sim_failsafe import *

## J1 = kp:397.5799999998996, ki:197.48000000002722, kd:207.32000000003237
#good gains? [[1914.59, 0.16, 16.65], [1920.58, 8.12, 119.02], [1739.63, 0.16, 149.67], [1170.31, 7.92, 8.5], [1001.75, 9.52, 15.4], [1085.76, 3.16, 6.81]]
INIT_GAINS = [[2933.78, 4.57, 298.07], [2974.91, 0.11, 19.78], [2985.38, 17.43, 3.08], [2985.18, 18.34, 1.65], [2761.19, 2.42, 23.51], [2847.56, 19.91, 0.03]]
pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]


simOnce(render=True,plot=True,pid_controllers=pid_controllers)