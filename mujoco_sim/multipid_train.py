from sim_failsafe import *
from tqdm import tqdm
import torch
import numpy as np
from actor_critic_continuous import Agent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

state_size = 2700*6
action_size = 18
scale = [1000, 1, 100]
alphas = [0.01,0.01,0.01,0.01,0.01,0.01]
best_errors = [1e1000,1e1000,1e1000,1e1000,1e1000,1e1000]
controller_range = 6
episodes = 100000
settling = [0,0,0,0,0,0]

INIT_GAINS = [[200,0,10],[800,0,100],[400,0,70],[200,0,30],[80,10,30],[50,0,10]]
initial_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]
pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]


class PIDENV:
    def __init__(self,controllers):
        self.controllers = controllers
        self.done = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = np.array([0]*18)
    def step(self,action,epoch):
        for i, controller in enumerate(self.controllers):
            self.controllers[i].reset_controller()
            self.controllers[i].set_gains(1000*action[3*i], action[3*i+1], 100*action[3*i+2])
            #self.controllers[i].update_gains(delta_kp,delta_ki,delta_kd)
            pid_controllers[i] = self.controllers[i]
        current_pos, fail = simOnce(render=False,plot=False,pid_controllers=pid_controllers)
        sums = []
        for i in range(len(self.controllers)):
            sums.append(np.sum(np.abs(np.array(self.controllers[i].tracking_errors, dtype=np.float32))))
        error_sum = sum(sums)
        reward = -error_sum
        if fail:
            reward -=1000
        elif error_sum < 0.005 * (state_size -1):
            reward = 10
            self.done = True
        print('\nCrashed:',fail, 'Reward',reward)
        next_state = (np.array(current_pos)).flatten()
        if epoch%20 == 0 :
            print()
            for i, controller in enumerate(self.controllers):
                print("error sum:{} , Joint:{}".format(round(sums[i],3),i+1))
        g = []
        for i, controller in enumerate(self.controllers):
            g.append([round(self.controllers[i].Kp,2), round(self.controllers[i].Ki,2), round(self.controllers[i].Kd,2)])
        print(f'Gains: {g}')
        return next_state, reward, self.done
    def reset(self):
        return np.array(simOnce(render=False,plot=False,pid_controllers=initial_controllers)[0]).flatten()

agent = Agent(alpha=0.00005, beta=0.0001, input_dims=state_size, gamma=0.0,
                  layer1_size=256, layer2_size=256, n_actions=2*action_size, n_outputs=action_size)

env = PIDENV(pid_controllers)
num_episodes = 500
score_history = []
observation = env.reset()
score = 0

for i in range(num_episodes):
    #env = wrappers.Monitor(env, "tmp/mountaincar-continuous-trained-1",
    #
    #                       video_callable=lambda episode_id: True, force=True)
    
    done = False
    
    action = np.array(agent.choose_action(observation))
    observation_, reward, done = env.step(action, i)
    agent.learn(observation, reward, observation_, done)
    observation = observation_
    score_history.append(reward)
    print('episode: ', i,'score: %.2f' % score)


plt.plot(plt.plot(score_history))
plt.show()
    