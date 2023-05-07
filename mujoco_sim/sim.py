import mujoco_py
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from controller import PIDController
from dqn import DQN_Agent
import copy

# Load the XML model file
model_path = os.path.join('../mujoco_model', 'model.xml')
model = mujoco_py.load_model_from_path(model_path)
# Create a simulation instance
sim = mujoco_py.MjSim(model)


# Print the names of all the joints
print("Joint names:")
joint_names = []
for i in range(model.njnt):
    joint_names.append(model.joint_id2name(i))
    print(model.joint_id2name(i))

# Print the names of all the bodies
print("Body names:")
body_names = []
for i in range(model.nbody):
    body_names.append(model.body_id2name(i))
    print(model.body_id2name(i))

# Get the IDs of the joint and end effector bodies
joint_id = sim.model.joint_name2id(joint_names[-1])
ee_id = sim.model.body_name2id(body_names[-1])

# Get the IDs of the joint bodies
joint_ids = [sim.model.joint_name2id(name) for name in joint_names]

print(f'joint ids:{joint_ids}')

# Set the initial joint angles
initial_joint_angles = [1.0] * len(joint_ids)
sim.data.qpos[joint_ids] = initial_joint_angles

# Create a viewer instance
#viewer = mujoco_py.MjViewer(sim)
action_space = [1]*18+[0]*18+[-1]*18

pid_controllers = target_angles = None
def reset_sim():
    global target_angles
    global pid_controllers
    target_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Create a PID controller for each joint
    gains = [[100,0,0],[100,5,0],[100,0,0],[80,0,0],[80,0,0],[0,0,0]]
    pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in gains]
    return np.concatenate((sim.data.qpos[joint_ids], sim.data.qvel[joint_ids]))

reset_sim()

def step(gain):
    # Update gain by incrementing by them by the quantity determined by the previous action (passed as parameter gains)
    print(action_space)
    gainType = gain%3
    value = action_space[gain]
    id = gain%18
    pid_controllers[id].update_gain(gainType,value)
    joint_positions = sim.data.qpos[joint_ids]
    joint_velocities = sim.data.qvel[joint_ids]
    # Determine motor torque based on gains and error between target and current position
    control_effort = [pid.compute(curr,target,sim.model.opt.timestep) for curr,target,pid in zip(joint_positions,target_angles,pid_controllers)]
    sim.data.ctrl[joint_ids] = control_effort
    reward = sum([math.abs(controller.get_prev_error()) for controller in pid_controllers])*-1
    done = reward > -0.05
    sim.step()
    return np.concatenate((sim.data.qpos[joint_ids], sim.data.qvel[joint_ids])), reward, done


###DQN TRAINING###


input_dim = 12
output_dim = action_space_len = 18
exp_replay_size = 256
agent = DQN_Agent(seed = 1423, layer_sizes = [input_dim, 64,64, output_dim], lr = 1e-3, sync_freq = 5, exp_replay_size = exp_replay_size)

# initiliaze experience replay      
index = 0
for i in range(exp_replay_size):
    obs = reset_sim()
    done = False
    while(done != True):
        A = agent.get_action(obs, action_space_len, epsilon=1)
        obs_next, reward, done = step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])
        obs = obs_next
        index += 1
        if( index > exp_replay_size ):
            break
            
# Main training loop
losses_list, reward_list, episode_len_list, epsilon_list  = [], [], [], []
index = 128
episodes = 10000
epsilon = 1
maxr = -10e9
best = None
reset_sim()
for i in range(episodes):
    obs, done, losses, ep_len, rew = step(), False, 0, 0, 0
    while(done != True and ep_len < 200):
      ep_len += 1 
      A = agent.get_action(obs, action_space_len, epsilon)
      obs_next, reward, done = step(A.item())
      agent.collect_experience([obs, A.item(), reward, obs_next])
      
      obs = obs_next
      rew  += reward
      index += 1
      
      if(index > 128):
          index = 0
          for j in range(4):
              loss = agent.train(batch_size=16)
              losses += loss  
    if rew > maxr:
      maxr = rew
      best = copy.deepcopy(agent)
      print('new max reward of ',maxr)

    if epsilon > 0.05 :
        epsilon -= (1 / 5000)
    if i%5 == 0:
      print('episode ',i, 'reward',rew, 'length', ep_len)
    
    losses_list.append(losses/ep_len), reward_list.append(rew), episode_len_list.append(ep_len), epsilon_list.append(epsilon)



# # Simulation loop
# for i in range(4000):
#     # Render the current frame
#     viewer.render()
#     #target_angles = [np.sin(i/200)]*6

#     # Step the simulation forward by one time step
#     joint_positions = sim.data.qpos[joint_ids]
#     joint_velocities = sim.data.qvel[joint_ids]
#     end_effector_pos = sim.data.body_xpos[ee_id]
#     end_effector_orient = sim.data.body_xquat[ee_id]


#     control_effort = [pid.compute(curr,target,sim.model.opt.timestep) for curr,target,pid in zip(joint_positions,target_angles,pid_controllers)]
#     print("Control effort:", control_effort)
#     print('timestep:',sim.model.opt.timestep)
#     sim.data.ctrl[joint_ids] = control_effort
    
#     sim.step()

# plt.figure()
# for i in range(6):
#     plt.plot(pid_controllers[i].tracking_errors)
# plt.show()