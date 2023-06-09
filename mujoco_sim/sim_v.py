import mujoco_py
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from controller import PIDController
from circular_trajectory import *
from reward import calculateReward
# Load the XML model file
model_path = os.path.join('../mujoco_model', 'model.xml')
model = mujoco_py.load_model_from_path(model_path)
# Create a simulation instance
sim = mujoco_py.MjSim(model)

# Print the names of all the joints
# print("Joint names:")
joint_names = []
for i in range(model.njnt):
    joint_names.append(model.joint_id2name(i))
    # print(model.joint_id2name(i))

# Print the names of all the bodies
# print("Body names:")
body_names = []
for i in range(model.nbody):
    body_names.append(model.body_id2name(i))
    # print(model.body_id2name(i))

# Get the IDs of the joint and end effector bodies
joint_id = sim.model.joint_name2id(joint_names[-1])
ee_id = sim.model.body_name2id(body_names[-1])

# Get the IDs of the joint bodies
joint_ids = [sim.model.joint_name2id(name) for name in joint_names]

# print(f'joint ids:{joint_ids}')
# initial_joint_angles, traj = getTrajAndInitJointAngles()


def simOnce(render=False,plot=False,pid_controllers=None):

    # Get initial trajectory and joint angles
    initial_joint_angles, traj = getTrajAndInitJointAngles()

    # Set initial joint angles
    sim.data.qpos[joint_ids] = initial_joint_angles

    # Added extra points for looking at settling error
    for i in range(200):
        traj = np.vstack((traj,traj[-1]))

    rewards = []
    if render:
        # Create a viewer instance
        viewer = mujoco_py.MjViewer(sim)
        viewer.render()

    current_pos = []
    # Simulation loop
    for target_angles in traj:
        # Render the current frame
        if render:
            viewer.render()
        # Step the simulation forward by one time step
        joint_positions = sim.data.qpos[joint_ids]
        current_pos.append(joint_positions)
        joint_velocities = sim.data.qvel[joint_ids]
        end_effector_pos = sim.data.body_xpos[ee_id]
        end_effector_orient = sim.data.body_xquat[ee_id]


        control_effort = [pid.compute(curr,target,sim.model.opt.timestep) for curr,target,pid in zip(joint_positions,target_angles,pid_controllers)]
        # print("Control effort:", control_effort)
        # print('timestep:',sim.model.opt.timestep)
        sim.data.ctrl[joint_ids] = control_effort
        # rewards.append(calculateReward(target_angles,joint_positions))
        sim.step()

    if plot:
        fig,ax = plt.subplots(2,3)
        current_pos = np.array(current_pos)
        for i in range(2):
            for j in range(3):
                ax[i][j].plot(current_pos[:,i*3+j])
                ax[i][j].plot(traj[:,i*3+j])
                ax[i][j].plot(pid_controllers[i*3+j].tracking_errors)
                ax[i][j].set_title(f'joint {i*3+j +1}')
                ax[i][j].legend(['current_pos','target','error'])
                ax[i][j].grid()
        plt.show()

    return current_pos

# simOnce()


# np.save('./benchmarks/benchmark_reward.npy',np.array(rewards))
# for idx,ctrl in enumerate(pid_controllers):
#     np.save('./benchmarks/benchmark_rr_ctrl'+str(idx)+'.npy',np.array(ctrl.tracking_errors))