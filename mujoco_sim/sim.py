import mujoco_py
import os
import matplotlib.pyplot as plt
import numpy as np

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
initial_joint_angles = [0.0] * len(joint_ids)
sim.data.qpos[joint_ids] = initial_joint_angles

# Create a viewer instance
viewer = mujoco_py.MjViewer(sim)

## 
target_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.tracking_errors = []
        
    def compute(self, current_value,target, dt):
        error = target - current_value
        self.tracking_errors.append(error)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        control_effort = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return control_effort

# Create a PID controller for each joint
gains = [[100,0,0],[100,5,0],[100,0,0],[80,0,0],[80,0,0],[0,0,0]]
pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in gains]
# Simulation loop
for i in range(10000):
    # Render the current frame
    viewer.render()

    # Step the simulation forward by one time step
    

    joint_positions = sim.data.qpos[joint_ids]
    print("Joint positions:", joint_positions)

    joint_velocities = sim.data.qvel[joint_ids]
    print("Joint velocities:", joint_velocities)

    end_effector_pos = sim.data.body_xpos[ee_id]
    end_effector_orient = sim.data.body_xquat[ee_id]
    print("End effector position:", end_effector_pos)
    print("End effector orientation:", end_effector_orient)


    control_effort = [pid.compute(curr,target,sim.model.opt.timestep) for curr,target,pid in zip(joint_positions,target_angles,pid_controllers)]
    print("Control effort:", control_effort)
    print('timestep:',sim.model.opt.timestep)
    sim.data.ctrl[joint_ids] = control_effort
    
    sim.step()

    if viewer._quit:
        break

plt.figure()
plt.plot(pid_controllers[1].tracking_errors)
plt.show()