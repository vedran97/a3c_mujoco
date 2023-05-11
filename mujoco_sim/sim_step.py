from agent.agent import *
import numpy as np    
from os import path
import csv
from controller import PIDController
import mujoco_py
from circular_trajectory import *
import sys
num_episodes = 500
# HYPERPARAMETERS BELOW
gamma = 0.75  # discount factor for rewards
learningRate = 1e-4  # learning rate for actor and critic networks ## j1, j2 = 2e-4
tau = 0.005  # tracking parameter used to update target networks slowly
actionSigma = 0.1  # contributes noise to deterministic policy output
trainingSigma = 0.2  # contributes noise to target actions
trainingClip = 0.5  # clips target actions to keep them close to true actions
miniBatchSize = 100  # how large a mini-batch should be when updating
policyDelay = 2  # how many steps to wait before updating the policy
resume = True  # resume from previous checkpoint if possible?
render = False  # render out the environment on-screen?
observation_size = state_size = 3
action_size = 1
## set env name
envName = "mujoco_a3c"
zero_epsilon = 1e-6
## initialize PID controllers
INIT_GAINS = [[4859.5810546875,1.00020770332776e-06,8.67470932006836],
              [24239.791015625,9.863824743661098e-07,70.85529327392578],
              [400,0+zero_epsilon,70],
              [200,0+zero_epsilon,30],
              [80,10,30],
              [50,0+zero_epsilon,10]]
## effort clamp for J1 = 20
## Best learnt gains for J1 = 4859.5810546875,1.00020770332776e-06,8.67470932006836 ## Initial = 200,0+zero_epsilon,10
## Best learnt gains for J2 = 24239.791015625,9.863824743661098e-07,70.85529327392578 ## Initial = 800,0+zero_epsilon,100
## Best learnt gains for J3 = 400,0+zero_epsilon,70 ## Initial = 400,0+zero_epsilon,70
## J1 reward chosen to exit the RL = 0.29
## J2 reward chosen to exit the RL = 0.29 // exited before this because this is the best learnt reward
J_REWARD_CUTOFF = 0.29
J1_EFFORT_CLAMP = 30
J2_EFFORT_CLAMP = 70
pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]

class PIDMujocoEnv:
    def __init__(self,controller:PIDController,idx,render=False):
        self.render = render
        self.idx = idx
        self.controller = controller
        self.done = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.observation_space = np.array([0,0,0])
        self.action_space = np.array([0]) ## action space = 1

        self.observation_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        print("observation_dim: ",self.observation_dim)
        print("action_dim: ",self.action_dim)
        self.name = "mujoco_a3c"

        # Load the XML model file
        self.model_path = path.join('../mujoco_model', 'model.xml')
        self.model = mujoco_py.load_model_from_path(self.model_path)

        # Create a simulation instance
        self.sim = mujoco_py.MjSim(self.model)
        ## Load trajectory
        initial_joint_angles, traj = getTrajAndInitJointAngles()
        # Added extra points for looking at settling error
        for i in range(10):
            traj = np.vstack((traj,traj[-1]))
        self.initial_joint_angles = initial_joint_angles
        self.traj = traj
        ## get names of joints and bodies
        # Print the names of all the joints
        # print("Joint names:")
        joint_names = []
        for i in range(self.model.njnt):
            joint_names.append(self.model.joint_id2name(i))
            # print(model.joint_id2name(i))

        # Print the names of all the bodies
        # print("Body names:")
        body_names = []
        for i in range(self.model.nbody):
            body_names.append(self.model.body_id2name(i))
            # print(model.body_id2name(i))

        # Get the IDs of the joint and end effector bodies
        joint_id = self.sim.model.joint_name2id(joint_names[-1])
        ee_id = self.sim.model.body_name2id(body_names[-1])

        # Get the IDs of the joint bodies
        self.joint_ids = [self.sim.model.joint_name2id(name) for name in joint_names]

        ## Set initial joint angles
        self.sim.data.qpos[self.joint_ids] = initial_joint_angles

        if self.render:
            # Create a viewer instance
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.render()
        self.traj_index = 0

    def init_actor(self,actor:Actor):
        gains = [self.controller.Kp, self.controller.Ki, self.controller.Kd]
        actor.init_weights(gains)
        return actor
    def call_render(self):
        if self.render:
            self.viewer.render()
    def reset(self):
        self.controller.reset_controller()
        for controller in pid_controllers:
            controller.reset_controller()
        self.traj_index = 0
        self.sim.data.qpos[self.joint_ids] = self.initial_joint_angles
        if self.render:
            self.viewer.render()
        self.done = False
        return np.zeros(state_size)
    
    def step(self,state,agent:Agent,gains_bench,r_bench,eval=False):
        if self.render:
            self.viewer.render()
        ## target angle , set of 6 target angles
        target_angle = self.traj[self.traj_index]
        ## current angle
        current_angles = self.sim.data.qpos[self.joint_ids]
        if self.traj_index != 0:
            ## pid gains are weights of actor
            k_p,k_i,k_d = np.exp(agent.actor.weight.data.cpu().numpy()).tolist()
            ## set gains
            self.controller.set_gains(k_p,k_i,k_d)
            pid_controllers[self.idx] =self.controller
            ## control effort and action are the same things
            action = agent.getNoiseAction(torch.FloatTensor(state).to(self.device)) 
        ## find control effort
        control_effort = [pid.compute(curr,target,self.sim.model.opt.timestep) for curr,target,pid in zip(current_angles,target_angle,pid_controllers)]
        if self.traj_index != 0:
            control_effort[self.idx] = action
        ## Clamp control effort
        effort_clamp = 70
        if control_effort[self.idx] < -effort_clamp:
            print("clamped effort",control_effort[self.idx])
            control_effort[self.idx] = -effort_clamp
        elif control_effort[self.idx] > effort_clamp:
            print("clamped effort",control_effort[self.idx])
            control_effort[self.idx] = effort_clamp

        if control_effort[self.idx] == -effort_clamp or control_effort[self.idx] == effort_clamp:
            control_was_clamped = True
        else:
            control_was_clamped = False
        
        self.sim.data.ctrl[self.joint_ids] = control_effort
        ## Apply step
        self.sim.step()
        ## Find RR 
        # rr = np.sum(np.array(self.controller.tracking_errors)**2)
        ## Apply supervisor logic
        # if(rr>1.5*r_bench) and eval==False:
        #     ## RESET PID CONTROLLER, RESET actor weight
        #     print("\r\nSUPERVISOR ACTION\r\n")
        #     p_bench,i_bench,d_bench = gains_bench
        #     self.controller.set_gains(p_bench,i_bench,d_bench)
        #     pid_controllers[self.idx] =self.controller
        #     agent.actor = self.init_actor(agent.actor)
        ## Get specific controller's next state
        ## Get current angles for the joint of concern
        current_angle = self.sim.data.qpos[self.joint_ids][self.idx]
        ## Calculate errors, but dont apply anything,  this is the next state
        next_state = self.controller.calculate_errors(current_angle,self.traj[self.traj_index+1][self.idx],self.sim.model.opt.timestep)
        ## Calculate reward LQG loss
        # print("error:",self.controller.current_error**2)
        # print("effort:",control_effort[self.idx]**2)
        reward = -1*(((self.controller.current_error)**2)+1e-6*(control_effort[self.idx]**2))
        if control_was_clamped: ## we dont want control saturation!!
            print("control was clamped")
            reward -= 100
            print('reward:',reward)
        # print("reward:",reward)
        ## Increment trajectory index
        self.traj_index+=1
        ## done condition
        done = False
        if self.traj_index == (len(self.traj)-1):
            done = True
        return next_state,reward,done,0
 
    def runTrajectoryOnce(self,agent):
        self.reset()
        control_efforts = []
        errors = []
        for target_angles in self.traj:
            if self.render:
                self.viewer.render()
            joint_positions = self.sim.data.qpos[self.joint_ids]
            ## Gains are weights of actor
            k_p,k_i,k_d = np.exp(agent.actor.weight.data.cpu().numpy()).tolist()
            ## set gains
            self.controller.set_gains(k_p,k_i,k_d)
            pid_controllers[self.idx] =self.controller
            ## find control effort
            control_effort = [pid.compute(curr,target,self.sim.model.opt.timestep) for curr,target,pid in zip(joint_positions,target_angles,pid_controllers)]
            ## get specific controller's state
            state = self.controller.get_state()
            ## get specific controller's action
            action = agent.getNoiseAction(state)
            ## control effort is action,action is control effort
            control_effort[self.idx] = action
            ## apply control effort
            self.sim.data.ctrl[self.joint_ids] = control_effort
            ## sim_step
            self.sim.step()
            # store control effort for this simulation_step
            control_efforts.append(control_effort)


def train(num_episodes=2,ctrl_indx = 2):
    for trial in range(5):
        pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]
        env = PIDMujocoEnv(pid_controllers[ctrl_indx],ctrl_indx,render=render)
        env.name = envName + "_" + str(trial) + "_" + str(ctrl_indx+1)
        csvName = env.name + "-data.csv"
        agent = Agent(env, learningRate, gamma, tau)
        env.init_actor(agent.actor)
        env.init_actor(agent.actor_target)
        state = env.reset()
        step = 0
        runningReward = None
        ## for supervisor, we need to first calculate a benchmark reward for each operation step
        env.runTrajectoryOnce(agent)
        gains_bench = INIT_GAINS[env.idx]
        r_bench = np.sum(np.array(env.controller.tracking_errors)**2)
        print("r_bench:",r_bench)
        env.reset()
        # determine the last episode if we have saved training in progress
        numEpisode = 0
        if path.exists(csvName):
            fileData = list(csv.reader(open(csvName)))
            lastLine = fileData[-1]
            numEpisode = int(lastLine[0])

        while numEpisode <= num_episodes:
            # # choose an action from the agent's policy
            action = agent.getNoiseAction(state)
            # take a step in the environment and collect information
            nextState, reward, done, _ = env.step(state,agent,gains_bench,r_bench,True)   ## literally take 1 step in the sim space.
            # store data in buffer
            agent.buffer.store(state, action, reward, nextState, done)

            if done:
                numEpisode += 1
                # evaluate the deterministic agent on a test episode
                sumRewards = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.getDetAction(state)
                    nextState, reward, done, info = env.step(state,agent,gains_bench,r_bench,True)
                    if render:
                        env.call_render()
                    state = nextState
                    sumRewards += reward
                state = env.reset()
                # keep a running average to see how well we're doing
                runningReward = (
                    sumRewards
                    if runningReward is None
                    else runningReward * 0.99 + sumRewards * 0.01
                )
                # log progress in csv file
                kp_learned, ki_learned, kd_learned = np.exp(agent.actor.weight.data.cpu().numpy()).tolist()
                fields = [numEpisode, sumRewards, runningReward,kp_learned, ki_learned, kd_learned ,env.idx]
                with open(env.name + "-data.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                # agent.save()
                # print episode tracking
                print(
                    f"episode {numEpisode:6d} --- "
                    + f"total reward: {sumRewards:7.2f} --- "
                    + f"running average: {runningReward:7.2f}",
                    flush=True,
                )
                print('Gains:{},{},{}'.format(kp_learned,ki_learned,kd_learned))
                if(np.abs(sumRewards)<J_REWARD_CUTOFF):
                    break
            else:
                state = nextState
            step += 1
            shouldUpdatePolicy = step % policyDelay == 0
            agent.update(miniBatchSize, trainingSigma, trainingClip, shouldUpdatePolicy)

if __name__ == "__main__":
    arg = int(sys.argv[1])
    print('The argument is:', arg)
    train(num_episodes,ctrl_indx=arg)