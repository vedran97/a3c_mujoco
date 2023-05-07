from sim_v import *
from A2C_torch import *
from tqdm import tqdm
init,traj = getTrajAndInitJointAngles()
state_size = traj.shape[0]+200
action_size = 3
agents = []
alphas = [0.01,0.01,0.01,0.01,0.01,0.01]*1000
best_errors = [1e1000,1e1000,1e1000,1e1000,1e1000,1e1000]
controller_range = 6
episodes = 1000000
settling = [0,0,0,0,0,0]

INIT_GAINS = [[200,0,10],[800,0,100],[400,0,70],[200,0,30],[80,10,30],[50,0,10]]
pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]

## create agents
for i in range(controller_range):
    agents.append(A2CPIDTuner(state_size, action_size, load_model=False))

## set alphas for controller
for ctrl,alpha in zip(pid_controllers,alphas):
    ctrl.set_alpha(alpha)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dones = [False for i in range(controller_range)]
## train   
states = [np.zeros(state_size),np.zeros(state_size),np.zeros(state_size),np.zeros(state_size),np.zeros(state_size),np.zeros(state_size)]
for episode in tqdm(range(episodes)):
    ## reset controllers:
    for controller in pid_controllers:
        controller.reset_controller()
    ## reset sim
    actions = []
    ## Get Actions
    for idx,agent in enumerate(agents):
        state = np.reshape(states[idx], [1, state_size])
        state_tensor = torch.tensor(state).to(torch.float32).to(device=device)
        actions.append(agents[idx].get_action(state_tensor))
    ## Update PID gains for joints
    for idx,action in enumerate(actions):
        # print("Joint : ",idx+1)
        # print("action : ",action)
        delta_kp = pid_controllers[idx].alpha * action[0]
        delta_ki = pid_controllers[idx].alpha * action[1]
        delta_kd = pid_controllers[idx].alpha *0.01* action[2]
        if not dones[idx]:
            pid_controllers[idx].update_gains(delta_kp,delta_ki,delta_kd)
    ## Simulate once
    current_pos = simOnce(render=False,plot=False,pid_controllers=pid_controllers)
    ## Collect rewards for each controller which is tuned
    next_states = []
    for idx,action in enumerate(actions):
        error_sum = np.sum(np.abs(np.array(pid_controllers[idx].tracking_errors, dtype=np.float32)))
        reward = -error_sum
        if error_sum < 0.005 * (state_size -1):
            reward = 10
            dones[idx] = True
        if error_sum < best_errors[idx]:
            best_errors[idx] = error_sum
        ## next state is current points used for that joint,in the simulator:
        next_state = np.array(current_pos)[:,idx]
        next_states.append([next_state, reward, dones[idx], error_sum])
        if episode%10 == 0 :
            print("error sum:{} , Joint:{}".format(error_sum,idx+1))
    ## Train each agent
    for next_state_itr,agent,action in zip(next_states,agents,actions):
        next_state = next_state_itr[0]
        next_state = np.reshape(next_state, [1, state_size])
        next_state_tensor = torch.tensor(next_state).to(torch.float32).to(device=device)
        reward = next_state_itr[1]
        done = next_state_itr[2]
        agent.train_model(state_tensor, action, reward, next_state_tensor, done)
    ## Update state
    for idx,next_state_itr in enumerate(next_states):
        states[idx] = next_state_itr[0]
    for idx,done in enumerate(dones):
        if done:
            print('Done for joint:{}'.format(idx+1))
            print('P:{},I:{},D:{}'.format(pid_controllers[idx].Kp,pid_controllers[idx].Ki,pid_controllers[idx].Kd))
    if (episode+1)%50 == 0 :
        for idx,controllers in enumerate(pid_controllers):
            print("Joint:{}".format(idx+1))
            print("P:{},I:{},D:{}".format(controllers.Kp,controllers.Ki,controllers.Kd))
    if episode == episodes-1:
        for idx,controllers in enumerate(pid_controllers):
            print("Joint:{}".format(idx+1))
            print("P:{},I:{},D:{}".format(controllers.Kp,controllers.Ki,controllers.Kd))
