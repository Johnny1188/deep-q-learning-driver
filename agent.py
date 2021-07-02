import numpy as np
import torch
import time

from q_learner import Q_Learner
from replay_buffer import ReplayBuffer
from game.game import Road,STATE_INPUT_DIM

# Defaults
DEVICE = "cpu"
LEARNING_RATE = 1e-4
GAMMA = .98
EXPLOIT_RATIO = 0.95
EXPLOIT_RATIO_DECAY = 1.00004
BATCH_SIZE = 12

class Agent():
    def __init__(self,
        device=DEVICE,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        exploit_ratio=EXPLOIT_RATIO,
        exploit_ratio_decay=EXPLOIT_RATIO_DECAY,
        batch_size=BATCH_SIZE):
        """
        Parameters
        ----------
        device: string
        Whether to utilize GPU or just CPU. (can be either "cuda" or "cpu")

        lr: float
        Learning rate for the deep Q-learning model optimizer.

        gamma: float
        Float between 0 and 1, inclusive. Tells how much the agent weights the future state's Q values.

        exploit_ratio: float
        Ratio of exploiting the learned Q-values instead of exploring (between 0 and 1, inclusive).

        exploit_ratio_decay: float
        Number by which to multiply expoit_ratio each episode (default 1.00004).

        batch_size: integer
        Number of state-action transition after which to update the Q-learning model.
        """
        self.device = device
        self.q_learner = Q_Learner(input_dim=STATE_INPUT_DIM).to(self.device).double()
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.q_learner.parameters(),lr=lr)
        self.exploit_ratio = exploit_ratio
        self.exploit_ratio_decay = exploit_ratio_decay
        self.batch_size = batch_size
    
    def update_q_learner(self):
        states,actions,rewards,next_states,dones = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.tensor(states,dtype=torch.double).to(self.device)
        next_states = torch.tensor(next_states,dtype=torch.double).to(self.device)
        rewards = torch.tensor(rewards,dtype=torch.double).to(self.device)
        actions = torch.tensor(actions,dtype=torch.int64).to(self.device)
        dones = torch.tensor(dones,dtype=torch.float32).to(self.device)

        q_values = self.q_learner(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_values_of_next_states = self.q_learner(next_states).detach().max(1)[0]

        target_q_values = rewards + GAMMA * (q_values_of_next_states - q_values) * 1-dones

        loss = (target_q_values - q_values).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return(loss)

    def train(self,max_episodes=1000,w_exploration=True,road_difficulty=2,silent=False):
        road = Road(difficulty=road_difficulty)
        time_until_crash = []
        loss_history = []
        episode_i = 0

        while episode_i < max_episodes:
            old_state = torch.tensor([road.get_state()],dtype=torch.double).to(self.device)

            action_to_take = np.zeros((4,),dtype=int)
            q_values_of_actions_to_take = self.q_learner(old_state)

            # Explore | exploit
            if np.random.rand() < (self.exploit_ratio*(self.exploit_ratio_decay**episode_i)) or w_exploration == False:
                action_to_take[torch.argmax(q_values_of_actions_to_take[0]).item()] = 1
            else:
                action_to_take[np.random.randint(0,3,size=1)] = 1
            
            reward, done = road.play_step(action_to_take)
            new_state = torch.tensor([road.get_state()],dtype=torch.double).to(self.device)

            self.replay_buffer.push(old_state[0].cpu().numpy(),np.argmax(action_to_take),reward,new_state[0].cpu().numpy(),done)

            if len(self.replay_buffer.buffer) > self.batch_size:
                loss = self.update_q_learner()
                loss_history.append(loss.detach().cpu())

            if done:
                if (episode_i+1) % 30 == 0 and silent != True:
                    if len(loss_history) != 0:
                        print(f"[{episode_i+1}/{max_episodes}] >>> Time traveled until the crash: {road.timer} >>> Loss: {loss_history[-1]}")
                    else:
                        print(f"[{episode_i+1}/{max_episodes}] >>> Time traveled until the crash: {road.timer}")
                
                if (episode_i+1) % 200 == 0:
                    self.q_learner.save_params(silent=silent)

                time_until_crash.append(road.timer)
                episode_i += 1
                road = Road(difficulty=road_difficulty)
        
        return(time_until_crash,loss_history)

    def test(self,num_of_episodes=10,slow_simulation=False,road_difficulty=2,silent=False):
        road_time_accum = 0.
        road_run_lengths = []
        agent_avg_speeds = []

        with torch.no_grad():
            for _ in range(num_of_episodes):
                agent_speed_accum = 0.
                road = Road(difficulty=road_difficulty)
                done = False

                while done == False:
                    old_state = torch.tensor([road.get_state()],dtype=torch.double).to(self.device)

                    q_values_of_actions_to_take = self.q_learner(old_state)

                    action_to_take = np.zeros((4,),dtype=int)
                    action_to_take[torch.argmax(q_values_of_actions_to_take[0]).item()] = 1
                    
                    _, done = road.play_step(action_to_take)
                    agent_speed_accum += road.agent.speed

                    if slow_simulation:
                        time.sleep(0.02)
                    
                road_time_accum += road.timer
                average_speed = agent_speed_accum//road.step
                road_run_lengths.append(road.timer)
                agent_avg_speeds.append(average_speed)

                if silent != True:
                    print(f">>> Time traveled until a crash: {road.timer} >>> Average speed: {average_speed}")

        if silent != True:
            print(f"\n>>> Average time traveled until a crash: {road_time_accum//num_of_episodes} <<<>>> Average speed across all test episodes: {np.mean(agent_avg_speeds)}")
        return(road_run_lengths,agent_avg_speeds)