import gym # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt # type: ignore


print("\n\n\n==========================================================\n\n\n")

print("Code Running")

print("\n\n\n==========================================================\n\n\n")


class QNetwork(nn.Module):
    # Q-Network
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        super(QNetwork, self).__init__()
        
        layers = []
        # input layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        # hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        # set model
        self.model = nn.Sequential(*layers)
        # initialize weights
        self.initialize_weights()
    
    # initialize weights
    def initialize_weights(self, initial_weigths=0.0001):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -initial_weigths, initial_weigths)
                nn.init.uniform_(m.bias, -initial_weigths, initial_weigths)
    
    # forward pass
    def forward(self, x):
        return self.model(x)
    
    
class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)
    
class ExpectedSarsaAgent: 
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99, epsilon=0.1, hidden_dim=256, num_layers=2):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers) #.to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
    
    def select_action(self, state): 
        # epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        # convert to tensors
        tate = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done, dtype=torch.float32) # 1 if done, 0 otherwise
        
        q_values = self.q_network(state) # Q-values
        q_value = q_values[action] # Q-value of the action taken
        next_q_values = self.q_network(next_state).detach() # Q-values of the next state
        
        # E[Q(s', a')] = (1 - epsilon) * max_a' Q(s', a') + epsilon * sum_a' Q(s', a') / |A|
        expected_q = (1 - self.epsilon) * next_q_values.max() + self.epsilon * next_q_values.mean() 
        target = reward + (1 - done) * self.gamma * expected_q # target = r + gamma * E[Q(s', a')]
        
        loss = nn.MSELoss()(q_value, target) # = (q_value - target) ** 2
        
        self.optimizer.zero_grad() # reset gradients
        loss.backward() # backpropagation
        self.optimizer.step() # update weights
        
class QLearningAgent: 
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2, gamma=0.99, learning_rate=0.001, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                return torch.argmax(q_values).item()
            
    def update(self, state, action, reward, next_state, done):
        # convert to tensors
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done, dtype=torch.float32) # done = 1 if the episode is done, 0 otherwise
        
        
        q_values = self.q_network(state) # get the q-values
        q_value = q_values[action] # get the q-value for the action taken
        next_q_values = self.q_network(next_state).detach() # get the q-values for the next state
        max_next_q = next_q_values.max() # get the maximum q-value
        
        target = reward + (1 - done) * self.gamma * max_next_q # target = reward + gamma * max_next_q
        loss = nn.MSELoss()(q_value, target) # loss = (q_value - target)^2
        
        # optimize the model
        self.optimizer.zero_grad() # set the gradients to zero
        loss.backward() # compute the gradients
        self.optimizer.step() # update the weights


# Common training function for QLearning and ExpectedSarsa Agents
def train(env_name, agent_class, episodes=5, epsilon=0.1, lr=0.01, trials=1):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    all_rewards = []
    for trial in range(trials):
        agent = agent_class(state_dim, action_dim, lr=lr, epsilon=epsilon)
        rewards = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            rewards.append(total_reward)
        all_rewards.append(rewards)
    
    env.close()
    return np.mean(all_rewards, axis=0), np.std(all_rewards, axis=0)
epsilons = [0.01 ] #, 0.001, 0.0001]
learning_rates = [0.25] #, 0.125, 0.0625]

# Function for plotting results
def plot_results(results_q, results_esarsa, env_name, use_replay):
        plt.figure(figsize=(12, 6))
        for (epsilon, lr), (q_mean, q_std) in results_q.items():
            linestyle = '-' if lr == 0.25 else '--' if lr == 0.125 else ':'
            plt.plot(q_mean, label=f"Q-Learning ε={epsilon}, α={lr}", color='green', linestyle=linestyle)
            plt.fill_between(range(len(q_mean)), q_mean - q_std, q_mean + q_std, color='green', alpha=0.3)
        for (epsilon, lr), (esarsa_mean, esarsa_std) in results_esarsa.items():
            linestyle = '-' if lr == 0.25 else '--' if lr == 0.125 else ':'
            plt.plot(esarsa_mean, label=f"Expected SARSA ε={epsilon}, α={lr}", color='red', linestyle=linestyle)
            plt.fill_between(range(len(esarsa_mean)), esarsa_mean - esarsa_std, esarsa_mean + esarsa_std, color='red', alpha=0.3)
        
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title(f"{env_name} {'with' if use_replay else 'without'} Replay Buffer")
        plt.legend()
        plt.savefig("../Results/test.png")

    # Function for running experiments
def run_experiment(environments, agent_classes, use_replay_options):
        for env_name in environments:
            for use_replay in use_replay_options:
                results_q = {}
                results_esarsa = {}
                for epsilon in epsilons:
                    for lr in learning_rates:
                        q_mean, q_std = train(env_name, agent_classes[0], epsilon=epsilon, lr=lr, episodes=1000, trials=50)
                        esarsa_mean, esarsa_std = train(env_name, agent_classes[1], epsilon=epsilon, lr=lr, episodes=1000, trials=50)
                        results_q[(epsilon, lr)] = (q_mean, q_std)
                        results_esarsa[(epsilon, lr)] = (esarsa_mean, esarsa_std)
                plot_results(results_q, results_esarsa, env_name, use_replay)

    # Running experiments
if __name__ == "__main__":
        environments = ["Acrobot-v1", "ALE/Assault-ram-v5"]
        agent_classes = [QLearningAgent, ExpectedSarsaAgent]
        use_replay_options = [False, True]
        run_experiment(environments, agent_classes, use_replay_options)
