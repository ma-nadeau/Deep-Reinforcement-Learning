import gym  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore
import gymnasium as gym  # type: ignore
import ale_py  # type: ignore
import numpy as np
import random
from collections import deque
import os


print("\n\n\n==========================================================\n\n\n")

print("Code Running")

print("\n\n\n==========================================================\n\n\n")


############################################################################################################

class QNetwork(nn.Module):
    # Q-Network
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        """
        Initializes the QNetwork.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output action.
            hidden_dim (int, optional): Number of units in the hidden layers. Defaults to 256.
            num_layers (int, optional): Number of hidden layers. Defaults to 2.
        """
        super(QNetwork, self).__init__()

        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

        nn.init.uniform_(self.layer1.weight, -0.001, 0.001)
        nn.init.uniform_(self.layer2.weight, -0.001, 0.001)
        nn.init.uniform_(self.layer3.weight, -0.001, 0.001)

    # forward pass
    def forward(self, x):
        """
        Forward pass through the Q-network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output Q-values tensor.
        """
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

############################################################################################################


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        """
        Initializes the buffer with a specified capacity.

        Args:
            capacity (int, optional): The maximum number of elements the buffer can hold. Defaults to 1,000,000.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the replay buffer.

        Args:
            state (object): The current state.
            action (object): The action taken.
            reward (float): The reward received after taking the action.
            next_state (object): The next state after taking the action.
            done (bool): A flag indicating whether the episode is done.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the replay buffer.

        Args:
             batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing arrays of states, actions, rewards, next states, and done flags.
        """

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
        np.array(state, dtype=np.float32),
        np.array(action),
        np.array(reward, dtype=np.float32),
        np.array(next_state, dtype=np.float32),
        np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


############################################################################################################

class ExpectedSarsaAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.01,
        gamma=0.99,
        epsilon=0.1,
        hidden_dim=256,
        num_layers=2,
    ):
        """
        Initializes the ExpectedSarsaAgent.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output action.
            lr (float, optional): Learning rate. Defaults to 0.01.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            epsilon (float, optional): Exploration rate. Defaults to 0.1.
            hidden_dim (int, optional): Number of units in the hidden layers. Defaults to 256.
            num_layers (int, optional): Number of hidden layers. Defaults to 2.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)

    def select_action(self, state):
        """
        Selects an action based on the current state using an epsilon-greedy policy.

        Args:
            state (numpy.ndarray): The current state of the environment.

        Returns:
            int: The selected action.
        """
        # epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-network based on the agent's experience.

        Args:
            state (array-like): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (array-like): The state of the environment after taking the action.
            done (bool): A flag indicating whether the episode has ended (True if done, False otherwise).

        Returns:
            None
        """
        # convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(
            self.device
        )  # 1 if done, 0 otherwise

        q_values = self.q_network(state)  # Q-values

        q_value = q_values[action]  # Q-value of the action taken
        next_q_values = self.q_network(
            next_state
        ).detach()  # Q-values of the next state

        # E[Q(s', a')] = (1 - epsilon) * max_a' Q(s', a') + epsilon * sum_a' Q(s', a') / |A|
        expected_q = (
            1 - self.epsilon
        ) * next_q_values.max() + self.epsilon * next_q_values.mean()
        target = (
            reward + (1 - done) * self.gamma * expected_q
        )  # target = r + gamma * E[Q(s', a')]

        loss = torch.nn.functional.mse_loss(
            q_value, target
        )  # = (q_value - target) ** 2

        self.optimizer.zero_grad()  # reset gradients
        loss.backward()  # backpropagation
        self.optimizer.step()  # update weights
    
    def batch_update(self, s_batch, a_batch, r_batch, ns_batch, d_batch):
        """
        Performs a batch update using sampled experiences from the replay buffer.
        """
        # Convert batches to tensors
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        a_batch = torch.tensor(a_batch).to(self.device)
        r_batch = torch.tensor(r_batch).to(self.device)
        ns_batch = torch.FloatTensor(ns_batch).to(self.device)
        d_batch = torch.tensor(d_batch, dtype=torch.float32).to(self.device)

        # Get Q-values for all states in the batch
        q_values = self.q_network(s_batch)

        # Select the Q-values for the taken actions
        q_value = q_values.gather(1, a_batch.unsqueeze(1))  # (batch_size, 1)

        # Get the max Q-values for the next states in the batch
        next_q_values = self.q_network(ns_batch).detach()
        max_next_q = next_q_values.max(1)[0]  # (batch_size, )

        # Calculate the expected Q-value for each experience in the batch
        expected_q = (1 - self.epsilon) * max_next_q + self.epsilon * next_q_values.mean(dim=1)

        # Compute the target for each experience in the batch
        target = r_batch + (1 - d_batch) * self.gamma * expected_q

        # Compute the loss (MSE loss)
        loss = torch.nn.functional.mse_loss(q_value.squeeze(), target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


############################################################################################################


class QLearningAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.01,
        gamma=0.99,
        epsilon=0.1,
        hidden_dim=256,
        num_layers=2,
    ):
        """
        Initializes the QLearningAgent.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output action.
            lr (float, optional): Learning rate. Defaults to 0.01.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            epsilon (float, optional): Exploration rate. Defaults to 0.1.
            hidden_dim (int, optional): Number of units in the hidden layers. Defaults to 256.
            num_layers (int, optional): Number of hidden layers. Defaults to 2.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)

    def select_action(self, state):
        """
        Selects an action based on the current state using an epsilon-greedy policy.

        Args:
            state (numpy.ndarray): The current state of the environment.

        Returns:
            int: The selected action.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-network based on the agent's experience.

        Args:
            state (array-like): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (array-like): The state of the environment after taking the action.
            done (bool): A flag indicating whether the episode has ended (True if done, False otherwise).

        Returns:
            None
        """
        # convert to tensors

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(
            self.device
        )  # done = 1 if the episode is done, 0 otherwise

        q_values = self.q_network(state)  # get the q-values
        q_value = q_values[action]  # get the q-value for the action taken
        next_q_values = self.q_network(
            next_state
        ).detach()  # get the q-values for the next state
        max_next_q = next_q_values.max()  # get the maximum q-value

        target = (
            reward + (1 - done) * self.gamma * max_next_q
        )  # target = reward + gamma * max_next_q
        loss = torch.nn.functional.mse_loss(
            q_value, target
        )  # loss = (q_value - target)^2

        # optimize the model
        self.optimizer.zero_grad()  # set the gradients to zero
        loss.backward()  # compute the gradients
        self.optimizer.step()  # update the weights

    def batch_update(self, s_batch, a_batch, r_batch, ns_batch, d_batch):
        """
        Perform a batch update using sampled experiences from the replay buffer.
        """
        # Convert batches to tensors
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        a_batch = torch.tensor(a_batch).to(self.device)
        r_batch = torch.tensor(r_batch).to(self.device)
        ns_batch = torch.FloatTensor(ns_batch).to(self.device)
        d_batch = torch.tensor(d_batch, dtype=torch.float32).to(self.device)

        # Get current Q-values for all states in the batch
        q_values = self.q_network(s_batch)

        # Select the Q-values for the taken actions
        q_value = q_values.gather(1, a_batch.unsqueeze(1))  # (batch_size, 1)

        # Get the max Q-value for the next states in the batch
        next_q_values = self.q_network(ns_batch).detach()
        max_next_q = next_q_values.max(1)[0]  # (batch_size, )

        # Compute the target (TD target) for each experience
        target = r_batch + (1 - d_batch) * self.gamma * max_next_q

        # Compute the loss
        loss = torch.nn.functional.mse_loss(q_value.squeeze(), target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

############################################################################################################

# Common training function for QLearning and ExpectedSarsa Agents
def train(
    env_name,
    agent_class,
    episodes=5,
    epsilon=0.1,
    lr=0.01,
    trials=1,
    use_replay=False,
    batch_size=64,
):
    """
    Train a reinforcement learning agent in a specified environment.

    Args:
        env_name (str): The name of the environment to train in.
        agent_class (class): The class of the agent to be trained.
        episodes (int, optional): The number of episodes to train for each trial. Defaults to 5.
        epsilon (float, optional): The exploration rate for the agent. Defaults to 0.1.
        lr (float, optional): The learning rate for the agent. Defaults to 0.01.
        trials (int, optional): The number of trials to run. Defaults to 1.
        use_replay (bool, optional): A flag indicating whether to use a replay buffer. Defaults to False.
    Return:
        tuple: A tuple containing the mean and standard deviation of rewards across trials.
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(
        f"Env Made, Starting to Train: {env_name} with {'replay' if use_replay else 'no replay'} buffer, epsilon={epsilon}, lr={lr}"
    )
    all_rewards = []
    for trial in tqdm(range(trials), desc="Trials", leave=False):
        print(f"Trial: {trial+1}")
        agent = agent_class(state_dim, action_dim, lr=lr, epsilon=epsilon)
        buffer = ReplayBuffer() if use_replay else None
        rewards = []
        pbar = tqdm(total=episodes, desc="Episodes", unit="episode", leave=False)
        for episode in range(episodes):
            if episode % 100 == 0 and episode != 0:
                pbar.update(100)
            state, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                if isinstance(state, tuple):
                    state = state[0]
                if env_name == "ALE/Assault-ram-v5":
                    state = state / 255.0
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                if env_name == "ALE/Assault-ram-v5":
                    next_state = next_state / 255.0
                done = terminated or truncated
                total_reward += reward

                if use_replay:
                    buffer.push(state, action, reward, next_state, done)
                    state = next_state

                    # Only update if enough samples are in the buffer
                    if len(buffer) >= batch_size:
                        s_batch, a_batch, r_batch, ns_batch, d_batch = buffer.sample(
                            batch_size
                        )
                        agent.batch_update(s_batch, a_batch, r_batch, ns_batch, d_batch)

                else:
                    agent.update(state, action, reward, next_state, done)
                    state = next_state

            rewards.append(total_reward)
        pbar.close()
        all_rewards.append(rewards)
    env.close()
    return np.mean(all_rewards, axis=0), np.std(all_rewards, axis=0)

############################################################################################################


def render_Acrobot():
    # env = gym.make("Acrobot-v1", render_mode="human")
    env = gym.make("Acrobot-v1", render_mode="human")
    state, _ = env.reset()

    for _ in range(200):  # Run for a few steps to visualize
        action = env.action_space.sample()  # Take a random action
        state, reward, done, _, _ = env.step(action)
        env.render()

        if done:
            state, _ = env.reset()
    print("rendered")


# render_Acrobot():


def render_Assault():
    # env = gym.make("Acrobot-v1", render_mode="human")
    env = gym.make("ALE/Assault-arm-v5", render_mode="human")
    state, _ = env.reset()

    for _ in range(200):  # Run for a few steps to visualize
        action = env.action_space.sample()  # Take a random action
        state, reward, done, _, _ = env.step(action)
        env.render()

        if done:
            state, _ = env.reset()
    print("rendered")


############################################################################################################


# Function for plotting results
def plot_results(results_q, results_esarsa, env_name, use_replay, epsilon, lr):
    plt.figure(figsize=(12, 6))

    # Q-Learning plot
    q_mean, q_std = results_q
    linestyle = "-" if lr == 0.25 else "--" if lr == 0.125 else ":"
    plt.plot(
        q_mean,
        label=f"Q-Learning ε={epsilon}, α={lr}",
        color="green",
        linestyle=linestyle,
    )
    plt.fill_between(
        range(len(q_mean)), q_mean - q_std, q_mean + q_std, color="green", alpha=0.3
    )

    # Expected SARSA plot
    esarsa_mean, esarsa_std = results_esarsa
    plt.plot(
        esarsa_mean,
        label=f"Expected SARSA ε={epsilon}, α={lr}",
        color="red",
        linestyle=linestyle,
    )
    plt.fill_between(
        range(len(esarsa_mean)),
        esarsa_mean - esarsa_std,
        esarsa_mean + esarsa_std,
        color="red",
        alpha=0.3,
    )

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(
        f"{env_name} {'with' if use_replay else 'without'} Replay Buffer\nε={epsilon}, α={lr}"
    )
    plt.legend()

    # Ensure results directory exists
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    # Construct unique filename dynamically
    if env_name == "ALE/Assault-ram-v5":
        env_name = "Assault-ram-v5"
    filename = (
        f"{env_name}_{'replay' if use_replay else 'no_replay'}_eps{epsilon}_lr{lr}.png"
    )
    filepath = os.path.join(results_dir, filename)

    # Save the figure
    plt.savefig(filepath)
    plt.close()


# Function for running experiments progressively
def run_experiment(
    environments, agent_classes, use_replay_options, epsilons, learning_rates
):
    for env_name in environments:
        for use_replay in use_replay_options:
            for epsilon in tqdm(
                epsilons, desc=f"Epsilons for {env_name}", unit="epsilon"
            ):
                for lr in tqdm(
                    learning_rates,
                    desc=f"Learning Rates for {env_name}, epsilon={epsilon}",
                    unit="lr",
                    leave=False,
                ):
                    # Train Q-Learning and Expected SARSA agents
                    print(
                        f"Training {env_name} with {'replay' if use_replay else 'no replay'} buffer, epsilon={epsilon}, lr={lr}"
                    )
                    q_mean, q_std = train(
                        env_name,
                        agent_classes[0],
                        epsilon=epsilon,
                        lr=lr,
                        episodes=1000,
                        trials=10,
                        use_replay=use_replay,
                    )
                    esarsa_mean, esarsa_std = train(
                        env_name,
                        agent_classes[1],
                        epsilon=epsilon,
                        lr=lr,
                        episodes=1000,
                        trials=10,
                        use_replay=use_replay,
                    )

                    # Plot and save immediately after training this configuration
                    plot_results(
                        (q_mean, q_std),
                        (esarsa_mean, esarsa_std),
                        env_name,
                        use_replay,
                        epsilon,
                        lr,
                    )


# Running experiments
if __name__ == "__main__":
    
    # render_Assault() # Render Assault
    learning_rates = [ 
        0.0001,
        0.001,
        0.01,
    ]  # https://edstem.org/us/courses/71533/discussion/6304331
    epsilons = [0.0625, 0.125, 0.25]
    environments = ["ALE/Assault-ram-v5"]
    agent_classes = [ExpectedSarsaAgent, QLearningAgent]
    use_replay_options = [True]
    run_experiment(
        environments, agent_classes, use_replay_options, epsilons, learning_rates
    )
