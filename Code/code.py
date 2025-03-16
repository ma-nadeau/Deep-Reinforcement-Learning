import gym # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt # type: ignore


class QNetwork(nn.Module):
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
            
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.model = nn.Sequential(*layers)
        
        self.initialize_weights()
    
    def initialize_weights(self, initial_weigths=0.0001):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -initial_weigths, initial_weigths)
                nn.init.uniform_(m.bias, -initial_weigths, initial_weigths)
                
    def forward(self, x):
        return self.model(x)
    