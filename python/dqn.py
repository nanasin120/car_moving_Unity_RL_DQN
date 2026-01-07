import time
from collections import deque, namedtuple
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

learning_rate = 0.001
batch_size = 64
gamma = 0.99
tau = 0.01
experiences = namedtuple('experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

class DQNNet(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(DQNNet, self).__init__()

        self.fc1 = nn.Linear(input_layer, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.fc3(x)
        return out
    
class DQNAgent(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(DQNAgent, self).__init__()

        self.q_net = DQNNet(input_layer, output_layer).to(device)
        self.target_q_net = DQNNet(input_layer, output_layer).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.memory_buffer = deque(maxlen=100_000)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss().to(device)
        self.epsilon = 1.0


    def act(self, state):
        self.q_net.eval()
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        
        # state는 [31]임, [1, 31]로 바꿔야함, 그래서 unsqueeze해줌
        state = torch.from_numpy(state).float().unsqueeze(0).to(device=device)
        with torch.no_grad():
            Q_value = self.q_net(state)
        # action은 유니티에 들어갈거임, item없이 하면 tensor가 가게 되고 오류남
        action = torch.argmax(Q_value).item() 
        self.q_net.train()
        return action

    def learn(self, state, action, reward, next_state, done): 
        #print(len(self.memory_buffer))
        self.q_net.train()
        self.memory_buffer.append(experiences(state, action, reward, next_state, done))
        if len(self.memory_buffer ) < 1_000 : return

        batch = random.sample(self.memory_buffer, batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in batch])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch]).astype(np.uint8)).float().to(device)
        
        with torch.no_grad():
            max_qsa = torch.max(self.target_q_net(next_states), dim=1)[0].unsqueeze(1)
            y_target = rewards + gamma * max_qsa * (1 - dones)

        q_values = self.q_net(states).gather(1, actions)

        loss = self.loss(q_values, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_pm, q_pm in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_pm.data.copy_(target_pm * (1 - tau) + q_pm * tau)

        #print(f"현재 loss : {loss}")

        self.epsilon = max(0.01, self.epsilon * 0.995)

    def save(self, episode):
        torch.save(self.q_net.state_dict(), f'save/q_net_{episode}.pth')
        torch.save(self.target_q_net.state_dict(), f'save/target_q_net_{episode}.pth')