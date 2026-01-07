# car_moving_Unity_RL_DQN
유니티 환경을 이용해 원형 트랙에서 자동차가 벽에 부딫히지 않고 움직일수 있게 DQN을 통해 학습시키기

# 프로젝트 소개

유니티의 mlAgent와 Pytorch를 활용해 레이캐스트를 기반으로 하는 자동차를 DQN기법을 통해 학습시킨 프로젝트입니다.

# 1. 기술 스택

Environment : Unity MlAgent

AI Framwork : Pytorch

Algorithm : DQN

# 2. 환경 구축

<img width="502" height="527" alt="image" src="https://github.com/user-attachments/assets/8e787bcf-d8d4-49a2-8839-3c7de5fb3e91" />
  
원형 트랙

<img width="1126" height="379" alt="image" src="https://github.com/user-attachments/assets/05d71863-4e49-4a82-82d6-6f21eb2c1f72" />

자동차 레이캐스트 (총 9개)

자동차 레이캐스트는 (탐지한 객체의 tag값을 one-hot 인코딩한 값, 탐지했는지 여부, 탐지 거리를 0~1로 정규화) 값을 반환합니다.

추가로 자동차의 현재 속도와 방향을 반환하여 총 9 * 3 + 1(속도) + 3(방향 벡터 값) = 31개의 데이터를 학습에 이용합니다.

## 2.1 보상 환경

<img width="539" height="577" alt="image" src="https://github.com/user-attachments/assets/2ffab0e2-578e-4d81-94e9-a6871bb9e030" /> 
<img width="490" height="477" alt="image" src="https://github.com/user-attachments/assets/a7b8ffd5-ebfc-4666-9d40-9c9d69bfcb25" />

CheckPoint에 도달할때마다 50의 보상을 얻으며 CheckPoint는 도달하면 꺼지고 

다음 CheckPoint가 켜지게 설정하여 계속 앞으로 나아가게 학습시켰습니다.

만약 벽에 닿을 경우 -100의 보상을 얻으며 매 순간 (현재 속도 - 2.5) / 5의 보상을 얻어 빠른 속도를 유지하게 했습니다.

## 2.2 행동

```
  switch (action)
  {
      case 1: dir = Vector3.forward; break;
      case 2: dir = -Vector3.forward; break;
      case 3: rot = -1f; break;
      case 4: rot = 1; break;
  }
```
앞, 뒤, 오른쪽, 왼쪽으로 이동하며 action이 0이면 가만히 있게 됩니다.

최대 속도는 10으로 고정해놨습니다.

# 3. 신경망 구조

```
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
```
신경망은 Input(31) -> Hidden(64, 64) -> Output(5)로 이루어져있습니다.

DQNNet 클래스를 통해 두개의 DQN network를 생성할수있습니다.

이 두개의 네트워크는 DQNAgent에서 관리합니다.

```
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
```
DQNAgent를 통해 두개의 네트워크 관리와 학습, 행동 등을 수행합니다.

```
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
```
행동의 경우 epsilon을 통해 탐욕적으로 선택하게 했으며, epsilon은 1.0에서 천천히 내려가게 했습니다.

```
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
```
학습은 메모리버퍼가 1000개 이상일때부터 시작합니다. 

메모리 버퍼에서 배치사이즈만큼 데이터를 뽑아 학습에 이용합니다.

또한 두개의 네트워크를 운영함으로서 너무 급격한 학습이 이루어지지 않게 조절합니다.

# 4. 학습 결과

현재 노트북을 사용하는 관계로 노트북이 버텨주는 200 epsisode까지 학습했습니다.

<img width="1608" height="724" alt="image" src="https://github.com/user-attachments/assets/b5061a26-623d-4251-9c57-9ab978ca3e01" />

뒤로 갈수록 평균적인 최종 보상값이 올라가는걸 확인할 수 있습니다.

# 5. 시행착오

해당 부분은 노션에 일기처럼 적은 부분을 그대로 옮겼음을 알립니다.
---

첫번째 시도에는 모든 에피소드마다 벽으로 돌진함.  epsilion을 1에서 서서히 낮아지게

두번째 시도에는 뭔가 되는거 같은데 나중에 벽으로 돌진함 벽 reward를 -10으로 변경 확인결과

지금 10,000 이상이면 학습을 시도하는데 10,000까지 너무 오래 걸림, 1,000으로 변경

세번째 시도, UnboundLocalError: local variable 'epsilon' referenced before assignment, 이런거 뜸

epsilon을 self.epsilon으로 변경

네번째 시도, 40 에피소드 정도 부터 그냥 벽 바로 앞에서 왔다갔다한다.

알고보니 내가 이상하게 설정을 해놨었다. agent가 반환하는 값을 앞 뒤 밖에 적용을 못시키게 설정을 해서 자꾸 벽으로 돌진했던거다. 돌진말곤 할게 없으니까.

다섯번째 시도.

이제 딱 봐도 정상적으로 작동하는게 보인다. 시간떄문에 여기서 멈춘다.

다음에는 보상을 좀더 촘촘히 하고 loss같은것도 계속 시각화 가능하게하고, save도 가능하게 할란다.

최대속도 5f에서 매 순간 (현재속도 - 2.5f)만큼의 보상을 준다.

텐서보드 적용 

tensorboard --logdir=runs --reload_interval 5

scaler에서 ignore 해제

이렇게 해서 돌렸는데 이놈이 내가 원하는데로 빙글빙글도는게 아니라 빠르게 왔다갔다만 하는 느낌이다.

속도에 대한 보상을 줄이고 checkpoint를 도달했을때 얻는 보상을 늘리자

(rb.velocity.magnitude - 2.5f) / 5f

벽에 부딫히면 -100, 체크포인트는 +50

어느정도 돌아다니기는 함. 학습을 더 시키고 싶지만 노트북이 너무 뜨거워질거 같아 무서움

