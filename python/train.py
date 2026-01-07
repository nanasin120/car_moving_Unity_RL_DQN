from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import BehaviorSpec
import numpy as np
import torch
from collections import deque
import random
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter 
from dqn import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Unity와 Python간 통신 채널
engine_config_channel = EngineConfigurationChannel() 
# 유니티 엔진과 대화하기 위한 환경, 이걸로 관측값, 보상 받고 액션 보냄
# file_name = None 이거 None으로 하면 유니티 에디터와 직접 연결 -> Play버튼으로 실시간 연결
# worker_id 기본 포트는 5004임 실제 연결할때는 5004 + worker_id로 연결 -> 5004 + 0이니 5004와 연결
# side_channels = [engine_config_channel] 학습 데이터 외의 데이터를 주고받는 별도 통로 ex 시간 배속
env = UnityEnvironment(file_name=None, worker_id=0, side_channels=[engine_config_channel])
# 시간 배속
engine_config_channel.set_configuration_parameters(time_scale=5.0)

# 임시용 DQN
agent = DQNAgent(input_layer=31, output_layer=5)

writer = SummaryWriter('runs')

try:
    # 에피소드 새로 시작 -> 시뮬레이션 초기화 -> 초기 상황 보고
    env.reset()

    # 모든 에이전트 그룹의 이름 가져옴, 현재는 Ray, 이걸 list에 넣어서 0번을 뽑을수있게 해줌
    behavior_names = list(env.behavior_specs.keys())
    # 0번을 뽑음 -> 첫번째 에이전트 그룹의 이름 -> 현 프로젝트에선 Ray가 first_behavior_name에 들어오게 됨
    first_behavior_name = behavior_names[0]
    # 0번의 관측치 정보, 액션 정보가 들어옴
    spec = env.behavior_specs[first_behavior_name]

    print(f"연결 성공! Behavior Name: {first_behavior_name}")
    print(f"관측치(Observation) 구조: {spec.observation_specs}")
    print(f"액션(Action) 구조: {spec.action_spec}")

    # 5번의 에피소드
    for episode in range(2000):
        # 에피소드 새로 시작 -> 시뮬레이션 초기화 -> 초기 상황 보고
        env.reset()
        # 초기화 -> 현재 정보 들어옴
        print(f'\n--- 에피소드 {episode} 시작 ---\n')

        # 처음 시작이니 죽은건 없어서 일단 _로 무시함
        decision_steps, _ = env.get_steps(first_behavior_name)
        # obs[0]은 레이캐스트고 obs[1]은 자동차 정보임, 이 둘을 일렬로 합쳐주는거임
        state = np.concatenate((decision_steps.obs[0], decision_steps.obs[1]), axis=1)[0]

        # reward 기록용
        total_reward = 0

        # (관측, 판단, 행동, 보상) 이걸 100번 할거임
        for step in range(1000):
            # decision_steps에는 아직 살아있는 에이전트가 들어옴 -> 벽에 안박은 에이전트
            # terminal_steps에는 종료된 에이전트가 들어옴
            # get_steps는 현재 상태를 가져오는것

            # agent에 state를 주면 행동을 반환
            action = agent.act(state)

            # 임시로 비어있는 행동을 하나 만들고
            actions_data = spec.action_spec.empty_action(1)
            # 비어있는 행동에 agent가 반환한 행동을 넣어서
            actions_data.discrete[0, 0] = action
            # 그 행동을 할 준비를 하고
            env.set_actions(first_behavior_name, actions_data)
            # 한 프레임을 진행시킴으로서 행동을 수행함
            env.step()

            # 행동을 수행한 다음의 정보들이 들어옴 -> 이것들이 next_state임
            decision_steps, terminal_steps = env.get_steps(first_behavior_name)

            # 지금은 하나의 자동차만 하는중임 -> terminal_steps가 하나라도 있다면 다 죽었다는 뜻임
            if len(terminal_steps) > 0:
                next_state = np.concatenate(terminal_steps.obs, axis=1)[0] # 이렇게 하면 [0], [1] 할필요 없이 한방에 됨
                reward = terminal_steps.reward[0]
                done = True
            else:
                next_state = np.concatenate(decision_steps.obs, axis=1)[0] # 이렇게 하면 [0], [1] 할필요 없이 한방에 됨
                reward = decision_steps.reward[0]
                done = False

            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done: 
                writer.add_scalar("reward", total_reward, episode)
                if episode % 100 == 0: agent.save(episode)
                break

finally:
    env.close()
    print('연결 종료')