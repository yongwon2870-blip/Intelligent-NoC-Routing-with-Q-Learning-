# 지능형 라우팅 에이전트 학습
import numpy as np
from noc_env import NoCRoutingEnv

def train_q_learning(episodes = 1000): # Q-Learning 에이전트를 학습시키고 Q-Table과 보상 기록을 반환
    env = NoCRoutingEnv() # 환경 및 Q-Table(라우팅 테이블) 초기화
    q_table = np.zeros((env.n_states, env.n_actions)) # 16개의 노드(State)에서 4가지 방향(Action)으로 갈 때의 가치를 저장하는 표 (모두 0으로 시작)

    # 1. 하이퍼파라미터 세팅(강화학습의 학습 성향 결정)
    alpha = 0.1  # 학습률 (Learning Rate): 새로운 정보를 얼마나 빠르게 받아들일지
    gamma = 0.9  # 할인율 (Discount Factor): 미래의 보상을 얼마나 중요하게 생각할지
    epsilon = 1.0  # 탐험률 (Exploration Rate): 처음엔 무작위로 길을 찾도록 100%로 설정
    epsilon_decay = 0.995  # 학습할수록 무작위 탐험을 줄여나가는 비율
    min_epsilon = 0.01

    reward_all_episodes = [] # 시각화를 위한 점수 기록

    # 2. Q-Learning 학습 루프
    for episode in range(episodes):
        state = env.reset()  # 에피소드마다 패킷을 출발지(S)로 원위치
        done = False
        total_reward = 0

        while not done:  # Action 선택 (Epsilon-Greedy)
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(env.n_actions)  # 무작위 탐색(새로운 경로 개척)
            else:
                action = np.argmax(q_table[state])  # 학습된 최적 경로 선택(경험 활용)

            next_state, reward, done = env.step(action)  # 선택한 방향으로 1 hop 이동

            # Q-Table(Routing Table) 업데이트 (Bellman Eqn)
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            # 새로운 가치 = (1-학습률)*기존 가치 + 학습률*(현재 보상 + 할인율*다음 상태의 최대 가치)
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            total_reward += reward
            state = next_state

        reward_all_episodes.append(total_reward)

        # 에피소드가 끝날 때마다 epsilon 감소(점점 더 배운 대로만 움직임)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return env, q_table, reward_all_episodes

def test_agent(env, q_table):
    # 3. 학습된 최적 경로(Optimal Path) 검증 및 시각화
    print('최적 라우팅 경로 탐색 결과')
    state = env.reset()
    env.render()  # 초기 상태 출력
    done = False
    step_count = 0

    action_names = ["상(Up)", "하(Down)", "좌(Left)", "우(Right)"]

    while not done:
        # 학습이 끝났으므로, 무작위 탐험 없이 가장 높은 가치의 방향으로만 이동
        action = np.argmax(q_table[state])
        print(f'\nStep{step_count + 1} : 패킷이 [{action_names[action]} 방향으로 전송]')

        state, reward, done = env.step(action)
        env.render()
        step_count += 1

        # 무한 루프 방지(길을 못 찾았을 경우)
        if step_count > 15:
            print('경로 탐색 실패 : 무한 루프에 걸림')
            break

    if reward == 100:
        print(f'\n데이터 패킷이 트래픽 혼잡 구역을 피해 {step_count} Hop 만에 목적지(G)에 도달')

if __name__ == '__main__':
    print('지능형 라우팅 에이전트 학습 시작')
    env, q_table, _ = train_q_learning(episodes=1000)
    test_agent(env,q_table)

