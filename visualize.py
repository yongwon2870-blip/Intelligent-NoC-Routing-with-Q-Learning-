# 학습 곡선 시각화 전용 모듈
import numpy as np
import matplotlib.pyplot as plt

from agent import train_q_learning

# 학습을 실행하고, 시각화에 필요한 '보상 기록(rewards_all_episodes)'만 받아옴.
_, _, rewards_all_episodes = train_q_learning(episodes=1000)

# 그래프 생성
moving_avg_rewards = []
window = 50

for i in range(len(rewards_all_episodes)):
    if i < window:
        moving_avg_rewards.append(np.mean(rewards_all_episodes[:i+1]))
    else:
        moving_avg_rewards.append(np.mean(rewards_all_episodes[i-window+1:i+1]))

plt.figure(figsize=(10, 5))
plt.plot(rewards_all_episodes, label='Raw Reward', alpha=0.3, color='gray')
plt.plot(moving_avg_rewards, label=f'Moving Average ({window} ep)', color='blue', linewidth=2)

plt.title('Q-Learning Agent Convergence over Episodes (NoC Routing)')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)


plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()