# 통신망 환경 만들기
import numpy as np

class NoCRoutingEnv:
#     Network-on-Chip (NoC) 라우팅 최적화를 위한 커스텀 강화학습 환경
#     - 4x4 Grid 네트워크에서 데이터 패킷이 출발지(S)에서 목적지(G)로 이동.
#     - 중간에 트래픽이 마비된 혼잡 구역(Congestion, X)을 피해야 함.

    def __init__(self):
        self.grid_size = 4
        self.n_states = self.grid_size * self.grid_size # 총 16개의 라우터 노드
        self.n_actions = 4 # 0: 상, 1: 하, 2: 좌, 3: 우

        # 네트워크 맵 정의 (S: 출발, G: 도착, X: 혼잡/단절, O: 정상 노드)
        self.desc = np.array([
            ['S', 'O', 'O', 'O'],
            ['O', 'X', 'O', 'X'],
            ['O', 'O', 'X', 'O'],
            ['O', 'O', 'O', 'G']
        ])

        self.current_state = 0 # 시작 위치(0번 노드)

    def reset(self): # 패킷을 출발지(0번 노드)로 초기화
        self.current_state = 0

        return self.current_state

    def step(self, action): # 에이전트(패킷)가 상하좌우로 이동했을 때의 결과를 반환
        row = self.current_state // self.grid_size
        col = self.current_state % self.grid_size

        # 행동(Action)에 따른 다음 위치 계산
        if action == 0 and row > 0:
            row -= 1                                    # up
        elif action == 1 and row < self.grid_size - 1:
            row += 1                                    # down
        elif action == 2 and col > 0:
            col -= 1                                    # left
        elif action == 3 and col < self.grid_size - 1:
            col += 1                                    # right

        next_state = row * self.grid_size + col
        node_type = self.desc[row,col]

        # 보상(reward) 설계
        if node_type == 'G':
            reward = 100                 # 목적지 도착(최고 보상)
            done = True
        elif node_type == 'X':
            reward = -10                 # 혼잡 구역 진입(페널티 부여, 통신 실패)
            done = True
        else:
            reward = -1                  # 1 hop 이동할 때마다 -1(최단 경로 유도)
            done = False

        self.current_state = next_state

        return next_state, reward, done

    def render(self): # 현재 패킷(P)의 위치를 시각적으로 출력
        env_map = self.desc.copy()
        row = self.current_state // self.grid_size
        col = self.current_state % self.grid_size
        env_map[row,col] = 'P'  # 현재 패킷 위치 표시

        print('Current NoC Routing State')
        for r in env_map:
            print(' '.join(r))
        print('==========================')

# 테스트 코드
if __name__ == '__main__':
    env = NoCRoutingEnv()
    print('초기 네트워크 상태 : ')
    env.render()