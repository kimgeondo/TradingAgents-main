import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. 강화학습 환경 정의 (AI가 뛰어놀 세상)
class DynamicWeightTradingEnv(gym.Env):
    def __init__(self, df):
        super(DynamicWeightTradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        
        # [행동 정의] AI가 할 수 있는 일: 4가지 지표에 대한 "가중치(Weight)" 정하기
        # 0: 뉴스, 1: RSI, 2: MACD, 3: 볼린저밴드
        # 결과값: 0~1 사이의 실수 4개
        self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # [관찰 정의] AI가 보는 것: 정규화된 지표 값들
        # [News(0~1), RSI(0~1), MACD(-1~1), BB(0~1)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self._next_observation(), {}

    def _next_observation(self):
        # 현재 날짜의 데이터 가져오기
        row = self.df.iloc[self.current_step]
        
        # 데이터 정규화 (AI가 이해하기 쉽게 0~1 사이로 변환)
        obs = np.array([
            row['News_Score'] / 100.0,  # 0~100 -> 0.0~1.0
            row['RSI'] / 100.0,         # 0~100 -> 0.0~1.0
            row['MACD_Hist'],           # 그대로 사용
            row['BB_Pct']               # 0~1 (가끔 벗어나지만 괜찮음)
        ], dtype=np.float32)
        return obs

    def step(self, action):
        # 1. AI가 정한 비중(Action) 가져오기
        # softmax를 써서 비중의 합이 1이 되도록 만듦 (예: [0.1, 0.4, 0.3, 0.2])
        weights = np.exp(action) / np.sum(np.exp(action))
        
        # 2. 현재 시장 상황 관찰
        obs = self._next_observation()
        
        # 3. 종합 점수 계산 (비중 x 지표값)
        # 뉴스와 RSI는 높을수록 매수 관점, MACD도 높을수록 상승, BB는 낮을수록(하단반등) 매수
        # (간단한 로직 예시: 가중 평균 점수가 0.5 넘으면 매수)
        
        # 지표별 매수 시그널 점수화 (0~1)
        signal_news = obs[0] 
        signal_rsi = 1.0 - obs[1] if obs[1] > 0.7 else (obs[1] if obs[1] < 0.3 else 0.5) # 역추세 전략 예시
        signal_macd = 1.0 if obs[2] > 0 else 0.0
        signal_bb = 1.0 if obs[3] < 0.1 else (0.0 if obs[3] > 0.9 else 0.5)
        
        signals = np.array([signal_news, signal_rsi, signal_macd, signal_bb])
        
        # ★ 핵심: AI가 정한 비중대로 종합 점수 산출
        final_score = np.sum(weights * signals)
        
        # 4. 포지션 결정 (종합 점수가 0.6 이상이면 매수)
        position = 1 if final_score > 0.6 else 0 
        
        # 5. 보상 계산 (수익률)
        # 내일 오르는데 샀으면(+), 내일 내리는데 샀으면(-)
        actual_return = self.df.iloc[self.current_step]['Next_Day_Return']
        reward = position * actual_return
        
        # 6. 다음 스텝으로 이동
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 로그 출력 (학습되는 거 보려고 100일마다 한 번씩)
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step}: AI Weights = {np.round(weights, 2)} -> Reward: {reward:.2f}%")

        return obs, reward, done, False, {}

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # 1. 데이터 로드
    try:
        df = pd.read_csv("final_rl_dataset_v2.csv")
        print(f"📂 데이터 로드 완료: {len(df)}건")
    except:
        print("❌ 데이터 파일이 없습니다. process_data.py를 먼저 실행하세요.")
        exit()

    # 2. 환경 만들기
    env = DummyVecEnv([lambda: DynamicWeightTradingEnv(df)])

    # 3. AI 모델 생성 (PPO 알고리즘)
    print("🧠 AI 모델 생성 중...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)

    # 4. 학습 시작! (과거 데이터를 보며 수천 번 연습)
    print("🚀 학습 시작! (잠시만 기다리세요...)")
    model.learn(total_timesteps=10000) # 10,000번 반복 학습

    # 5. 모델 저장
    model.save("my_trading_ai")
    print("🎉 학습 완료! 'my_trading_ai.zip' 파일로 저장되었습니다.")

    # --- 테스트: 학습된 AI가 실제로 어떻게 판단하는지 보기 ---
    print("\n[AI의 판단 테스트]")
    obs = env.reset()
    for i in range(5): # 5일치만 보여줘
        action, _ = model.predict(obs)
        weights = np.exp(action) / np.sum(np.exp(action)) # 비중으로 변환
        print(f"📅 Day {i+1}: 뉴스비중({weights[0][0]:.2f}) vs 차트비중({weights[0][1]:.2f})")
        # 다음 날로 이동
        obs, rewards, dones, info = env.step(action)