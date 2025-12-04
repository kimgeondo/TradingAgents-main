import pandas as pd
import os
from stable_baselines3 import PPO
from train_rl import DynamicWeightTradingEnv
from process_data import process_indicators # (데이터 가공 로직 함수화 필요, 아래 설명 참고)

# 설정
DATA_FILE = "final_rl_dataset_long.csv"
MODEL_FILE = "my_trading_ai"
PERFORMANCE_THRESHOLD = -2.0 # 최근 5일 수익률이 -2%보다 나쁘면 재학습

def evaluate_performance():
    """
    최근 매매 기록을 분석해서 성적표를 냅니다.
    (여기서는 시뮬레이션을 위해 최근 데이터 30일치로 백테스트 수행)
    """
    print("🕵️ 감독관: 최근 성적을 감사하는 중...")
    
    try:
        df = pd.read_csv(DATA_FILE)
        model = PPO.load(MODEL_FILE)
        
        # 최근 30일 데이터만 잘라서 테스트
        recent_df = df.tail(30).reset_index(drop=True)
        env = DynamicWeightTradingEnv(recent_df)
        obs, _ = env.reset()
        
        total_reward = 0
        for _ in range(len(recent_df)-1):
            action, _ = model.predict(obs)
            obs, reward, _, _, _ = env.step(action)
            total_reward += reward # 여기서는 reward가 수익률과 비례
            
        print(f"📊 최근 30일 누적 성과 점수: {total_reward:.2f}")
        return total_reward
        
    except Exception as e:
        print(f"⚠️ 평가 실패: {e}")
        return 0

def retrain_model():
    """
    모델을 '더 빡세게' 재학습 시킵니다.
    """
    print("\n⚠️ [경고] 성적 부진 확인! AI를 재교육합니다...")
    
    # 1. 데이터 로드
    df = pd.read_csv(DATA_FILE)
    env = DynamicWeightTradingEnv(df)
    
    # 2. 기존 모델 불러오기 (지식 계승)
    model = PPO.load(MODEL_FILE, env=env)
    
    # 3. 추가 학습 (Fine-tuning)
    # 기존 지식 위에 5,000번 더 연습시킴
    print("🏋️ 훈련소 입소: 5,000번 추가 학습 중...")
    model.learn(total_timesteps=5000)
    
    # 4. 저장
    model.save(MODEL_FILE)
    print("✨ 재학습 완료! AI가 한 단계 진화했습니다.")

def run_evolution_cycle():
    print("🔄 [System] 자기 주도 학습 사이클 시작")
    
    # 1. 평가
    score = evaluate_performance()
    
    # 2. 판단 및 조치
    if score < PERFORMANCE_THRESHOLD:
        print(f"❌ 기준 미달! (점수 {score:.2f} < 기준 {PERFORMANCE_THRESHOLD})")
        retrain_model()
    else:
        print(f"✅ 성적 양호. (점수 {score:.2f} >= 기준 {PERFORMANCE_THRESHOLD})")
        print("💤 현재 모델을 유지합니다.")

if __name__ == "__main__":
    # 주기적으로 이 파일을 실행하면 됩니다 (예: 매주 금요일 밤)
    run_evolution_cycle()