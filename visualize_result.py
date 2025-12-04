import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from train_rl import DynamicWeightTradingEnv

# --- 설정 구간 ---
DATA_FILE = "final_rl_dataset_long.csv" # 데이터 가공이 끝난 파일 이름
MODEL_FILE = "my_trading_ai"            # 학습된 모델 이름
TARGET_TICKER = "MSFT"                  # [중요] 그래프로 그려볼 종목 (데이터에 있는 것 중 하나)
# ----------------

# 1. 데이터 로드 및 필터링
try:
    df_all = pd.read_csv(DATA_FILE)
    # 특정 종목만 뽑아내기 (그래프가 예쁘게 나오게)
    df = df_all[df_all['Ticker'] == TARGET_TICKER].copy()
    
    # 날짜순 정렬 (혹시 섞여있을까봐)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"📊 '{TARGET_TICKER}' 종목으로 백테스팅을 시작합니다. (데이터 {len(df)}일)")
except FileNotFoundError:
    print(f"❌ 파일이 없습니다. process_data.py를 실행해서 '{DATA_FILE}'을 먼저 만드세요.")
    exit()

# 2. 모델 로드
try:
    model = PPO.load(MODEL_FILE)
except:
    print(f"❌ 모델 파일('{MODEL_FILE}')이 없습니다. train_rl.py를 먼저 실행하세요.")
    exit()

# 3. 시뮬레이션 환경 준비
env = DynamicWeightTradingEnv(df)
obs, _ = env.reset()

# 4. 타임머신 타고 매매 시작
dates = []
portfolio_values = [10000] # 초기 자본금 $10,000
benchmark_values = [10000] 
weight_history = [] 

print("🚀 AI가 과거 데이터를 복기하며 매매 중...")

for i in range(len(df) - 1):
    # AI의 판단
    action, _ = model.predict(obs)
    weights = np.exp(action) / np.sum(np.exp(action)) # 비중(%)으로 변환
    weight_history.append(weights)
    
    # 환경 진행 (하루 지남)
    obs, reward, done, _, _ = env.step(action)
    
    # --- 자산 가치 계산 (가상 매매) ---
    # reward는 '수익률(%)'을 의미함 (train_rl.py 로직 기반)
    # AI 포트폴리오 업데이트
    current_value = portfolio_values[-1]
    # reward가 +면 수익, -면 손실
    # (train_rl.py에서 reward = position * actual_return 으로 정의됨)
    # 수익률은 퍼센트 단위가 아니라 소수점 단위여야 계산되므로 /100 처리 주의
    # 여기서는 간편하게 reward 자체가 수익률 변화라고 가정하고 복리 계산
    
    # 실제 변동폭 가져오기
    actual_return_pct = df.iloc[i]['Next_Day_Return']
    
    # AI가 매수 포지션을 잡았는지 역산 (reward가 0이 아니면 포지션 잡은 것)
    # 또는 AI 점수 로직 다시 계산
    row = df.iloc[i]
    obs_temp = np.array([row['News_Score']/100, row['RSI']/100, row['MACD_Hist'], row['BB_Pct']])
    
    # 지표별 시그널 (AI 로직)
    sig_news = obs_temp[0]
    sig_rsi = 1.0 - obs_temp[1] if obs_temp[1] > 0.7 else (obs_temp[1] if obs_temp[1] < 0.3 else 0.5)
    sig_macd = 1.0 if obs_temp[2] > 0 else 0.0
    sig_bb = 1.0 if obs_temp[3] < 0.1 else (0.0 if obs_temp[3] > 0.9 else 0.5)
    
    final_score = np.sum(weights * np.array([sig_news, sig_rsi, sig_macd, sig_bb]))
    
    # 포지션: 점수 0.6 이상이면 매수(1), 아니면 현금보유(0)
    position = 1 if final_score > 0.6 else 0
    
    # 자산 업데이트
    if position == 1:
        new_value = current_value * (1 + actual_return_pct/100)
    else:
        new_value = current_value # 현금 보유 (변동 없음)
    
    portfolio_values.append(new_value)
    
    # 벤치마크 (무조건 보유)
    bench_value = benchmark_values[-1] * (1 + actual_return_pct/100)
    benchmark_values.append(bench_value)
    
    dates.append(df.iloc[i]['Date'])

# 5. 결과 그리기 (스타일링 추가)
sns.set_style("whitegrid")
plt.figure(figsize=(14, 10))

# [상단] 수익률 그래프
plt.subplot(2, 1, 1)
plt.plot(dates, portfolio_values, label='AI Trader (Adaptive)', color='blue', linewidth=2)
plt.plot(dates, benchmark_values, label='Buy & Hold (Benchmark)', color='gray', linestyle='--', alpha=0.7)
plt.title(f'AI Trading Performance Analysis ({TARGET_TICKER})', fontsize=16, fontweight='bold')
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.fill_between(dates, portfolio_values, benchmark_values, where=(np.array(portfolio_values) > np.array(benchmark_values)), interpolate=True, color='blue', alpha=0.1)

# [하단] AI 비중 변화 (Stackplot)
plt.subplot(2, 1, 2)
weight_history = np.array(weight_history)
labels = ["News (Fundamental)", "RSI (Momentum)", "MACD (Trend)", "Bollinger (Volatility)"]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

plt.stackplot(range(len(weight_history)), 
              weight_history[:, 0], 
              weight_history[:, 1], 
              weight_history[:, 2], 
              weight_history[:, 3], 
              labels=labels, colors=colors, alpha=0.85)

plt.title('Dynamic Feature Importance (AI Decision Logic)', fontsize=16, fontweight='bold')
plt.ylabel('Weight Allocation (0~1)', fontsize=12)
plt.xlabel('Trading Days', fontsize=12)
plt.legend(loc='lower left', fontsize=10, ncol=4)
plt.margins(0, 0)

plt.tight_layout()
plt.savefig("final_result.png", dpi=300) # 고해상도 저장
plt.show()

print("\n✨ 그래프 생성 완료!")
print(f"👉 'final_result.png' 파일을 열어보세요. (총 수익률: {(portfolio_values[-1]-10000)/100:.2f}%)")