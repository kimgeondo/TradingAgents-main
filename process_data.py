import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta

# 1. 뉴스 데이터 로드
input_file = "training_data.csv"
output_file = "final_rl_dataset_v2.csv"

try:
    df_news = pd.read_csv(input_file)
    print(f"📂 뉴스 데이터 로드 성공! ({len(df_news)}개)")
except Exception:
    print("❌ 뉴스 데이터 파일(training_data.csv)이 없습니다.")
    exit()

df_news['Date'] = pd.to_datetime(df_news['Date'])
tickers = df_news['Ticker'].unique()
final_data = []

print("🚀 고급 기술적 지표 계산 및 병합 시작...")

for ticker in tickers:
    print(f"\n📈 {ticker} 지표 분석 중...")
    
    # 데이터 충분히 가져오기 (MACD 계산 등을 위해 100일 전부터)
    start_date = df_news['Date'].min() - timedelta(days=100)
    end_date = datetime.now()
    
    df_price = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # MultiIndex 컬럼 평탄화 (yfinance 버전에 따라 필요)
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)

    # -------------------------------------------------------
    # 🛠️ 기술적 지표 추가 (AI를 위한 3대장)
    # -------------------------------------------------------
    
    # 1. RSI (탄력성): 14일 기준
    df_price.ta.rsi(length=14, append=True)
    
    # 2. MACD (추세): MACD 히스토그램(MACDh)이 추세 전환 파악에 유리함
    # 결과 컬럼: MACD_12_26_9, MACDh_12_26_9(히스토그램), MACDs_12_26_9(신호)
    df_price.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # 3. Bollinger Bands (변동성): %B 지표 사용
    # %B (BBP)는 주가가 밴드 상단에 있으면 1, 하단에 있으면 0 근처가 됨.
    df_price.ta.bbands(length=20, std=2, append=True)

    # 4. 다음 날 수익률 (정답지)
    df_price['Next_Return'] = df_price['Close'].shift(-1).pct_change() * 100
    
    # -------------------------------------------------------
    
    # 뉴스 데이터와 합치기
    ticker_news = df_news[df_news['Ticker'] == ticker].copy()
    
    for idx, row in ticker_news.iterrows():
        date = row['Date']
        
        if date in df_price.index:
            try:
                price_row = df_price.loc[date]
                
                # pandas_ta 컬럼명 찾기 (자동으로 생성된 이름 사용)
                rsi_val = price_row.get('RSI_14')
                macd_hist = price_row.get('MACDh_12_26_9') # 히스토그램
                bb_pct = price_row.get('BBP_5_2.0')       # 볼린저 밴드 %B (%B가 없으면 BBP 확인)
                
                # 가끔 컬럼명이 다를 수 있어 안전장치
                if bb_pct is None: 
                    # 기본 설정인 경우 BBP_20_2.0 일 수 있음
                    bb_pct = price_row.get('BBP_20_2.0')

                # 결측치가 있으면 건너뜀
                if pd.isna(rsi_val) or pd.isna(macd_hist) or pd.isna(bb_pct):
                    continue

                merged_row = {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    # --- AI 상태(State) ---
                    "News_Score": row['News_Score'], # 0~100 (감성)
                    "RSI": round(rsi_val, 2),        # 0~100 (과열)
                    "MACD_Hist": round(macd_hist, 4),# 음수/양수 (추세 힘)
                    "BB_Pct": round(bb_pct, 4),      # 0~1 (상대적 위치)
                    "Close_Price": round(price_row['Close'], 2),
                    # --- 정답(Reward) ---
                    "Next_Day_Return": round(price_row.get('Next_Return', 0), 4)
                }
                final_data.append(merged_row)
            except Exception as e:
                print(f"⚠️ 데이터 처리 중 오류 ({date}): {e}")

# 저장
df_final = pd.DataFrame(final_data)
df_final.to_csv(output_file, index=False)

print("\n" + "="*40)
print(f"🎉 3대 지표 병합 완료! '{output_file}' 생성됨.")
print(df_final.head())
print("="*40)