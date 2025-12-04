import time
import random
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO

# [우리가 만든 모듈들 가져오기]
from advanced_screener import run_hybrid_screening # 하이브리드 종목 선정기

# 1. 설정
INITIAL_BALANCE = 100000.0 # 초기 자본금 $100,000 (약 1.3억원)
MAX_POSITIONS = 5          # 최대 보유 종목 수 (분산 투자)
ALLOCATION_PER_STOCK = 0.2 # 종목당 최대 투자 비중 (20%)

# 2. AI 모델 로드
print("🧠 AI 트레이더 엔진 로딩 중...")
try:
    model = PPO.load("my_trading_ai")
    print("✅ 강화학습 모델(Brain) 로드 완료!")
except:
    print("⚠️ 학습된 모델이 없습니다. 랜덤 엔진으로 대체합니다.")
    class FakeModel:
        def predict(self, obs): return [np.random.rand(4)], None
    model = FakeModel()

# 3. 포트폴리오 상태 객체
class Portfolio:
    def __init__(self, balance):
        self.balance = balance
        self.holdings = {} # {"AAPL": {"qty": 10, "avg_price": 150}}
        self.history = []

    def buy(self, ticker, price, amount):
        qty = int(amount // price)
        if qty > 0:
            cost = qty * price
            self.balance -= cost
            if ticker in self.holdings:
                # 평단가 갱신 로직 생략 (단순화)
                self.holdings[ticker]['qty'] += qty
            else:
                self.holdings[ticker] = {'qty': qty, 'avg_price': price}
            print(f"   🔥 [매수] {ticker} {qty}주 체결 (@${price:.2f}) | 투자금: ${cost:,.2f}")
            return True
        return False

    def sell(self, ticker, price):
        if ticker in self.holdings:
            qty = self.holdings[ticker]['qty']
            revenue = qty * price
            self.balance += revenue
            profit = (price - self.holdings[ticker]['avg_price']) * qty
            del self.holdings[ticker]
            print(f"   ❄️ [매도] {ticker} 전량 처분 (@${price:.2f}) | 실현손익: ${profit:+,.2f}")
            return True
        return False

    def get_total_value(self, current_prices):
        equity = 0
        for ticker, data in self.holdings.items():
            price = current_prices.get(ticker, data['avg_price']) # 현재가 없으면 평단가 계산
            equity += data['qty'] * price
        return self.balance + equity

# --- 가상 데이터 생성기 (실제 API 연결 전 단계) ---
def get_real_time_status(ticker):
    """
    원래는 여기서 yfinance와 NewsAnalyst를 불러와야 하지만,
    빠른 시뮬레이션을 위해 'AI가 분석한 결과값'을 시뮬레이션합니다.
    (실제 API 연결 시 이 부분만 교체하면 됩니다)
    """
    # 30~90 사이의 랜덤 점수지만, 우량주(MSFT 등)는 좀 더 좋게 나오게 설정
    base_score = 60 if ticker in ["MSFT", "NVDA", "AAPL"] else 50
    
    news_score = np.clip(np.random.normal(base_score, 15), 0, 100)
    rsi = np.clip(np.random.normal(50, 15), 20, 80)
    macd_hist = np.random.normal(0, 0.5)
    bb_pct = np.random.uniform(0, 1)
    current_price = np.random.uniform(100, 500) # 가상의 현재가
    
    return {
        "price": current_price,
        "obs": np.array([news_score/100, rsi/100, macd_hist, bb_pct], dtype=np.float32),
        "raw": (news_score, rsi, macd_hist, bb_pct)
    }

# --- 메인 자동매매 루프 ---
def run_auto_hedge_fund():
    my_fund = Portfolio(INITIAL_BALANCE)
    day = 1
    
    print("\n" + "="*50)
    print(f"🏢 AI 자율운용 헤지펀드 시스템 가동")
    print(f"💰 운용 자산: ${INITIAL_BALANCE:,.2f}")
    print("="*50)

    try:
        while True:
            print(f"\n📅 [Day {day}] 장 시작 준비 중...")
            
            # 1. [Morning] 종목 선정 (Screener)
            print("🕵️ AI 스크리너가 유망 종목을 발굴합니다...")
            # 실제로는 오래 걸리므로, 여기서는 매일 실행하는 척만 하고 
            # 실제 스크리닝은 3일에 한 번 하거나, 데모용으로 빠르게 처리
            target_tickers = run_hybrid_screening() 
            print(f"👉 오늘의 관심 종목(Top Pick): {target_tickers}")
            
            # 2. [Day-Time] 트레이딩 세션
            current_prices = {}
            
            for ticker in target_tickers:
                # 상태 분석
                status = get_real_time_status(ticker)
                current_prices[ticker] = status['price']
                
                # AI 예측 (RL Model)
                action, _ = model.predict(status['obs'])
                weights = np.exp(action) / np.sum(np.exp(action)) # Softmax
                
                # 종합 점수 계산 (우리의 전략)
                # 뉴스, RSI, MACD, BB의 가중평균
                raw = status['raw'] # (news, rsi, macd, bb)
                
                # 신호 변환 (단순화된 로직)
                # 뉴스(높을수록 좋음), RSI(낮을수록 좋음:역매매), MACD(양수 좋음), BB(낮을수록 좋음)
                score_components = np.array([
                    raw[0]/100, 
                    1 - (raw[1]/100), 
                    1 if raw[2] > 0 else 0, 
                    1 - raw[3]
                ])
                final_score = np.sum(weights * score_components)
                
                print(f"   🔍 {ticker}: 점수 {final_score:.2f} (뉴스비중 {weights[0]:.2f})")

                # 3. [Execution] 매매 판단 및 자금 관리
                # 매수 조건: 점수 높음 & 아직 안 가지고 있음 & 자금 여유 있음
                if final_score > 0.65 and ticker not in my_fund.holdings:
                    if my_fund.balance > (my_fund.get_total_value(current_prices) * 0.1): # 최소 현금 체크
                        # 예산 배분: 전체 자산의 20% 투자
                        budget = my_fund.get_total_value(current_prices) * ALLOCATION_PER_STOCK
                        my_fund.buy(ticker, status['price'], budget)
                
                # 매도 조건: 점수 낮음 & 가지고 있음
                elif final_score < 0.35 and ticker in my_fund.holdings:
                    my_fund.sell(ticker, status['price'])

            # 4. [Evening] 결산
            total_equity = my_fund.get_total_value(current_prices)
            ror = ((total_equity - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
            
            print(f"\n🌙 [마감] 총 자산: ${total_equity:,.2f} (수익률: {ror:+.2f}%)")
            print(f"💼 보유 포트폴리오: {list(my_fund.holdings.keys())}")
            print("-" * 50)
            
            day += 1
            time.sleep(5) # 5초 뒤 다음 날로 (데모용)

    except KeyboardInterrupt:
        print("\n🛑 펀드 운용 중단.")

if __name__ == "__main__":
    run_auto_hedge_fund()