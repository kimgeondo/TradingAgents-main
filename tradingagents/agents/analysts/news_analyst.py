import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from tradingagents.agents.utils.news_data_tools import get_news

class NewsAnalyst:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a Sentimental Analyst. 
            CRITICAL: Provide a 'Sentiment Score' (0-100).
            Output Format: "SCORE: [Score]" then reasoning.
            """),
            ("human", "Analyze news for {ticker} with data: {news_data}"),
        ])

    def analyze(self, ticker: str, target_date: str = None):
        # 1. 날짜 설정
        if target_date:
            current_dt = datetime.strptime(target_date, "%Y-%m-%d")
        else:
            current_dt = datetime.now()
        
        date_str = current_dt.strftime("%Y-%m-%d")
        start_date_str = (current_dt - timedelta(days=3)).strftime("%Y-%m-%d")
        
        print(f"📰 [News Analyst] '{ticker}' ({date_str})", end=" ")

        # [전략 1] 최신 데이터(최근 7일)는 리얼 API 사용 (검증용)
        if (datetime.now() - current_dt).days < 7:
            try:
                news_data = get_news.invoke({"ticker": ticker, "start_date": start_date_str, "end_date": date_str})
                if news_data and len(news_data) > 10 and "No news" not in news_data:
                    chain = self.prompt | self.llm
                    response = chain.invoke({"ticker": ticker, "news_data": news_data})
                    content = response.content
                    score = 50
                    for line in content.split('\n'):
                        if line.strip().startswith("SCORE:"):
                            score = int(line.split(":")[1].strip())
                            break
                    print(f"✅ {score}점 (Real)")
                    return {"date": date_str, "ticker": ticker, "score": score, "reason": "Real Analysis"}
            except:
                pass

        # [전략 2] 과거 데이터: "시나리오 기반" 학습 데이터 생성
        try:
            next_day = current_dt + timedelta(days=1)
            df = yf.download(ticker, start=start_date_str, end=next_day.strftime("%Y-%m-%d"), progress=False)
            
            if len(df) > 0:
                # 데이터 추출
                if isinstance(df.columns, pd.MultiIndex):
                    close_p = df['Close'][ticker].iloc[-1]
                    open_p = df['Open'][ticker].iloc[-1]
                else:
                    close_p = df['Close'].iloc[-1]
                    open_p = df['Open'].iloc[-1]
                
                daily_return = (close_p - open_p) / open_p
                
                # ★ 핵심: 70% 확률로만 뉴스와 주가를 연동시킴 (Regime Mixing)
                # 30% 확률로는 "뉴스는 잠잠한데 주가가 튀는 상황"을 연출하여 AI가 차트도 보게 만듦
                is_news_driven = np.random.rand() < 0.7 
                
                if is_news_driven:
                    # [상황 A] 뉴스 주도 장세: 주가 변동폭만큼 점수 부여
                    # AI 학습목표: "이럴 땐 뉴스 비중을 높이자"
                    base_score = 50 + (daily_return * 100 * 12) # 민감도 높임
                    note = "[Training] News Driven Market (High Correlation)"
                else:
                    # [상황 B] 수급/차트 주도 장세: 주가는 변해도 뉴스는 중립(50) 유지
                    # AI 학습목표: "뉴스가 별거 없네? 이번 상승은 차트 때문이구나. 차트 비중 높이자"
                    base_score = 50 + np.random.randint(-5, 5) # 거의 변화 없음
                    note = "[Training] Tech Driven Market (Low Correlation)"

                final_score = int(np.clip(base_score + np.random.randint(-5, 5), 10, 90))
                
                print(f"✅ {final_score}점 ({'News-Driven' if is_news_driven else 'Tech-Driven'})")
                
                return {
                    "date": date_str,
                    "ticker": ticker,
                    "score": final_score,
                    "reason": note
                }

        except Exception:
            pass

        print(f"⚠️ 데이터 없음")
        return {"date": date_str, "ticker": ticker, "score": 50, "reason": "No Data"}

def create_news_analyst(llm):
    return NewsAnalyst(llm)