import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 설정
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7) # 일기는 좀 감성적으로(0.7)

def write_diary_entry(date, ticker, decision, weights, outcome, close_price):
    """
    AI가 그날의 매매를 회고하며 일기를 씁니다.
    """
    # 가중치 중 가장 높았던 1등, 2등 요소를 찾음
    labels = ["뉴스(Fundamental)", "RSI(Technical)", "MACD(Trend)", "Bollinger(Vol)"]
    sorted_indices = np.argsort(weights)[::-1] # 내림차순 정렬
    
    top1 = labels[sorted_indices[0]]
    top1_score = weights[sorted_indices[0]] * 100
    top2 = labels[sorted_indices[1]]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an AI Fund Manager named 'Alpha-1'.
        Write a short, professional yet slightly emotional trading diary entry (3-4 sentences).
        
        Context:
        - Date: {date}
        - Stock: {ticker} (Close: ${close_price})
        - Decision: {decision}
        - Top Reason: I relied heavily on {top1} ({top1_score:.1f}%) because the signal was strong.
        - Secondary Reason: {top2} also supported my view.
        - Outcome: Daily Profit {outcome:.2f}%
        
        Write in Korean. Start with "📅 [날짜] 오늘의 매매 일지".
        If profit is positive, be proud. If negative, be reflective but determined.
        """),
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "date": date, 
        "ticker": ticker, 
        "decision": decision,
        "top1": top1, 
        "top1_score": top1_score,
        "top2": top2,
        "outcome": outcome,
        "close_price": close_price
    })
    
    return response.content

# --- 테스트 실행 ---
if __name__ == "__main__":
    print("✍️ AI가 일기를 쓰는 중입니다...")
    
    # 예시 데이터 (나중엔 실제 매매 결과랑 연결하면 됨)
    sample_log = write_diary_entry(
        date="2024-05-20",
        ticker="MSFT",
        decision="STRONG BUY",
        weights=[0.8, 0.1, 0.05, 0.05], # 뉴스를 80% 믿음
        outcome=3.5, # 3.5% 수익
        close_price=420.50
    )
    
    print("\n" + "="*40)
    print(sample_log)
    print("="*40)