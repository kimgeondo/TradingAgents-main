from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import re

# 설정
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7) # 창의성을 위해 온도 0.7

def get_ai_picked_tickers(theme="top performing tech stocks in 2024"):
    print(f"🧠 AI가 '{theme}' 테마에 맞는 종목을 고르는 중...")
    
    prompt = f"""
    You are a professional fund manager.
    Please recommend 5 stock ticker symbols related to the theme: "{theme}".
    
    CRITICAL RULE:
    - Only return the Ticker Symbols in a standard Python list format.
    - Do not say anything else.
    - Example output: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    """
    
    response = llm.invoke(prompt)
    content = response.content
    
    # 텍스트에서 리스트 모양만 추출 ["A", "B"]
    try:
        # 정규표현식으로 ["..."] 부분만 찾기
        match = re.search(r'\[.*?\]', content)
        if match:
            tickers_str = match.group(0)
            # 문자열을 실제 리스트로 변환
            tickers = eval(tickers_str)
            return tickers
        else:
            print("⚠️ AI 응답 해석 실패. 기본값 사용.")
            return ["AAPL", "MSFT"] # 실패 시 기본값
    except Exception as e:
        print(f"❌ 에러: {e}")
        return ["AAPL", "MSFT"]

# 테스트 실행
if __name__ == "__main__":
    # 원하는 테마를 입력해보세요
    my_theme = "High volatility AI and Semiconductor stocks"
    picks = get_ai_picked_tickers(my_theme)
    
    print("\n" + "="*30)
    print(f"🎯 테마: {my_theme}")
    print(f"🤖 AI의 선택: {picks}")
    print("="*30)