import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests
from io import StringIO
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 설정
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def get_sp500_tickers():
    """위키백과에서 S&P 500 종목 리스트를 긁어옵니다 (헤더 추가하여 차단 우회)"""
    print("📋 S&P 500 리스트 가져오는 중...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        # [수정된 부분] 봇 차단을 막기 위한 헤더(User-Agent) 추가
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # requests로 먼저 html을 가져옴
        response = requests.get(url, headers=headers)
        
        # StringIO를 사용해 pandas가 읽을 수 있게 변환
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        tickers = df['Symbol'].tolist()
        
        # '.'이 들어간 티커 수정 (BRK.B -> BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f"✅ 총 {len(tickers)}개 종목 확보 완료!")
        return tickers
        
    except Exception as e:
        print(f"⚠️ 리스트 가져오기 실패: {e}")
        # 실패 시 기본값 리턴
        return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "GOOGL", "AMZN"]

def apply_technical_filter(tickers, top_n=20):
    """
    [1차 필터] 파이썬 연산으로 기술적 조건에 맞는 종목만 남깁니다.
    전략: '최근 많이 떨어졌지만(RSI < 40) 거래량은 살아있는 종목'
    """
    print(f"📉 {len(tickers)}개 종목 기술적 분석 중... (시간이 좀 걸립니다)")
    
    # [Tip] 500개를 다 하면 너무 오래 걸릴 수 있으니, 테스트할 땐 100개만 잘라서 하세요.
    # 실전에서는 tickers[:100]을 그냥 tickers로 바꾸면 됩니다.
    tickers = tickers[:100] 
    
    # yfinance 다운로드 (스레드 사용하여 속도 향상)
    data = yf.download(tickers, period="3mo", progress=True, threads=True)
    
    # 데이터 구조 정리 (yfinance 버전에 따라 다를 수 있음)
    if isinstance(data.columns, pd.MultiIndex):
        # 종가(Close)만 추출
        try:
            close_df = data['Close']
        except KeyError:
            # yfinance 최신 버전 대응
            close_df = data.xs('Close', level=0, axis=1)
    else:
        close_df = data[['Close']]

    candidates = []
    
    for ticker in tickers:
        try:
            # 해당 종목의 종가 시리즈
            if ticker not in close_df.columns:
                continue
                
            series = close_df[ticker].dropna()
            
            if len(series) < 14: continue 

            # RSI 계산
            rsi = ta.rsi(series, length=14).iloc[-1]
            
            # 조건: RSI 40 이하 (과매도)
            if rsi < 40:
                candidates.append({
                    "Ticker": ticker,
                    "RSI": round(rsi, 2),
                    "Price": round(series.iloc[-1], 2)
                })
        except Exception:
            continue
            
    # 정렬 및 상위 N개 추출
    candidates_df = pd.DataFrame(candidates)
    if candidates_df.empty:
        print("⚠️ 조건에 맞는 종목이 없습니다. (하락장이 아니라면 RSI<40이 잘 안 나옵니다)")
        return tickers[:top_n] # 없으면 그냥 앞의 N개
        
    candidates_df = candidates_df.sort_values(by='RSI', ascending=True)
    print(f"\n✅ 1차 필터 통과 종목 ({len(candidates_df)}개):")
    print(candidates_df.head())
    
    return candidates_df['Ticker'].head(top_n).tolist()

def get_ai_final_picks(candidates, theme="undervalued tech stocks"):
    """
    [2차 필터] 살아남은 후보들을 AI에게 보여주고 최종 선택
    """
    print(f"\n🧠 AI가 최종 {len(candidates)}개 후보 중에서 '{theme}' 테마로 선별 중...")
    
    candidates_str = ", ".join(candidates)
    
    prompt = f"""
    You are a portfolio manager. 
    Here is a list of candidate stocks that have passed a technical filter (Oversold/Low RSI):
    [{candidates_str}]
    
    From this list, select the Top 5 stocks that best fit the theme: "{theme}".
    Consider their fundamentals and sector potential based on your knowledge.
    
    Output strictly a Python list of strings. Example: ["AAPL", "TSLA"]
    Do not add any explanation.
    """
    
    response = llm.invoke(prompt)
    content = response.content
    
    # 결과 파싱
    import re
    match = re.search(r'\[.*?\]', content)
    if match:
        final_picks = eval(match.group(0))
        return final_picks
    else:
        return candidates[:5]

# --- 실행 함수 ---
def run_hybrid_screening():
    # 1. 유니버스 확보 (S&P 500)
    all_tickers = get_sp500_tickers()
    
    # 2. 기술적 필터
    tech_picks = apply_technical_filter(all_tickers, top_n=20)
    
    # 3. AI 필터
    final_picks = get_ai_final_picks(tech_picks, theme="Technology and Growth stocks with recovery potential")
    
    return final_picks

if __name__ == "__main__":
    picks = run_hybrid_screening()
    print("\n" + "="*30)
    print(f"🏆 최종 선정된 종목: {picks}")
    print("="*30)