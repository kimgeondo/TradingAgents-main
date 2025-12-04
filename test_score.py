# test_score.py
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tradingagents.agents.analysts.news_analyst import NewsAnalyst

# 1. 환경변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 에러: OPENAI_API_KEY가 없습니다. .env 파일을 확인해주세요.")
    sys.exit(1)

if not os.getenv("ALPHA_VANTAGE_API_KEY"):
    print("⚠️ 경고: ALPHA_VANTAGE_API_KEY가 없습니다. 뉴스가 안 받아질 수 있습니다.")

# 2. LLM 설정 (비용 절약을 위해 gpt-4o-mini 추천)
print("🚀 테스트 시작...")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. 뉴스 에이전트 생성
agent = NewsAnalyst(llm)

# 4. 애플(AAPL) 주식으로 테스트
result = agent.analyze("AAPL")

# 5. 결과 출력
print("\n" + "="*30)
print(f"📈 종목: AAPL")
print(f"💯 점수: {result['score']}점")
print(f"🚦 신호: {result['signal']}")
print(f"📝 AI 리포트:\n{result['comment']}")
print("="*30)
