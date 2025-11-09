from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from datetime import datetime
from konlpy.tag import Okt
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. FastAPI 앱 초기화 및 설정 ---
app = FastAPI()

os.makedirs("static", exist_ok=True) # Ensure static directory exists BEFORE mounting
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
os.makedirs("static/images", exist_ok=True)


# --- 2. 핵심 로직 ---
from fastapi.responses import JSONResponse

def generate_graph_data(text: str) -> dict:
    """
    텍스트를 분석하여 감정 정보가 포함된 노드와 링크로 구성된 그래프 데이터를 생성합니다.
    """
    print(f"그래프 데이터 생성 시작: {text[:50]}...")

    # 감정 사전 정의
    positive_words = {'성공', '기쁨', '행복', '사랑', '감사', '즐거움', '재미', '희망', '최고', '좋아'}
    negative_words = {'실패', '슬픔', '우울', '짜증', '스트레스', '문제', '걱정', '불안', '나쁜', '싫어'}

    okt = Okt()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if len(sentences) < 1:
        return None

    G = nx.DiGraph()
    word_counts = Counter()
    
    all_sentence_nouns = []
    for sentence in sentences:
        # 명사와 형용사를 함께 추출하여 감정 분석의 정확도를 높입니다.
        morphs = okt.pos(sentence)
        words = [m[0] for m in morphs if m[1] in ('Noun', 'Adjective') and len(m[0]) > 1]
        
        if words:
            all_sentence_nouns.append(words)
            word_counts.update(words)

    if not all_sentence_nouns:
        return None

    # 문장 내, 그리고 문장 간의 흐름을 연결 (순차적 연결)
    for i, words in enumerate(all_sentence_nouns):
        for j in range(len(words) - 1):
            G.add_edge(words[j], words[j+1])
        
        if i > 0 and all_sentence_nouns[i-1]:
            prev_last_word = all_sentence_nouns[i-1][-1]
            current_first_word = words[0]
            if prev_last_word != current_first_word:
                G.add_edge(prev_last_word, current_first_word)

    # 동시 등장 관계 추가 (비순차적 연결)
    co_occurrence = Counter()
    for words_in_sentence in all_sentence_nouns:
        # 단어 쌍 생성 (중복 없이)
        unique_words_in_sentence = list(set(words_in_sentence))
        for i in range(len(unique_words_in_sentence)):
            for j in range(i + 1, len(unique_words_in_sentence)):
                word1, word2 = sorted((unique_words_in_sentence[i], unique_words_in_sentence[j]))
                co_occurrence[(word1, word2)] += 1
    
    # 일정 횟수 이상 동시 등장한 단어들 연결
    co_occurrence_threshold = 2 # 최소 2번 이상 함께 등장해야 연결
    for (word1, word2), count in co_occurrence.items():
        if count >= co_occurrence_threshold:
            # 양방향으로 연결하여 관계를 표현 (또는 단방향으로 추가)
            G.add_edge(word1, word2)
            G.add_edge(word2, word1) # 양방향으로 추가하여 더 풍성한 연결


    if G.number_of_nodes() == 0:
        return None

    # JSON 형식으로 데이터 변환 및 감정/연결성 정보 추가
    nodes = []
    degrees = dict(G.degree())
    for node in G.nodes():
        emotion = "neutral"
        if node in positive_words:
            emotion = "positive"
        elif node in negative_words:
            emotion = "negative"
        
        nodes.append({
            "id": node, 
            "size": word_counts.get(node, 1), 
            "emotion": emotion,
            "degree": degrees.get(node, 0)
        })

    links = [{"source": u, "target": v} for u, v in G.edges()]
    
    graph_data = {"nodes": nodes, "links": links}
    
    print("감정 및 연결성 정보 포함 그래프 데이터 생성 완료")
    return graph_data


# --- 3. 웹페이지 라우트(경로) 설정 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 페이지를 보여줍니다."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=JSONResponse)
async def analyze(request: Request, text: str = Form(...)):
    """
    사용자가 입력한 텍스트를 받아 분석을 수행하고,
    그래프 데이터를 JSON으로 반환합니다.
    """
    print("'/analyze' 경로로 POST 요청 받음")
    if not text.strip():
        return JSONResponse(content={"error": "내용을 입력해주세요."}, status_code=400)

    graph_data = generate_graph_data(text)
    
    if graph_data is None:
        return JSONResponse(content={"error": "분석할 내용이 부족하여 지도를 생성할 수 없습니다."}, status_code=400)

    return JSONResponse(content=graph_data)

# --- 4. (선택) 서버 실행 코드 ---
# 터미널에서 'uvicorn main:app --reload' 명령어로 실행하는 것을 권장합니다.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
