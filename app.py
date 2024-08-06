from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import PIL.Image
import pandas as pd
#변경 필요
import config
GEM_API_KEY = config.GEM_API_KEY
# GEM_API_KEY = os.getenv("GEM_API_KEY")


genai.configure(api_key=GEM_API_KEY)

# SBERT 모델 로드
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

datafile_path = "loogle_sbert_embeddings_v1.csv"
# datafile_path = "loogle_sbert_embeddings_v2.csv"


df = pd.read_csv(datafile_path, encoding='utf-8')
df['Embeddings_Title'] = df['Embeddings_Title'].apply(eval)  
df['Embeddings_Keyword'] = df['Embeddings_Keyword'].apply(eval)

def search_clothes(query, df_o, top_n=48, weight_title=1, weight_keyword=2):
    print('check point 0')
    df = df_o.copy()
    query_embedding = model.encode(query)
    print('check point 1')

    try:
        df['Similarity_Title'] = df['Embeddings_Title'].apply(lambda x: util.cos_sim(query_embedding, x))
        print('checkpoint 3')
        df['Similarity_Keyword'] = df['Embeddings_Keyword'].apply(lambda x: util.cos_sim(query_embedding, x))
        print('checkpoint 4')

        # 가중치를 적용하여 평균 유사도 계산
        df['Weighted_Average_Similarity'] = (
            (df['Similarity_Title'] * weight_title) + 
            (df['Similarity_Keyword'] * weight_keyword)
        ) / (weight_title + weight_keyword)

        # float 타입으로 변환
        df['Weighted_Average_Similarity'] = df['Weighted_Average_Similarity'].astype(float)
        
        print('checkpoint 5')
        
        # 유사도 기준으로 상위 N개 상품 반환
        top_results = df.nlargest(top_n, 'Weighted_Average_Similarity')
        print("Top Results : ")
        for index, row in top_results.iterrows():
            print(f"Title: {row['title']}, Similarity: {row['Weighted_Average_Similarity']}")
        
        # 제목과 유사도 값을 반환
        return {
            "titles": top_results['title'].tolist(),
            "similarities": top_results['Weighted_Average_Similarity'].tolist()
        }
    except Exception as e:
        print(f"Error during processing: {e}")
        return None  
    
def generate_fashion_analysis(api_key, image_path):
    # API 키 설정
    genai.configure(api_key=api_key)
    # 이미지 파일 열기
    userImageFile = PIL.Image.open(image_path)
    # 모델 초기화
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    # 콘텐츠 생성 요청
    response = model.generate_content([userImageFile, "너는 패션 분석가로서 100자 이내로 분석해주고 뉘앙스적인 것도 포함되었으면 해. 답변은 개조식의 긴 하나의 문장으로 부탁"])
    # 응답의 텍스트 반환
    return response.text



app = Flask(__name__)


@app.route('/search', methods=['POST'])
def search():
    print("flask 시작 제발!")
    
    try:
        data = request.get_json()
        keyword_name = data['name']
        print("1")
        print(GEM_API_KEY)
        print('key 출력 완료')
        
        results = search_clothes(keyword_name, df, top_n=48, weight_title=1, weight_keyword=2)
        print("2")
        
        # 반환된 결과 출력
        print("Returned Titles:", results["titles"])
        print("Returned Similarities:", results["similarities"])

        # titles와 similarities를 직접 가져오기
        titles = results["titles"]  # 수정된 부분
        similarities = results["similarities"]  # 수정된 부분

        return jsonify({'titles': titles, 'similarities': similarities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/imageSearch', methods=['POST'])
def imageSearch():
    print("Flask 이미지 검색 시작!")
    
    # 이미지 파일이 요청에 포함되어 있는지 확인
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    print('이미지 파일 확인 완료.')
    imageFile = request.files['image']

    # 이미지 파일이 비어있지 않은지 확인
    if imageFile.filename == '':
        return jsonify({"error": "No selected file."}), 400
    
    # 이미지 분석 요청
    try:
        response_text = generate_fashion_analysis(GEM_API_KEY, imageFile.stream)
        print("응답 텍스트:", response_text)
        
        # 결과 검색
        results = search_clothes(response_text, df, top_n=48, weight_title=1, weight_keyword=2)
        print('키워드 검색 진행 중...')
        
        # 결과 추출
        titles = results.get("titles", [])
        similarities = results.get("similarities", [])
        
        print("반환된 제목들:", titles)
        print("반환된 유사도:", similarities)

        return jsonify({'titles': titles, 'similarities': similarities})
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return jsonify({"error": "An error occurred during processing."}), 500
    
    
@app.route('/health')
def health_check():
    return 'flask 성공'
