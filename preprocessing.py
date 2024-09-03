import re
import ast
import pandas as pd

def hashtag_extraction(data):
    # 해시태그 추출 작업 (여기서는 간단히 "#"으로 시작하는 단어를 추출하는 예시)
    return " ".join([word for word in data.split() if word.startswith("#")])

def clean_text(data): 
    # 문자열을 리스트로 변환
    word_list = data.split(" ")
    # 리스트의 항목들을 띄어쓰기로 묶어서 하나의 문장으로 표현
    result = ' '.join(re.sub(r'[^a-zA-Z0-9가-힣\s]', '', word) for word in word_list)
    return result

def latest_data(df, path="./latest_df.csv"):
# 날짜를 datetime 형식으로 변환
    df['vd.video_published'] = pd.to_datetime(df['vd.video_published'])

    # 각 vd.video_id 그룹에서 가장 최근의 데이터를 선택
    df_latest = df.groupby('vd.video_id').apply(lambda x: x.loc[x['vd.video_published'].idxmax()])

    # 그룹화 후 인덱스 정리
    df_latest.reset_index(drop=True, inplace=True)

    # 결과 저장 
    df_latest.to_csv(path, index=False)
def safe_extract_hashtag(description):
    try:
        return hashtag_extraction(description)
    except Exception as e:
        print(f"Error processing: {description}. Error: {e}")
        return description  # 또는 적절한 기본값

def safe_clean_text(data):
    try:
        return clean_text(data)
    except Exception as e:
        return data