from opensearchpy import OpenSearch
import os
import pandas as pd
import warnings
from datetime import datetime, timedelta

# OpenSearch 클러스터에 연결
OPENSEARCH_HOST = "dothis2.iptime.org"
OPENSEARCH_PORT = 9211
OPENSEARCH_USER = 'admin'
OPENSEARCH_PW = 'data123!'
host = OPENSEARCH_HOST
port = OPENSEARCH_PORT
user = OPENSEARCH_USER
password = OPENSEARCH_PW

# OpenSearch 클라이언트 설정
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=(user, password),
    use_ssl=True,
    verify_certs=False,  # SSL 인증서 검증 비활성화
    time_out=360
)

# 클러스터 상태 조회
def check_cluster_health():
    try:
        health = client.cluster.health()
        print("Cluster health:", health)
    except Exception as e:
        print("Error connecting to OpenSearch:", e)

# 클러스터 상태 확인
check_cluster_health()

# 날짜 범위 설정
end_date = datetime.now() - timedelta(days=30)  # 1달 전
start_date = end_date - timedelta(days=180)     # 6개월 전

# 날짜를 문자열로 변환 (YYYY-MM-DD 형식)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

size = 100000
scroll_duration = '2m'

# OpenSearch 쿼리 설정
query = {
    "size": size,  # 최대 100000개의 결과 반환
    "_source": ["video_id", "channel_id", "video_title", "video_tags", "video_description", 
                "video_cluster", "video_views", "video_performance", "video_published", 
                "channel_subscribers", "channel_average_views", "video_tags"],  # 반환할 필드 지정
    "query": {
        "bool": {
            "filter": [
                {
                    "range": {
                        "video_published": {
                            "gte": start_date_str,  # 6개월 전 시작 날짜
                            "lte": end_date_str    # 1달 전 종료 날짜
                        }
                    }
                },
                {
                    "range": {
                        "video_views": {
                            "gte": 1000  # 조회수 1000 이상
                        }
                    }
                }
            ]
        }
    }
}

# 초기 검색 요청 및 스크롤 시작
response = client.search(
    index="video_history", #video-history에 다 떄려넣었다고 함.
    body=query,
    scroll=scroll_duration  # 스크롤 유지 시간 설정
)

scroll_id = response['_scroll_id']
hits = response['hits']['hits']

# 검색된 문서를 데이터프레임에 저장
df = pd.DataFrame([hit['_source'] for hit in hits])

while True:
    # 스크롤 요청을 사용하여 다음 데이터 셋 가져오기
    response = client.scroll(
        scroll_id=scroll_id,
        scroll=scroll_duration  # 스크롤 유지 시간 설정
    )

    hits = response['hits']['hits']

    # 더 이상 결과가 없으면 루프 종료
    if not hits:
        break

    # 검색된 문서를 데이터프레임에 추가
    etc = pd.DataFrame([hit['_source'] for hit in hits])
    df = pd.concat([df, etc], axis=0)

    # 스크롤 ID 갱신
    scroll_id = response['_scroll_id']

# 최종 데이터프레임을 CSV 파일로 저장
df.to_csv('data.csv', index=False, encoding='utf-8-sig')  # UTF-8 인코딩 설정 (한글 포함 시)
print("데이터가 data.csv 파일로 저장되었습니다.")
