from datetime import datetime, timedelta
import os
# 종료 날짜 설정
end_date_str = '2024-07-30'
end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
from preprocessing import *

# 6개월 전 시작 날짜 계산
start_date = end_date - timedelta(days=4*30)  # Roughly 4 months back

# 날짜 리스트 초기화
date_list = []

# 시작 날짜부터 종료 날짜까지의 날짜를 차례대로 추가
current_date = start_date
while current_date <= end_date:
    date_list.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)

# 중복 제거 (연월 부분만 필요하므로)

import pymysql
import pandas as pd
from tqdm import tqdm
from datetime import datetime
host = 'dothis2.iptime.org' # RDS 엔드포인트 URL 또는 IP 주소
port = 3366 # RDS 데이터베이스 포트 
user = 'root'# MySQL 계정 아이디
password = 'data123!' # MySQL 계정 비밀번호

# RDS MySQL 연결하기
conn = pymysql.connect(host=host, 
                    port=port, 
                    user=user, 
                    password=password)

# 데이터베이스 커서(Cursor) 객체 생성
cursor = conn.cursor()

# column_list = ["vd.video_id", "vd.channel_id", "vd.video_title", "vd.video_tags", "vd.video_description", "vd.video_cluster", "vh.video_views", "vh.video_performance", "vd.video_published"]
column_list = ["ch.channel_id", "ch.channel_subscribers", "ch.channel_average_views"]
# video_id_list = []

# 파일 열기 및 읽기
# with open("/home/bailey/workspace/channel_ids.txt", 'r') as file:
#     # 파일의 각 줄 읽기
#     for line in file:
#         # 줄의 끝에 있는 개행 문자 제거 및 쉼표로 분리
#         ids = line.strip().split(',')
#         # id를 리스트에 추가
#         video_id_list.extend(ids)
channel_id = pd.read_csv("/home/bailey/workspace/channel_id.csv")
channel_id = channel_id['channel_id']
print(channel_id)
idx = 0 
for date in date_list:
    # if os.path.isfile(f"./test/df_{date}.csv"):
    #     continue
    df = pd.DataFrame()
    yearmonth = date.split("-")[0] + date.split("-")[1]
    for cluster in tqdm(range(100)):
        if cluster < 100:
            cluster = str(cluster).zfill(2)
        # sql_query = f"""
        #         # SELECT vd.video_id, vd.channel_id, vd.video_title, vd.video_tags, vd.video_description, vd.video_cluster, vh.video_views, vh.video_performance, vd.video_published
        #         SELECT ch.channel_id, ch.channel_subscribers, ch.channel_average_views
        #         # FROM dothis_svc.video_data_{cluster} AS vd
        #         FROM dothis_svc.channel_history_{yearmonth} AS ch
        #         # JOIN dothis_svc.video_history_{cluster}_{yearmonth} AS vh ON vh.video_id = vd.video_id
        #         JOIN dothis_svc.channel_history_{yearmonth} AS ch ON ch.channel_id = vd.channel_id
        #         # WHERE vd.video_published = '{date}' AND vh.video_views > 1000
        #         WHERE ch.channel_id in ({', '.join([f"'{id}'" for id in channel_id])})
        #         ;"""
        # 수정된 SQL 쿼리
        sql_query = f"""
            SELECT ch.channel_id, ch.channel_subscribers, ch.channel_average_views
            FROM dothis_svc.channel_history_{yearmonth} AS ch
            WHERE ch.channel_id in ({', '.join([f"'{id}'" for id in channel_id])})
            ;
        """
        cursor.execute(sql_query)
        for row in (cursor):
            for c, col in enumerate(column_list):
                df.loc[idx, col] = row[c]
            idx += 1
    # if not df.empty:
    #     df['vd.hashtag'] = df['vd.video_description'].apply(lambda x: safe_extract_hashtag(x))
    #     df['vd.hashtag'] = df['vd.hashtag'].apply(lambda x: safe_clean_text(x))
    #     df['vd.video_tags'] = df['vd.video_tags'].apply(lambda x: safe_clean_text(x))
    #     df.drop('vd.video_description', axis=1, inplace=True)
    #     latest_data(df, path=f"./test/df_{date}.csv")

   
