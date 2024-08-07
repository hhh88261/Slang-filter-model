import pymysql
import requests
from bs4 import BeautifulSoup

BASE_URL = ("https://gall.dcinside.com/board/lists/?id=leagueoflegends6&page=7")


#demo DB 연동
conn = pymysql.connect(host='localhost', user='root', password='', db='demo', charset='utf8')

# 파라미터 설정
params = {
            'id': 'leagueoflegends6',}

# 헤더 설정
headers = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'}


resp = requests.get(BASE_URL, params=params, headers=headers)
soup = BeautifulSoup(resp.content, 'html.parser')
contents = soup.find('tbody').find_all('tr')

# 커서 생성
db = conn.cursor()

# 크롤링한 데이터 삽입
for i in contents:
    title_tag = i.find('a')
    if title_tag:
        title = title_tag.text.strip()
        print("제목: ", title)

        # 데이터 삽입 쿼리
        insert_query = "INSERT INTO dataSet (content) VALUES (%s)"
        db.execute(insert_query, (title,))

# 변경사항 저장
conn.commit()

# 연결 종료
db.close()
conn.close()
