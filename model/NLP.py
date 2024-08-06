import pymysql
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.tag import pos_tag
from khaiii import KhaiiiApi

# 데이터베이스 연결
conn = pymysql.connect(host='localhost', user='root', password='', db='demo', charset='utf8')
cur = conn.cursor()

# 데이터베이스에서 문장 가져오기
cur.execute("SELECT content FROM dataset")
rows = cur.fetchall()

# 연결 종료
cur.close()
conn.close()


# 가져온 데이터를 DataFrame으로 변환
df = pd.DataFrame(rows, columns=['text'])

print(df)

labels = [0, 1, 0, 1]  # 실제 데이터를 바탕으로 라벨링

api = KhaiiiApi()

banned_words = []


#text = "돼지 저금통"
#tokenized_sentence = word_tokenize(text)

#print('단어 토큰화1 :', tokenized_sentence)
#print('품사 토큰화 :', pos_tag(tokenized_sentence) )

for word in api.analyze(df):
    print(word)


def filter_banned_words(df, banned_words):
    words = df.split()
    filtered_words = [word for word in words if word.lower() not in banned_words]
    return ' '.join(filtered_words)

clean_text = filter_banned_words(text, banned_words)
print(clean_text)
