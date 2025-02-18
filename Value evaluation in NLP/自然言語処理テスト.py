# -*- coding: utf-8 -*-
import pandas as pd
import MeCab
from bs4 import BeautifulSoup
import requests
import re
import time

def is_japanese(str):
    return True if re.search(r'[ぁ-んァ-ン]', str) else False 

def scraping(url):
    #スクレイピングの準備
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    #取得した文字列を整形
    contents = soup.find('div', id="the-content")
    text_input = contents.get_text()
    text_origin = [i for i in text_input.split("\n") if i and is_japanese(i)]
    text_list = []
    for sentence in text_origin:
        if "コメントしてBTCを貰おう" in sentence:
            break 
        text_list.append(sentence)
    
    #評価するテキスト
    text = "".join(text_list)
    return text

def scraping_topic(url):
    url_dict = {}
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    contents = soup.find_all(class_="sok-list")
    # 取得したclass="title"で始まるクラスがリスト形式でtagsに格納されている
    for tag in contents:
        # 一つづつ取り出した〇〇〇クラスの中の"a"タグの情報を取得
        for a in tag.select("a"):
            title = a.string
            if title == "First":
                break
            # aタグの中の　href="〇〇〇"　というようなhref=のあとのURLを取得する
            url = a.get('href')
            url_dict[title] = url
    return url_dict

#MeCab準備
tagger = MeCab.Tagger()

df = pd.read_csv("C:\sample\自然言語処理用データセット.csv", header=None, encoding='utf-8')
df.tail()

df2 = pd.read_csv("C:\sample\自然言語処理用データセット(用言).csv", header=None, encoding='utf-8')
df2.tail()

meishi = {}
yougen = {}
already_searched = []

for data in df.itertuples():
    meishi[data[1]] = data[2]
    
for data in df2.itertuples():
    input = data[2]
    if " " in data[2]:
        input = data[2].split(" ")[0]
    yougen[input] = data[1]
    
# 辞書読み込む
word_dic = d = {**meishi, **yougen}

#MeCab準備
tagger = MeCab.Tagger()

url_dict = scraping_topic("https://coinpost.jp/?cat=313")
for title in url_dict:
    if title in already_searched:
        continue
    text = scraping(url_dict[title])
    already_searched.append(title)
    #print("評価するテキスト：", text)
    time.sleep(10)
    point = 0
    #     MeCab
    s = tagger.parse(text)
    
    #         スペースで分割して品詞側を取得
    for line in s.split("\n"):
    #         EOSだったらループを抜ける
        if line == "EOS": break
    #         さらにカンマで分割
        params = line.split("\t")[1].split(",")
    #         品詞を取得
        hinshi = params[0]
    #         単語の原型を取得
        word = params[6] 
        if not (hinshi in ['名詞', '動詞', '形容詞']): continue
    #             取得した単語の原型が辞書に含まれているか調べて点数をつける
        if word in word_dic:
            negaposi = word_dic[word]
            if negaposi == 'n':
                point -= 1
            elif negaposi == 'p':
                point += 1
            else:
                point += 0
            #print(word, negaposi)
    
    print("score:", point)
    #print("形態素解析結果")
    #print(s)
    #print("---------------------------------------------------------------------------------------------------------")
    time.sleep(10)