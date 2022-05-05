# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2022/4/29 11:18 
# Description:  
# --------------------------------------------
import re
import jieba
import numpy as np
from utils import load_idf
from typing import List, Tuple
from collections import defaultdict, Counter
def chinese_tokenizer(text: str) -> List[str]:
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9，。！？]')
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]

def get_idf(input_path: str, tokenizer_res_path: str, idf_res_path: str):
    sentNum = 0
    sentCounter = defaultdict(int) # 记录词在多少个sentence中出现过
    writer = open(tokenizer_res_path, 'w', encoding='utf-8')
    with open(input_path, 'r', encoding='utf-8') as reader:
        line = reader.readline()
        while line:
            texts = line.strip().split('\t')
            if len(texts) != 3:
                line = reader.readline()
                continue
            _, label, txt = texts
            tokens = chinese_tokenizer(txt) # 分词
            if len(tokens) <= 10: # 筛选句子长度大于20的句子，短句直接剔除
                line = reader.readline()
                continue

            writer.write(label + ' ' + ' '.join(tokens) + '\n') # 将分词好的句子写入文本保存

            sentNum += 1
            for token in set(tokens):
                sentCounter[token] += 1

            line = reader.readline()
    writer.close()
    print(f'the number of sentence: {sentNum}')
    idfDic = defaultdict(float)
    for token in sentCounter:
        if sentCounter[token] > 10: # 基于token在句子中出现的频次，筛选token
            idfDic[token] = np.log(float(sentNum)/(sentCounter[token]+1))

    print(f'the size of vocabulary: {len(idfDic)}')

    # sorted and dump idf
    sortedIdfDic = sorted(idfDic.items(), key=lambda x: x[1], reverse=True)
    with open(idf_res_path, 'w', encoding='utf-8') as writer:
        for token, idf in sortedIdfDic:
            writer.write(token + ' ' + str(idf) + '\n')

def calculate_tf_idf(tokens: List[str], sortedIdfDic: List[Tuple[str, float]]) -> List[float]:
    counter = Counter(tokens)
    weights = []
    for token, idf in sortedIdfDic:
        tf = float(counter[token]) / sum(counter.values())
        tf_idf = tf * idf
        weights.append(tf_idf)
    return weights


def get_tf_idf(tokenizer_res_path: str, tf_idf_res_path: str, idf_res_path: str):
    sortedIdfDic = load_idf(idf_res_path)
    count = 0
    writer = open(tf_idf_res_path, 'w', encoding='utf-8')
    with open(tokenizer_res_path, 'r', encoding='utf-8') as reader:
        line = reader.readline()
        while line:
            count += 1
            label, *tokens = line.strip().split(' ')
            weights = calculate_tf_idf(tokens, sortedIdfDic)
            writer.write(label + ' ' + ' '.join([str(w) for w in weights]) + '\n')
            line = reader.readline()
            if count % 1000 == 0:
                print(f'preprocess to line: {count}')
    writer.close()

def main():
    input_path = '../data/raw/train.tsv'
    tokenizer_res_path = '../data/preprocess/tokenizer_res'
    idf_dump_path = '../data/preprocess/idf_res'
    tf_idf_res_path = '../data/preprocess/tf_idf_res'

    get_idf(input_path, tokenizer_res_path, idf_dump_path)

    get_tf_idf(tokenizer_res_path, tf_idf_res_path, idf_dump_path)



if __name__ == '__main__':
    main()

