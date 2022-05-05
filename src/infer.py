# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2022/5/5 9:01 
# Description:  
# --------------------------------------------
import numpy as np
from typing import List
from trainer import predict
from utils import load_params, load_idf
from prerocess import chinese_tokenizer, calculate_tf_idf

def inference(sents: List[str]) -> List[str]:
    LABEL_MAP = {1.0: '正面', 0.0: '负面'}
    model_params = load_params('../data/model/param_step-500_acc-0.646.npy') # 训练完成后，从../data/model/ 选择一个文件
    sortedIdfDic = load_idf()

    labels = []
    for sent in sents:
        tokens = chinese_tokenizer(sent)
        weights = calculate_tf_idf(tokens, sortedIdfDic)
        weights.insert(0, 1.0)
        X = np.array([weights])
        logit = float(predict(X, model_params))
        labels.append(LABEL_MAP[logit])

    return labels


if __name__ == '__main__':
    sent1 = '驾驶室左手边的门窗控制面板有点不人性化，用起来不太顺手。'
    sent2 = '方向轻盈，空间大，操控可以，高速稳！'
    sent3 = '内饰外观都很漂亮，12寸大屏很给力，底盘不错，全景天窗1.5t发动机，省油空调给力'
    sent4 = '噪音有点大，扶手储物空间有点小，其余没有！'
    print(inference([sent1, sent2, sent3, sent4]))


