# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2022/5/5 8:42 
# Description:  
# --------------------------------------------
import os.path

import numpy as np
from typing import List, Tuple


def stats_time(start, end, step, total_step):
    t = end -start
    return '{:.3f}'.format((t / step * (total_step - step) / 3600))

def load_idf(idf_res_path: str = '../data/preprocess/idf_res') -> List[Tuple[str, float]]:
    sortedIdfDic = []
    with open(idf_res_path, 'r', encoding='utf-8') as reader:
        line = reader.readline()
        while line:
            token, idf = line.strip().split(' ')
            sortedIdfDic.append((token, float(idf)))
            line = reader.readline()
    return sortedIdfDic

def load_params(model_path: str):
    return np.load(model_path)
