'''
Author: your name
Date: 2021-03-26 16:19:11
LastEditTime: 2021-04-12 14:20:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Homework3\\Utils.py
'''
import math
import collections
import torch
from Hyper_parameters import *


def bleu(pred_tokens, label_tokens, k):
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

