'''
Author: your name
Date: 2021-04-08 23:46:57
LastEditTime: 2021-04-12 14:39:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Homework3\\Test.py
'''

import torch
import torch.nn as nn
from Utils import device
from Model import model
import math
import collections
from Data_Process import test_iter
from Train import criterion

def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def bleu(pred_tokens, label_tokens, k=100):
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

model.load_state_dict(torch.load('Model/en_de_model_loss4.03.pth'))

def test(model,iterator,criterion):
    model.eval()
    epoch_loss = 0
    epoch_bleu = 0
    for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            bleu=bleu(output,trg)
            epoch_loss += loss.item()
            epoch_bleu+=bleu
    return epoch_loss / len(iterator),epoch_bleu / len(iterator)

print(test(model,test_iter,criterion))