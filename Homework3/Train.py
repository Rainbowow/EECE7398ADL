'''
Author: your name
Date: 2021-04-12 14:18:29
LastEditTime: 2021-04-12 14:27:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Homework3\Train.py
'''

from Utils import device
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from Data_Process import en_vocab
PAD_IDX = en_vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


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


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 2000
CLIP = 1
PATH='./NMT_Model.pth'
best_valid_loss = float('inf')


def train_iters(model: nn.Module,
          train_iter: torch.utils.data.DataLoader,
          valid_iter: torch.utils.data.DataLoader,
          test_iter: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):
    for epoch in range(N_EPOCHS):
        
        start_time = time.time()

        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iter, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss<best_valid_loss:
            torch.save(model.state_dict(),PATH)
            best_valid_loss=valid_loss
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s \tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} \t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = evaluate(model, test_iter, criterion)