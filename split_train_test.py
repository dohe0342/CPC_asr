import os
import glob
import random

list_file = open('./LibriSpeech/list/train1.txt', 'r').readlines()

train_num = int(len(list_file)*0.6)
val_num = int(len(list_file)*0.2)
test_num = int(len(list_file)*0.2)

random.shuffle(list_file)

train_list = open('./LibriSpeech/list/spk_train2.txt', 'w')
val_list = open('./LibriSpeech/list/spk_val2.txt', 'w')
test_list = open('./LibriSpeech/list/spk_test2.txt', 'w')

for enum, file_name in enumerate(list_file):
    if enum < train_num:
        train_list.write(f'{file_name}')
    elif enum >= train_num and enum < train_num + val_num:
        val_list.write(f'{file_name}')
    else:
        test_list.write(f'{file_name}')
