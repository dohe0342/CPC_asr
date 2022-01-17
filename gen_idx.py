import os
import glob

list_file = open('LibriSpeech/list/train1.txt', 'r').readlines()

idx_dict = {}

idx = 0
for file_name in list_file:
    spk_id = int(file_name.split('-')[0])
    if spk_id not in idx_dict:
        idx_dict[spk_id] = idx
        idx += 1

for key, value in idx_dict.items():
    print(f'{key} {value}')
