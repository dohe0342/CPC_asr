import os
import glob

transcript = open('./LibriSpeech/list/transcript_train', 'r').readlines()
char_idx = open('./LibriSpeech/list/char_idx.txt', 'r').readlines()

groundtruth = open('./LibriSpeech/list/gt.txt', 'w')

char_dict = {}

for char in char_idx:
    char = char.split('\t')
    char_dict[char[0]] = str(char[1].split('\n')[0])

print(char_dict)
for line in transcript:
    file_name = line.split('\t')[0].split('/')[-1].split('.')[0]
    sentence = line.split('\t')[1].split('\n')[0]

    sentence_idx = []
    for char in sentence:
        sentence_idx.append(char_dict[char])
    
    sentence_idx = ' '.join(sentence_idx) + ' 28'
    #print(f'{file_name}\t{sentence_idx}')
    groundtruth.write(f'{file_name}\t{sentence_idx}\n')
