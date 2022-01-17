import glob
import os

log_file = open('./vanilla_GRU.log', 'r').readlines()

all_corr_count = 0
all_char_count = 0
for log in log_file:
    batch_idx, corr_num, char_num = int(log.split(' ')[0]), \
                                    int(log.split(' ')[1]), \
                                    int(log.split(' ')[2].split('\n')[0])

    all_corr_count += corr_num
    all_char_count += char_num


print(all_corr_count, all_char_count)
print(all_corr_count / all_char_count)
