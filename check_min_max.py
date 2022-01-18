f = open('output_list.txt', 'r').readlines()

output = list()
for line in f:
    output.append(int(line.split('\n')[0]))

print(min(output))
print(max(output))
