import torch
import torch.nn as nn
# Target are to be padded
T = 500      # Input sequence length
C = 20      # Number of classes (including blank)
N = 512      # Batch size
S = 40      # Target sequence length of longest target in batch (padding length)
S_min = 30  # Minimum target length, for demonstration purposes
# Initialize random batch of input vectors, for *size = (T,N,C)
device = 'cuda'
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_().to(device)

print('input size = ', input.size())
# Initialize random batch of targets (0 = blank, 1:C = classes)
#print(input)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long).to(device)
print('target size = ', target.size())
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
#input_lengths = torch.randint(low=200, high=T, size=(N,), dtype=torch.long).to(device)
print('input length size = ', input_lengths.size())
print('input lengths = ', input_lengths)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long).to(device)
#print('target length size = ', target_lengths.size())
#print(target_lengths)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
print('loss = ', loss.item())
loss.backward()
# Target are to be un-padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
# Initialize random batch of targets (0 = blank, 1:C = classes)
target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()
