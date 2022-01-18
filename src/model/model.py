from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

## PyTorch implementation of CDCK2, CDCK5, CDCK6, speaker classifier models
# CDCK2: base model from the paper 'Representation Learning with Contrastive Predictive Coding'
# CDCK5: CDCK2 with a different decoder
# CDCK6: CDCK2 with a shared encoder and double decoders
# SpkClassifier: a simple NN for speaker classification

class CDCK6(nn.Module):
    ''' CDCK2 with double decoder and a shared encoder '''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK6, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru1 = nn.GRU(512, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk1  = nn.ModuleList([nn.Linear(128, 512) for i in range(timestep)])
        self.gru2 = nn.GRU(512, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk2  = nn.ModuleList([nn.Linear(128, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru1 and gru2
        for layer_p in self.gru1._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru1.__getattr__(p), mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru2._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru2.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden1(self, batch_size): # initialize gru1
        #return torch.zeros(1, batch_size, 128).cuda()
        return torch.zeros(1, batch_size, 128)

    def init_hidden2(self, batch_size): # initialize gru2
        #return torch.zeros(1, batch_size, 128).cuda()
        return torch.zeros(1, batch_size, 128)

    def forward(self, x, x_reverse, hidden1, hidden2):
        batch = x.size()[0]
        nce = 0 # average over timestep and batch and gpus
        t_samples = torch.randint(self.seq_len/160-self.timestep, size=(1,)).long() # randomly pick time stamps. ONLY DO THIS ONCE FOR BOTH GRU.

        # first gru
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512) # z_tk e.g. size 8*512
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512
        output1, hidden1 = self.gru1(forward_seq, hidden1) # output size e.g. 8*100*256
        c_t = output1[:,t_samples,:].view(batch, 128) # c_t e.g. size 8*256
        pred = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk1[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor

        # second gru
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x_reverse)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512) # z_tk e.g. size 8*512
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512
        output2, hidden2 = self.gru2(forward_seq, hidden2) # output size e.g. 8*100*256
        c_t = output2[:,t_samples,:].view(batch, 128) # c_t e.g. size 8*256
        pred = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk2[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor

        nce /= -1.*batch*self.timestep
        nce /= 2. # over two grus
        accuracy = 1.*(correct1.item()+correct2.item())/(batch*2) # accuracy over batch and two grus
        #print(torch.cat((output1, output2), dim=2).shape)

        return accuracy, nce, hidden1, hidden2

    def predict(self, x, x_reverse, hidden1, hidden2):
        batch = x.size()[0]

        # first gru
        # input sequence is N*C*L, e.g. 8*1*20480
        z1 = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z1 = z1.transpose(1,2)
        output1, hidden1 = self.gru1(z1, hidden1) # output size e.g. 8*128*256

        # second gru
        z2 = self.encoder(x_reverse)
        z2 = z2.transpose(1,2)
        output2, hidden2 = self.gru2(z2, hidden2)

        return torch.cat((output1, output2), dim=2) # size (64, seq_len, 256)
        #return torch.cat((z1, z2), dim=2) # size (64, seq_len, 512*2)


class CDCK5(nn.Module):
    ''' CDCK2 with a different decoder '''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK5, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.gru = nn.GRU(512, 40, num_layers=2, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(40, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size):
        #return torch.zeros(2*1, batch_size, 40).cuda()
        return torch.zeros(2*1, batch_size, 40)

    def forward(self, x, hidden):
        batch = x.size()[0]
        t_samples = torch.randint(self.seq_len/160-self.timestep, size=(1,)).long() # randomly pick time stamps
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        nce = 0 # average over timestep and batch
        encode_samples = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512) # z_tk e.g. size 8*512
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512
        output, hidden = self.gru(forward_seq, hidden) # output size e.g. 8*100*40
        c_t = output[:,t_samples,:].view(batch, 40) # c_t e.g. size 8*40
        pred = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            decoder = self.Wk[i]
            pred[i] = decoder(c_t) # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden

    def predict(self, x, hidden):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        output, hidden = self.gru(z, hidden) # output size e.g. 8*128*40

        return output, hidden # return every frame
        #return output[:,-1,:], hidden # only return the last frame per utt


class CDCK2(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK2, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(256, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu: return torch.zeros(1, batch_size, 256).cuda()
        else: return torch.zeros(1, batch_size, 256)

    def forward(self, x, hidden):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.seq_len/160)-self.timestep, size=(1,)).long() # randomly pick time stamps
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        nce = 0 # average over timestep and batch
        encode_samples = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,512) # z_tk e.g. size 8*512
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512
        output, hidden = self.gru(forward_seq, hidden) # output size e.g. 8*100*256
        c_t = output[:,t_samples,:].view(batch, 256) # c_t e.g. size 8*256
        pred = torch.empty((self.timestep,batch,512)).float() # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden

    def predict(self, x, hidden):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        '''
        for i in range(batch):
            for j in range(20):
                part_sum = torch.sum(z[i][1][20*j:20*(j+1)])
                print(z[i][1][20*j:20*(j+1)].size(), part_sum)
                if part_sum.item() == 0.:
                    print(j)
        '''
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        output, hidden = self.gru(z, hidden) # output size e.g. 8*128*256

        return output, hidden # return every frame
        #return output[:,-1,:], hidden # only return the last frame per utt
    
    def predict_dynamic_seq(self, x, hidden, seq_len):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        new_z = []
        for i in range(batch):
            new_z.append(z[i,:,:(seq_len[i].item()+1)*128])
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        output_list = []
        hidden_list = []
        
        hidden = hidden.squeeze().unsqueeze(1)

        for i, z in enumerate(new_z):
            z = z.unsqueeze(0).transpose(1,2)
            #output, hidden = self.gru(z, hidden) # output size e.g. 8*128*256
            output, h = self.gru(z, hidden[i].unsqueeze(0)) # output size e.g. 8*128*256
            output_list.append(output)
            hidden_list.append(h.squeeze().tolist())

        return output_list, hidden_list # return every frame


class SpkClassifier(nn.Module):
    ''' linear classifier '''
    def __init__(self, spk_num):

        super(SpkClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, spk_num)
            #nn.Linear(256, spk_num)
        )

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def forward(self, x):
        x = self.classifier(x)

        return F.log_softmax(x, dim=-1)

class GRU_DEC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU_DEC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = nn.Linear(num_classes, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h): 
        with torch.autograd.set_detect_anomaly(True):
            batch_size = x.size()[0]
            input = self.emb(x).view(batch_size, 1, -1) 
            out, h_out = self.gru(input, h)

            out = self.softmax(self.fc(out.reshape(out.shape[0], -1)))
        return out, h_out

    def init_hidden(self):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


class GRU_DEC_ATTN(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers, dropout_p=0.1, max_length=200):
          super(GRU_DEC_ATTN, self).__init__()
          self.hidden_size = hidden_size
          self.output_size = input_size
          self.dropout_p = dropout_p
          self.max_length = max_length
      
          self.embedding = nn.Linear(self.output_size, self.hidden_size)
          self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
          self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
      
          self.dropout = nn.Dropout(self.dropout_p)
     
          self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
          self.out = nn.Linear(self.hidden_size, self.output_size)
          self.softmax = nn.LogSoftmax(dim=1)
  
      def forward(self, input, hidden, encoder_outputs):
          embedded = self.embedding(input).view(batch_size, -1) 
          embedded = self.dropout(embedded)
  
          attn_weights = F.softmax(
              self.attn(torch.cat((embedded, hidden[0]), dim=1)), dim=1)
          encoder_outputs = encoder_outputs.squeeze(2)
          attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                   encoder_outputs).squeeze(1)
          output = torch.cat((embedded, attn_applied), dim=1)
          output = self.attn_combine(output) # N X 512 -> N X 256
          output = F.relu(output)
          output = output.view(batch_size, 1, -1) 
          output, hidden = self.gru(output, hidden)
          output = self.softmax(self.out(output).view(batch_size, self.output_size))
          return output, hidden, attn_weights
  
      def initHidden(self):
          return torch.zeros(num_layers, batch_size, self.hidden_size, device=device)


class CTC(nn.Module):
      def __init__(self, input_size, output_size):
          super(CTC, self).__init__()
          self.input_size = input_size
          self.output_size = output_size
      
          self.clf = nn.Linear(self.input_size, self.output_size)
          self.bn = nn.BatchNorm1d(self.output_size)
          self.softmax = nn.LogSoftmax(dim=1)
  
      def forward(self, input):
          output = self.softmax(self.clf(input))
          return output
