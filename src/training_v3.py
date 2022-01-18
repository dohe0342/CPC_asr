import random
import torch
import logging
import os
import torch.nn.functional as F
from tqdm import tqdm

## Get the same logger from main"
logger = logging.getLogger("cdc")

def trainXXreverse(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, [data, data_r] in enumerate(train_loader):
        data   = data.float().unsqueeze(1).to(device) # add channel dimension
        data_r = data_r.float().unsqueeze(1).to(device) # add channel dimension
        optimizer.zero_grad()
        hidden1 = model.init_hidden1(len(data))
        hidden2 = model.init_hidden2(len(data))
        acc, loss, hidden1, hidden2 = model(data, data_r, hidden1, hidden2)

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))

def train_spk(args, cdc_model, spk_model, device, train_loader, optimizer, epoch, batch_size, frame_window):
    cdc_model.eval() # not training cdc model 
    spk_model.train()
    for batch_idx, [data, target] in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device) # add channel dimension
        target = target.to(device)
        hidden = cdc_model.init_hidden(len(data))
        output, hidden = cdc_model.predict(data, hidden)
        #data = output.contiguous().view((-1,256))
        data = hidden.contiguous().view((-1,256))
        target = target.view((-1,1))
        shuffle_indexing = torch.randperm(data.shape[0]) # shuffle frames 
        data = data[shuffle_indexing,:]
        target = target[shuffle_indexing,:].view((-1,))
        optimizer.zero_grad()
        output = spk_model.forward(data) 
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        acc = 1.*pred.eq(target.view_as(pred)).sum().item()/len(data)
        
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) / frame_window, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))


def train_asr(args, cdc_model, dec, device, train_loader, optimizer, epoch, batch_size, frame_window):
    cdc_model.eval() # not training cdc model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_acc = 0
    all_count = 0
    #criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CTCLoss()
    
    for batch_idx, [data, target] in enumerate(train_loader):
        #print(data.size())
        batch_size = data[0].size()[0]
        data_len = data[1]
        data = data[0]
        
        target_len = target[1]
        target = target[0]
        
        data = data.float().unsqueeze(1).to(device) # add channel dimension
        target = target.to(device)
        hidden = cdc_model.init_hidden(len(data))
        output, hidden = cdc_model.predict_dynamic_seq(data, hidden, data_len)
        all_hidden = torch.Tensor(hidden).to(device)
    
        ctc_input = output
        ctc_output = torch.zeros(batch_size, 2560, 27).to(device)
        
        for i in range(batch_size):
            #print(output[i].size()[1])
            temp = dec(ctc_input[i])
            temp = torch.nn.functional.pad(temp[0], (0,0,0,2560-temp.size()[1]), "constant", 0)
            ctc_output[i] = temp

        ctc_output = ctc_output.transpose(0, 1)
        #print('ctc output size = ', ctc_output.size())
        #print('min target len = ', data_len.min())
        #print('max target len = ', target_len.max())
        data_len = data_len * 128
        #print('data len = ', data_len)
        #print('target len = ', target_len)
        #exit()
        #print('target size = ', target.size())
        #print('data len size = ', data_len.size())
        #print('target len size = ', target_len.size())
        loss = criterion(ctc_output, target, data_len, target_len) 
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        #print('learning rate = ', lr)
        print('loss = ', loss.item())

        '''
        num_classes = 29
        all_dec_h = data
        all_dec_input = torch.zeros([batch_size, num_classes], device=device)
        
        forcing_prob = 0.5
        use_forcing = True if random.random() < forcing_prob else False
        optimizer.zero_grad() 
        output = []
        
        attention = False
        loss = 0 
        
        all_dec_input = all_dec_input.unsqueeze(1)
        all_dec_h = all_dec_h.unsqueeze(0)
        
        decoder_output = None
        
        for i in range(500):
            dec_out, dec_h = dec(all_dec_input, all_dec_h)
            if i == 0:
                decoder_output = dec_out.unsqueeze(0).clone()
            else:
                decoder_output = torch.cat([decoder_output, dec_out.unsqueeze(0)])

            dec_input = dec_out.detach()
        input_len = torch.full(size=(batch_size, ), fill_value=400, dtype=torch.long).to(device)
        '''
    '''
        loss = criterion(decoder_output, target, input_len, target_len)
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        print(loss.item())

    print('----------acc-----------')
    print(f'{all_acc} / {all_count}')
    print('------------------------')
    torch.save(dec.state_dict(), f'./rand_init_{epoch}.pt')
    '''


def train(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device) # add channel dimension
        optimizer.zero_grad()
        hidden = model.init_hidden(len(data), use_gpu=True)
        acc, loss, hidden = model(data, hidden)

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))

def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))
