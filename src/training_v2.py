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
    criterion = torch.nn.NLLLoss()
    
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
        #print(output.size(), hidden.size()) 
        #all_hidden = None
        '''
        for i in range(batch_size):
            if i == 0:
                all_hidden = hidden[i]
            else:
                all_hidden = torch.cat([all_hidden, hidden[i]], dim=0)
        '''
        all_hidden = torch.Tensor(hidden).to(device)
        data = all_hidden.contiguous().view((-1,256))

        if 0:
            shuffle_indexing = torch.randperm(data.shape[0]) # shuffle frames 
            data = data[shuffle_indexing,:]
            target = target[shuffle_indexing,:].view((-1,))
        
        num_classes = 29
        all_dec_h = data
        all_dec_input = torch.zeros([batch_size, num_classes], device=device)
        
        forcing_prob = 0.5
        use_forcing = True if random.random() < forcing_prob else False
        optimizer.zero_grad() 
        output = []
        
        attention = False
        loss = 0 
        
        for i in range(batch_size):
            output.append([])
            all_dec_input[i][0] = 1
            
            if use_forcing:
                for j in range(target_len[i]):
                    if attention:
                        dec_out, dec_h, dec_att = dec(dec_input, dec_h, enc_out_list)
                        dec_out = dec_out.view(batch_size, num_classes)
                    else:
                        dec_out, dec_h = dec(all_dec_input[i].unsqueeze(0), all_dec_h[i].unsqueeze(0).unsqueeze(0))
                    dec_input = F.one_hot(target[i][j], num_classes=num_classes).type(torch.cuda.FloatTensor)
                    if j == 0:
                        loss = criterion(dec_out, target[i][j].unsqueeze(0))
                    else:
                        loss += criterion(dec_out, target[i][j].unsqueeze(0))
                    
                    output[i].append(dec_out)

                    _, top1 = dec_out.max(1)
                    if top1 == 28:
                        break
            else:
                for j in range(target_len[i]):
                    if attention:
                        dec_out, dec_h, dec_att = dec(dec_input, dec_h, enc_out_list)
                        dec_out = dec_out.view(batch_size, num_classes)
                    else:
                        dec_out, dec_h = dec(all_dec_input[i].unsqueeze(0), all_dec_h[i].unsqueeze(0).unsqueeze(0))
                    
                    if j == 0:
                        loss = criterion(dec_out, target[i][j].unsqueeze(0))
                    else:
                        loss += criterion(dec_out, target[i][j].unsqueeze(0))

                    output[i].append(dec_out)
                    loss += F.nll_loss(dec_out, target[i][j].unsqueeze(0))
                    dec_input = dec_out.detach()

                    _, top1 = dec_out.max(1)
                    if top1 == 28:
                        break


        loss /= batch_size
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        
        acc = 0
        count = 0
        for i in range(batch_size):
            for j in range(target_len[i]):
                _, top1 = output[i][j].max(1)
                acc += (top1 == target[i][j]).sum().item()
                count += 1
        print(batch_idx, acc, count)
        all_acc += acc
        all_count += count
        

        '''
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        acc = 1.*pred.eq(target.view_as(pred)).sum().item()/len(data)
        
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) / frame_window, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))
        '''


    print('----------acc-----------')
    print(f'{all_acc} / {all_count}')
    print('------------------------')
    torch.save(dec.state_dict(), f'./rand_init_{epoch}.pt')


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
