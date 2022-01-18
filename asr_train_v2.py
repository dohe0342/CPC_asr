## Utilities
from __future__ import print_function
import argparse
import random
import time
import os
import logging
from timeit import default_timer as timer

## Libraries
import numpy as np

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

## Custrom Imports
from src.logger_v1 import setup_logs
from src.data_reader.dataset import RawDatasetSpkClass, RawDatasetAsrClass
from src.training_v3 import train_spk, train_asr, snapshot
from src.validation_v1 import validation_spk
from src.prediction_v1 import prediction_spk
from src.model.model import CDCK2, SpkClassifier, GRU_DEC, GRU_DEC_ATTN, CTC
############ Control Center and Hyperparameter ###############
run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
  
random_seed(777)


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128 
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""
        '''
        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
        '''
        new_lr = 0.00001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

def main():
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--raw-hdf5', required=True)
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--validation-list', required=True)
    parser.add_argument('--eval-list')
    parser.add_argument('--index-file')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--model-path')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8, 
                        help='batch size')
    parser.add_argument('--audio-window', type=int, default=20480, 
                        help='window length to sample from each utterance')
    parser.add_argument('--frame-window', type=int, default=1)
    parser.add_argument('--spk-num', type=int, default=251)
    parser.add_argument('--timestep', type=int, default=12) 
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer() # global timer
    logger = setup_logs(args.logging_dir, run_name) # setup logs
    device = torch.device("cuda" if use_cuda else "cpu")
    cdc_model = CDCK2(args.timestep, args.batch_size, args.audio_window).to(device)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage) # load everything onto CPU
    cdc_model.load_state_dict(checkpoint['state_dict'])
    for param in cdc_model.parameters():
        param.requires_grad = False
    
    #spk_model = SpkClassifier(args.spk_num).to(device)
    '''
    attention = False
    if attention:
        asr_model = GRU_DEC_ATTN(29, 256, 1).to(device)
    else:
        asr_model = GRU_DEC(32, 256, 1, 29).to(device)
    '''
    asr_model = CTC(256, 27).to(device)
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}

    logger.info('===> loading train, validation and eval dataset')
    training_set   = RawDatasetAsrClass(args.raw_hdf5, args.train_list, args.audio_window, args.frame_window)
    #validation_set = RawDatasetSpkClass(args.raw_hdf5, args.validation_list, args.index_file, args.audio_window, args.frame_window)
    #eval_set = RawDatasetSpkClass(args.raw_hdf5, args.eval_list, args.index_file, args.audio_window, args.frame_window)

    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **params) # set shuffle to True
    #validation_loader = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, **params) # set shuffle to False
    #eval_loader = data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, **params) # set shuffle to False
    # nanxin optimizer  

    print('load done')
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, asr_model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)
    #optimizer = optim.Adam(asr_model.parameters())

    model_params = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(asr_model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    ## Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    
    label_len_all = None
    for epoch in tqdm(range(1, 2)):#args.epochs + 1)):
        epoch_timer = timer()

        # Train and validate
        #train_spk(args, cdc_model, spk_model, device, train_loader, optimizer, epoch, args.batch_size, args.frame_window)
        train_asr(args, cdc_model, asr_model, device, train_loader, optimizer, epoch, args.batch_size, args.frame_window)
        #val_acc, val_loss = validation_spk(args, cdc_model, spk_model, device, validation_loader, args.batch_size, args.frame_window)
        
        # Save
    '''
        if val_acc > best_acc: 
            best_acc = max(val_acc, best_acc)
        #if val_loss < best_loss:
            #best_loss = min(val_loss, best_loss)
            snapshot(args.logging_dir, run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc, 
                'state_dict': spk_model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1
        
        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))
    '''

    '''
    ## prediction 
    logger.info('===> loading best model for prediction')
    checkpoint = torch.load(os.path.join(args.logging_dir, run_name + '-model_best.pth'))
    #checkpoint = torch.load('./snapshot/cdc/CPC_spk_classification.pth')
    spk_model.load_state_dict(checkpoint['state_dict'])

    prediction_spk(args, cdc_model, spk_model, device, eval_loader, args.batch_size, args.frame_window)
    ## end 
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))
    '''

if __name__ == '__main__':
    main()