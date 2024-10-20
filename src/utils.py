# References:
# https://github.com/sindhura97/STraTS


"""This file contain common utility functions."""
from datetime import datetime
import argparse
import os
import random
from pytz import timezone
from tqdm import tqdm
tqdm.pandas()
from transformers import set_seed
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Optimizer
from typing import Any, Union
import matplotlib.pyplot as plt
import pickle


def get_curr_time() -> str:
    """Get current date and time in PST as str."""
    return datetime.now().astimezone(
            timezone('US/Pacific')).strftime("%d/%m/%Y %H:%M:%S")

def parse_args(model_type,
               max_obs = 880,
               hid_dim = 32,
               num_layers = 2,
               num_heads = 4,
               dropout = 0.2,
               attention_dropout = 0.2,
               kernel_size = 4,
               max_timesteps = 880,
               hours_look_ahead = 24,
               ref_points = 24,
               max_epochs = 50,
               lr = 5e-4,
               gradient_accumulation_steps = 32,
               pretrain = 0,
               load_ckpt_path = None,
               num_ts_feat = 51,
               num_demo_feat = 3,
               ) -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()

    # dataset related arguments
    parser.add_argument('--dataset', type=str, default='mimic_iii')
    parser.add_argument('--train_frac', type=float, default=0.7)
    parser.add_argument('--run', type=str, default='1o10')#different data for different runs - to give an estimate of variance of evaluation metrics
    
    # model related arguments
    parser.add_argument('--model_type', type=str, default=model_type,
                        choices=['gru','tcn','sand','grud','interpnet',
                                 'strats','istrats'])
    ##  strats and istrats
    parser.add_argument('--max_obs', type=int, default=max_obs)
    parser.add_argument('--hid_dim', type=int, default=hid_dim)
    parser.add_argument('--num_layers', type=int, default=num_layers)
    parser.add_argument('--num_heads', type=int, default=num_heads)
    parser.add_argument('--dropout', type=float, default=dropout)
    parser.add_argument('--attention_dropout', type=float, default=attention_dropout)
    ## gru: hid_dim, dropout
    ## tcn: dropout, filters=hid_dim
    parser.add_argument('--kernel_size', type=int, default=kernel_size)
    ## sand: num_layers, hid_dim, num_heads, dropout
    parser.add_argument('--r', type=int, default=24)
    parser.add_argument('--M', type=int, default=12)
    ## grud: hid_dim, dropout
    parser.add_argument('--max_timesteps', type=int, default=max_timesteps)
    parser.add_argument('--num_ts_feat', type=int, default=num_ts_feat)
    parser.add_argument('--num_demo_feat', type=int, default=num_demo_feat)
    ## interpnet: hid_dim
    parser.add_argument('--hours_look_ahead', type=int, default=hours_look_ahead)
    parser.add_argument('--ref_points', type=int, default=ref_points)

    # training/eval realated arguments
    parser.add_argument('--pretrain', type=int, default=pretrain)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default='')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--max_epochs', type=int, default=max_epochs)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=gradient_accumulation_steps)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--print_train_loss_every', type=int, default=100)
    parser.add_argument('--validate_after', type=int, default=-1)
    parser.add_argument('--validate_every', type=int, default=None)

    parser.add_argument('--load_ckpt_path', type=str, default=load_ckpt_path)

    #args = parser.parse_args()
    return parser

def set_all_seeds(seed: int) -> None:
    """Function to set seeds for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count()>0:
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    set_seed(seed)

def set_output_dir(args: argparse.Namespace) -> None:
    """Function to automatically set output dir 
    if it is not passed in args."""
    if args.output_dir is None:
        if args.pretrain:
            args.output_dir = '../outputs/'+args.dataset+'/'+args.output_dir_prefix+'pretrain/'
        else:
            args.output_dir = '../outputs/'+args.dataset+'/'+args.output_dir_prefix
            args.output_dir += args.model_type 
            if args.model_type in ['strats','new']:
                for param in ['num_layers', 'hid_dim', 'num_heads', 'dropout', 'attention_dropout', 'lr','max_epochs']:
                    args.output_dir += ','+param+':'+str(getattr(args, param))
            elif args.model_type in ['gru','grud']:
                for param in ['hid_dim','dropout', 'lr', 'gradient_accumulation_steps','max_epochs']:
                    args.output_dir += ','+param+':'+str(getattr(args, param))
            for param in ['train_frac','run']:
                args.output_dir += '|'+param+':'+str(getattr(args, param))
    os.makedirs(args.output_dir, exist_ok=True)


def save_results(args: argparse.Namespace, results) -> None:
    file_name = args.model_type + '_results.pkl'
    with open(os.path.join(args.output_dir,file_name), 'wb') as f:
        pickle.dump(results, f)
    plt.plot(results['epoch'], results['train_auroc'])
    plt.plot(results['epoch'], results['val_auroc'])
    plt.plot(results['epoch'], results['test_auroc'])
    plt.title(args.model_type)
    plt.xlabel('train epoch')
    plt.ylabel('auroc')
    plt.legend(['train_auroc', 'val_auroc', 'test_auroc'])
    plt.savefig(os.path.join(args.output_dir,'auroc.png'))

class Logger: 
    """Class to write message to both output_dir/filename.txt and terminal."""
    def __init__(self, output_dir: str=None, filename: str=None) -> None:
        if filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message: Any, show_time: bool=True) -> None:
        "write the message"
        message = str(message)
        if show_time:
            # if message starts with \n, print the \n first before printing time
            if message.startswith('\n'): 
                message = '\n'+get_curr_time()+' >> '+message[1:]
            else:
                message = get_curr_time()+' >> '+message
        print (message)
        if hasattr(self, 'log'):
            with open(self.log, 'a') as f:
                f.write(message+'\n')


class CycleIndex:
    """Class to generate batches of training ids, 
    shuffled after each epoch.""" 
    def __init__(self, indices:Union[int,list], batch_size: int,
                 shuffle: bool=True) -> None:
        if type(indices)==int:
            indices = np.arange(indices)
        self.indices = indices
        self.num_samples = len(indices)
        self.batch_size = batch_size
        self.pointer = 0
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle

    def get_batch_ind(self):
        """Get indices for next batch."""
        start, end = self.pointer, self.pointer + self.batch_size
        # If we have a full batch within this epoch, then get it.
        if end <= self.num_samples:
            if end==self.num_samples:
                self.pointer = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
            else:
                self.pointer = end
            return self.indices[start:end]
        # Otherwise, fill the batch with samples from next epoch.
        last_batch_indices_incomplete = self.indices[start:]
        remaining = self.batch_size - (self.num_samples-start)
        self.pointer = remaining
        if self.shuffle:
            np.random.shuffle(self.indices)
        return np.concatenate((last_batch_indices_incomplete, 
                               self.indices[:remaining]))
    
