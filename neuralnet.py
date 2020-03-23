#-*- coding: utf-8 -*-

# romanorac's notebook:
# https://romanorac.github.io/machine/learning/2019/09/27/time-series-prediction-with-lstm.html

#preproc
import glob
import numpy as np
import pandas as pd

import random
from typing import List, Optional
#brainy juicy
import torch
import torch.nn as nn
import torch.nn.functional as F

kw_trends = "kw-trends/"
category = "example_category"

def load_data(topic:str=category) -> None:
    files = sorted(glob.glob(kw_trends+category+"/*"))
    df = pd.concat(map(pd.read_pickle, files))

    print(f"Loaded {len(files)} dataframes for topic '{topic}'")
    print(f"df.shape: {df.shape}")
    return df


def preproc(df:pd.DataFrame) -> pd.DataFrame:
    if df is None:
        df = load_data() #use defaults

    raise NotImplementedError    

 
def renormalize_and_integrate(old_data:np.ndarray=np.array([]), new_points:np.ndarray=np.array([])) ->np.ndarray:
    """
    old and new data are always normalized to the largest value
    
    if we start with a normalized array and repeatedly call
    this function with new_points the volume '100' spike on the start array
    out unit will always be 1% of the volume of that first highest 
    initial spike

    this is needed to train the neural network on consistent data
    an alternative would just be to adjust the nets bias every day lol 

    :param old_data: n_keywords x t_days_yesterday

    [
        [30,50,80],
        [20,40,40],
        [70,80,100] #rapid growth here -> below new datapoint is 5x 100
        [0,0,10]
    ]

    :param new_points: n_keywords x t_days_yesterday+1

    --- <- separates examples ... #TODO find out what the :param name: syntax says
    either like so if new data contains new spike:

    new_points = nd.array(
    [
        [6,10,16,20],
        [4,8,8,8],
        [14,16,20,100] #new datapoint 5x volume of yesterday -> google normalized this 
        [0,0,2,3]
    ])
    should return:
    [
        [30,50,80,100],
        [20,40,40,40],
        [70,80,100,500] 
        [0,0,10,15]
    ]
    ---
    or like so
    new_points = nd.array(
    [
        [30,50,80,100],
        [20,40,40,40],
        [70,80,100,80] #spike was reached yesterday -> new datapoint doesnt exceed this val -> no normalization by google
        [0,0,10,15]
    ])
    should return:
    new_points #exactly as it is above
    ---

    :return integrated: n_keywords x t_days_yesterday+1
    """

    assert integrated.shape == new_points.shape
    return integrated 

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init_()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,input, future=0, y=None) -> List[nn.module]:
        """
        future is somewhere between 1 and half seq len
        and determines from whereon we teacher force
        why is this random and only done on somewhere after the
        second half? 
        maybe to allow creative starts and not overfit 
        but doesnt this mean the model is trained on    
        self gen self gen self gen self gen JUMP TO GOLD gold gold gold 
        ??? i trust this implementation for now
        """
        outputs = []

        #reset the state of GRU 
        #state is kept until the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtye=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtye=torch.float32)

        # chunk time series data (what size does chunk chunk to??)
        for i, input_t in enumerate(input.chunk(input.size(1),dim=1)):
            #unroll one step

            h_t, c_t = self.gru(input_t, (h_t,c_t))
            output = self.linear(h_t)
            outputs.append(output)
        
        for i in range(future):
            if y is not None and random.random() > .5:
                output = y[:, [i]]

            h_t, c_t = self.gru(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)

        #concatenate time interval batched data back again
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class Optimization:
    """ Helper class to train, test, diagnose the LSTM"""

    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.futures = []

    @staticmethod
    def generate_batch_data(x,y,batch_size):
        for batch, i in enumerate(range(0,len(x) -bach_size, batch_size)):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch, batch
    
    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=100,
        n_epochs=15,
        teacher_forcing=None,
    ):
        seq_len = x_train.shape[1]
        for epoch in range(n_epochs):
            start_time = time.time()
            self.futures = []

            pass
            

            
    