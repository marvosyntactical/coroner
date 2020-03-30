#-*- coding: utf-8 -*-

# romanorac's notebook:
# https://romanorac.github.io/machine/learning/2019/09/27/time-series-prediction-with-lstm.html

#preproc
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import random
from typing import List, Optional, Tuple
#brainy juicy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

global kw_trends
kw_trends = "kw-trends/"


def load_data(topic:str=category) -> None:
    files = sorted(glob.glob(kw_trends+category+"/*"))
    df = pd.concat(map(pd.read_pickle, files))

    print(f"Loaded {len(files)} dataframes for topic '{topic}'")
    print(f"df.shape: {df.shape}")
    return df


def preproc(df:pd.DataFrame, splits: Tuple = (0.8,0.1,0.1)) -> Tuple[pd.DataFrame]:
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
    integrated=old_data #TODO 

    assert integrated.shape == new_points.shape
    return integrated 

# helper functions

def generate_sequence(scaler, model, x_sample, future=1000):
    """ Generate future values for x_sample with the model """
    y_pred_tensor = model(x_sample, future=future)
    y_pred = y_pred_tensor.cpu().tolist()
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred

def to_dataframe(actual, predicted):
    return pd.DataFrame({"actual": actual, "predicted":predicted})

def scale(scaler: StandardScaler,trdf: pd.Dataframe,
    valdf: pd.DataFrame, testdf:pd.DataFrame):
    # fits scaling to what has been seen in training
    # and transforms val and test to this 
    # TODO: what does transform actually mean?

    train_arr = scaler.fit_transform(trdf)
    val_arr = scaler.transform(valdf)
    test_arr = scaler.transform(testdf)

    return train_arr, val_arr, test_arr
    

def inverse_transform(scaler, df, columns):
    # what does this mean
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def transform_data(arr, seq_len):
    x,y = [], []
    for i in range(len(arr)-seq_len):
        x_i = arr[i:i+seq_len]
        y_i = arr[i+1 : i+ seq_len+1] #TODO is the shift we want to predict actually value at the next time interval?
    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)
    x_arr = Variable(torch.from_numpy(x_arr).float())
    y_arr = Variable(torch.from_numpy(y_arr).float())
    return x_arr, y_arr



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

class TrainManager:
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
        for batch, i in enumerate(range(0,len(x)-batch_size, batch_size)):
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
            
            train_loss = 0
            for x_batch, y_batch, batch_no in self.generate_batch_data(x_train,y_train, batch_size):
                y_pred = self._predict(x_batch, y_batch, seq_len, teacher_forcing)
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch) #why doesnt self.loss_fn take 
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() 
            self.scheduler.step()
            train_loss/= batch_no
            self.train_losses.append(train_loss)

            self._validation(x_val, y_val, batch_size)

            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}, Train loss: {train_loss}. Validation loss: {self.val_losses[-1]}. Avg future: {np.average(self.futures)}. Epoch time: {elapsed}")

    def _predict(self, x_batch, y_batch, seq_len, teacher_forcing):
        if teacher_forcing:
            future = random.randint(1, int(seq_len) / 2) #how many last steps to do teacher forcing for
            limit = x_batch.size(1) - future
            # model.forward pass
            # x_batch: batch x seq_len
            y_pred = self.model(x_batch[:, :limit], future=future, y=y_batch[:, limit])
        else:
            future = 0
            y_pred = self.model(x_batch)
        self.futures.append(future)
        return y_pred
    
    def _validation(self, x_val, y_val, batch_size):
        if x_val or y_val is None:
            return
        with torch.no_grad(): #context handler for no learning during val
            val_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_val, y_val, batch_size):
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch) 
        

    def eruieren(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            actual, predicted = [], []
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test, batch_size):
                y_pred = self.model(x_batch, future=future)
                y_pred = (
                    y_pred[:, -len(y_batch):] if y_pred.shape[1] > y_batch.shape[1] else y_pred
                )
                loss = self.loss_fn(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:,-1]).data.cpu().numpy.tolist()
                predicted += torch.squeeze(y_pred[:,-1]).data.cpu().numpy.tolist()
            test_loss /= batch
            return actual, predicted, test_loss
        
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")



def run_1():

    category = "social"

    data = load_data(category)
    
    train_df, val_df, test_df = preproc(data)

    scaler = StandardScaler()

    train_arr, val_arr, test_arr = scale()

    seq_len = 100

    x_train, y_train = transform_data(train_arr, seq_len)
    x_val, y_val = transform_data(val_arr, seq_len)
    x_test, y_test = transform_data(test_arr, seq_len)


    model_1 = Model(input_size=1, hidden_size=21, output_size=1)
    loss_fn_1 = nn.MSELoss()
    optimizer_1 = optim.Adam(model_1.parameters(), lr=1e-3)
    scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=5, gamma=.1)

    optimization_1 = TrainManager(model_1, loss_fn_1, optimizer_1, scheduler_1)

    teacher_forcing = True

    # train the model
    optimization_1.train(x_train, y_train, x_val, y_val, teacher_forcing)
    return 0
