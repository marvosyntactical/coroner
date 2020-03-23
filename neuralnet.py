#-*- coding: utf-8 -*-

# romanorac's notebook:
# https://romanorac.github.io/machine/learning/2019/09/27/time-series-prediction-with-lstm.html

#preproc
import glob
import numpy as np
import pandas as pd


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

    def forward(self,input, future=0, y=None) -> Lis[nn.module]:
        outputs = []

        #reset the state of GRU
        h_t = torch.zeros(input.size(0), self.hidden_size, dtye=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtye=torch.float32)
        # Latest TODO: implement
        raise NotImplementedError
    