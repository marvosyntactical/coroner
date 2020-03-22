#-*- coding: utf-8 -*-

# romanorac's notebook:
# https://romanorac.github.io/machine/learning/2019/09/27/time-series-prediction-with-lstm.html

#preproc
import glob
import numpy as np
import pandas as pd

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
        df = load_data()

    df_vwap = 
    
    return None

