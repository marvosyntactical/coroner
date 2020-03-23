from pytrends.request import TrendReq
from trendsdata import load_kws, langs, topics, kw_tmpl, ext, read_proxies
import os
from typing import List, Optional
import numpy as np


# mock data
topic = "finance"

# log to path
log_path = "kw-trends/"

def state_lang(country: str="") -> str:
    """
    Returns language spoken in a country
    """
    raise NotImplementedError

def update_trends(proxies:List[str]=["http://179.108.169.71:8080"]) -> None:

    for topic in topics:
        for country in countries:

            lang = state_lang(country) #lang should be like in LANGUAGES.txt

            kw_list = load_kws(kw_tmpl+lang+ext)

            #pytrends initialization
            pytrends = TrendReq(hl=lang, tz=0)
            #pytrends = TrendReq(hl='lang', tz=360, timeout=(10,25), proxies=['https://34.203.233.13:80',], retries=2, backoff_factor=0.1)

            pytrends.build_payload(kw_list, cat=0, timeframe='today 1-w', geo='', gprop='')

def create_topics():
    with_keywords = os.listdir(kw_tmpl)
    for topic in with_keywords:
        new = log_path+topic
        try:
            os.mkdir(new)
        except FileExistsError:
            pass
   
def top_searches_no_outlier():
    raise NotImplementedError
