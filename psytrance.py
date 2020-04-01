from pytrends.request import TrendReq
from trendsdata import load_kws, langs, topics, kw_tmpl, ext 
import os
import json
from typing import List, Optional
import numpy as np
import pandas as pd
from functools import wraps
import warnings


# mock data
topic = "finance"
countries = ["germany, united kingdom"]

# log to path
global log_path, country_info_path, countriesINFO

log_path = "kw-trends/"
country_info_path = "langcountrycodes"
countriesINFO = country_info_path+"countriesINFO.pd"



def country_lang(country: str="", geo_lkp_file: str=countriesINFO, proximity=5) -> str:
    """

    converts country str (MUST be in geo_lkp_file["Country"] to language) 
     to list of languages spoken in that country
    

    :param country: country string    
    Returns language id spoken in a country
    """

    #preproc
    country = country.lower()

    df = pd.read_pickle(lkp_file) 

    index = df.index[df["Country"]==country]
    country_row = df.iloc[index]
    languages = country_row["Languages"]
    try:
        lang_codes = languages.item() #if succeeds, found exactly one matching country
    except ValueError:
        countries = df["Country"].to_list()

        
        lang_codes = None #failed, string didnt match any country
        warnings.warn(f"Couldnt find country={country} in country info file={geo_lkp_file};\n\
            here 5 closest matches: {close[:proximity-1]}")
        

    return lang_codes

def distanceFrom(s1, *args):
    #how to return copy of levenshteindist func wherein
    #s1 is specified???
        
    def levenshteinDistance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]
    return levenshteinDistance(s1,*args)

def country_iso(country: str="", geo_lkp_file: str=countriesINFO) -> str:
    """

    converts country str (MUST be in geo_lkp_file["Country"] to language) 
     to iso string

    :param country: country string    
    Returns language id spoken in a country
    """

    #preproc
    country = country.lower()

    df = pd.read_pickle(lkp_file) 

    index = df.index[df["Country"]==country]
    country_row = df.iloc[index]
    languages = country_row["Languages"]
    try:
        lang_codes = languages.item() #if succeeds, found exactly one matching country
    except ValueError:
        warnings.warn(f"Couldnt find country={country} in country info file={geo_lkp_file}")
        lang_codes = None #failed, string didnt match any country

    return lang_codes



def update_trends(proxies:List[str]=["http://179.108.169.71:8080"],topics=topics, countries=countries) -> None:

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


def main():


    raise NotImplementedError

if __name__ == "__main__":
    main()
