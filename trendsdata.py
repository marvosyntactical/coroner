import os
import sys
import glob
from typing import List, Optional
from googletrans import Translator

global langs, topics, kw_tmpl, ext

#templates
langs = ["en", "de", "fr", "es", "hu", "ru", "da"]
topics = ["finance", "social", "virus"]

#paths
kw_tmpl = "kw-tmpl/" #path relative to this script
ext = ".txt" #file extension

#files are named kw_tmpl/topic/lang.txt

translator = Translator()

def load_kws(file: str) -> List[str]:
    with open(file, "r") as in_file:
        keywords = in_file.readlines()
    r = []
    for kw in keywords:
        res = kw.strip()
        if res:
            r.append(res)
        
    print(r)
    return r

def save_kws(kw_list:List[str]=[], file: str ="en.txt"):
    tmp = []
    for kw in kw_list:
        tmp.append(kw.strip())
    out = []
    for kw in kw_list:
        out.append(kw+"\n")
    with open(file, "w") as out_file:
        out_file.writelines(kw_list)
    return

def translate_kws(kw_list, src: str = "en", trg: str = "de") -> List[str]:

    translations = []

    for kw in kw_list:

        translation = translator.translate(kw, dest=trg)

        translations.append(translation)
    
    return translations

def generate_translations(topic:str="virus", lang:str="en"):

    print(f"Generating translations for keyword bag for")
    print("topic={topic} from lang={lang} to other languages:")
    print(langs)

    file = lang+ext
    template = kw_tmpl+topic

    src_keywords = load_kws(template+file)

    print()
    print("First 10 Keywords: ")
    for i in range[9]:
        print(src_keywords[i])
    for trg in langs:

        if trg == lang:
            continue

        trg_keywords = translate_kws(src_keywords, lang, trg) 

        l_file = trg+ext

        save_kws(trg_keywords, l_file)

def search_existing_langs(topic:str, ext: str) -> List[str]:
    """
    searches directory for existing file and returns its name
    minus the extension 
    """
    wildcard = kw_tmpl+topic+"/*"

    matches = glob.glob(wildcard)

    existing_langs = []

    for match in matches:
        existing_langs.append(match.split(".")[-2])
    
    return existing_langs

def populate_topic(topic:str) -> List[str]:

    existing_langs = search_existing_langs(topic, ext)

    
    if len(existing_langs) != 1:
        if len(existing_langs) > 1:
            raise NotImplementedError("TODO: Implement merging of keyword files from two different languages")
        else:
            raise FileNotFoundError(f"The topic directory under ./kw-tmpl/{topic} does not seem to have any keyword file to generate translations from")

    src = existing_langs[0] 
    
    generate_translations(topic, src) 
