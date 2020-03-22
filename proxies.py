import os
import sys
import glob
from typing import List, Optional
import requests
from bs4 import BeautifulSoup as BS
import re
import json
import time
from copy import deepcopy


class ProxyListServer(object):
    # uses unix time

  
    def __init__(self,name:str="MyProxyListServer",interv_minutes:int=60):

        self._update = int(interv_minutes)
        self.name = name

        #below is static

        #use this file for saving
        self.proxies_path = "proxies/"
        self.proxies_file= "proxies"    
        self.json_ext = ".json"
        # proxy list self.site
        self.site = "https://www.proxynova.com/proxy-server-list/"
        # filter the proxies for quality
        self.min_speed = 1000 #ms #TODO find out why larger speed value better?
        self.min_uptime = 50
        # ip to url help
        self.secure_prefix = "https://"
    
        print(f"Successfully setup server {self.name}")

    def __repr__(self):
        return self.name

    def collect_contents(self, page:str=None) -> BS:
        if page is None:
            page = self.site

        htmlstr = requests.get(page) 
        soup = BS(htmlstr.text, "html.parser")
        
        print(f"Collected new IPs off {self.site} at {time.time()}")
        return soup

    def proxynova_soup_to_proxy_dict(self,borscht:BS) -> dict:
        """
        Returns dict with entries like so:
        {
            ID:int : List[str], len(List) == 3; List = [ip: str,port: str,speed: int,uptime_pct:int],
            2 : ["39.137.95.74", "8080", 2315,57]
        }
        """
        first_tbody = borscht.find("tbody") 
        rows = first_tbody.find_all("tr")

        proxytable = dict() 

        ID = 0

        #precompile regex patterns
        id_ = re.compile("'[^']+'")
        port_ = re.compile("[\w]+")
        uptime_pct_ = re.compile("[0-9]+\%")
        speed_ = re.compile("[0-9]+") #TODO bugtest

        for row in rows:
            columns = row.find_all("td")
            if len(columns) < 8: continue
            else: 
                ID += 1
                proxytable[ID] = []

            assert type(ID) == type(42)

            for j, col in enumerate(columns):
                #todo put all this code in 1 function

                if j==0:#proxy ip
                    ip = re.search(id_,str(col.text))
                    ip = ip.group(0)
                    ip = ip.replace("'","")
                    proxytable[ID].append(ip)
                elif j ==1:#proxy port
                    port = re.search(port_,str(col.text))
                    port = port.group(0)
                    port = port.strip()
                    proxytable[ID].append(port)
                elif j == 3:#speed
                    speed = re.search(speed_, str(col.text))
                    speed = speed.group(0)
                    speed = int(speed)
                    proxytable[ID].append(speed)
                elif j == 4:#uptime percentage
                    uptime = re.search(uptime_pct_,str(col.text))
                    uptime = uptime.group(0)
                    uptime = uptime.replace("%","")
                    uptime = int(uptime)
                    proxytable[ID].append(uptime)

        return proxytable

    def serialize_proxy_dict(self,proxy_dict:dict=dict(), file:str=None)-> str:
        if file is None:
            file = self.proxies_file

        filepath = self.proxies_path+file+self.json_ext

        with open(filepath, "w") as out:
            json.dump(proxy_dict,out)
        
        return filepath

    def read_proxies(self,file:str=None) -> dict:
        if file is None:
            file = self.proxies_file

        filepath = self.proxies_path+file+self.json_ext
        with open(filepath, "r") as proxy_file:
            proxydict = json.load(proxy_file)

        return proxydict 

    def update_proxy_dict(self,proxy_dict:dict=dict(), file:str=None,copy:str="") -> str:
        if file is None:
            file = self.proxies_file

        """
        update json by new proxy list dict
        if file doesnt exist, create one
        else shift IDS of new file to start counting after highest ID of 
        current ones

        if copy is specified, instead of appending to opened dict, copy to new file

        returns output filepath
        """
        old_path = self.proxies_path+file
        if not os.path.isfile(old_path):
            print(f"Creating json file {old_path+self.json_ext} !")
            self.serialize_proxy_dict(proxy_dict,file)
            return old_path
        #TODO at some point below here this function rewrites the old dictionary withe new one...

        proxies_so_far = self.read_proxies(file)

        increment = max([int(ID) for ID in proxies_so_far.keys()])
        #watch out to start IDs at 1!

        new = proxies_so_far
        for ID, cols in proxy_dict.items():
            if cols in proxies_so_far.values():
                continue
            incr_ID = int(ID)+increment
            new[incr_ID] = cols

        write_to = self.proxies_path+copy if copy else old_path
        self.serialize_proxy_dict(new,write_to)

        return write_to
 
    
    def filter_proxies(self,proxydict:dict=dict())-> dict:

        r = dict()
        for ID, cols in proxydict.items():
            if cols[2] < self.min_speed:
                continue
            elif cols[3] < self.min_uptime:
                continue
            else:
                r[ID] = cols
        return r

    def retrieve_proxies(self, file:str=None) -> List[str]:
        if file is None:
            file=self.proxies_file

        proxy_dict = self.read_proxies(file)

        #should be filtered, but just to be safe
        good_proxy_dict = self.filter_proxies(proxy_dict)
        assert proxy_dict == good_proxy_dict

        #create list of urls
        urls = []
        for ID, cols in sorted(good_proxy_dict.items()):
            url = self.secure_prefix+cols[0]+":"+cols[1]
            urls.append(url)
         
        return urls
        

    def update(self, copy=""):
        borschtsch = self.collect_contents()
        muesli = self.proxynova_soup_to_proxy_dict(borschtsch)
        better_muesli = self.filter_proxies(muesli)
        shelf = self.update_proxy_dict(better_muesli,copy=copy)
        print(f"\nGot some juicy fresh IPs off {self.site}")
 
    def run(self,until_minutes:int=24*60):
        #run for 1 day per default
        self.until_minutes = until_minutes

        startTime = time.time()
        update_counter = 0

        self.update()

        while True:

            currentTime = time.time()
            upTime = int((currentTime-startTime)/60)
            print(f"Been up since {upTime} minutes")
            print(f"Ive updated {'once' if update_counter==0 else str(update_counter+1)+' times'}")
            print(f"atm upTime%self._update > update_counter: {upTime%self._update} > {update_counter}")

            if upTime > self.until_minutes:

                print(f"{self} is shutting down after {upTime} minutes of Proxy List updating")
                break

            elif upTime % self._update > update_counter: #perform update

                update_counter+=1
                self.update()

            else:
                try:
                    print(f"Will check back in {self._update} minutes. Cya! :>")
                    time.sleep(self._update*60)
                except KeyboardInterrupt:
                    break
        print(f"\nStopping self.run().")
        print(f"Remember ips are available at {self.proxies_path+self.proxies_file+self.json_ext} ;)")

         
        










