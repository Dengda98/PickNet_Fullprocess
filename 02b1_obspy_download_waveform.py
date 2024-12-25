"""
    :file:     02b1_obspy_download_waveform.py.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2024-06

    使用obspy的批量下载工具从全球开放数据库中下载地震波形，以及对应的台站元数据
"""

import os
import pandas as pd
import numpy as np 
from obspy import *
from obspy.clients.fdsn.mass_downloader import \
    Domain, CircularDomain, RectangularDomain,\
    Restrictions, MassDownloader
from typing import List
import time
import logging
import shutil
from datetime import datetime
from copy import deepcopy
import tqdm
import fnmatch

from pytz import UTC
logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
# logger.setLevel(logging.ERROR)  


def log(txt: str):
    string = f"[{datetime.today()}] {txt}"
    print(string)

def get_free_GB(path:str):
    '''获得当前路径下磁盘的可用空间，返回GB'''
    total, used, free = shutil.disk_usage(path)
    return  free / (1024**3)


def download_one_event(code:str, savedir:str, mdl:MassDownloader, domain:Domain, 
                       start_shift:float, end_shift:float,
                       Restrictions_kwargs:dict, stations_dir:str):
    
    orig_in_min:UTCDateTime = UTCDateTime(code)
    restrictions = Restrictions(
        starttime=orig_in_min + start_shift,
        endtime=orig_in_min + end_shift,
        # You might not want to deal with gaps in the data. If this setting is
        # True, any trace with a gap/overlap will be discarded.
        reject_channels_with_gaps=True,

        **Restrictions_kwargs
    )

    # time.sleep(np.random.uniform(2, 6))

    codesavedir = os.path.join(savedir, code)
    # 循环尝试5次
    for _ in range(5):
        t1 = time.time()
        t2 = 0
        try:
            mdl.download(
                domain, restrictions, 
                mseed_storage=(f"{codesavedir}/" + 
                            "{network}.{station}.{channel}.{location}.{starttime}.{endtime}.mseed"),
                stationxml_storage=stations_dir,
                threads_per_client=3
                )
            t2 = time.time()
        except Exception as e:
            print(str(e), "try again.")
            time.sleep(np.random.uniform(2, 6))
            continue 

        # if abs(t2-t1) < 0.5:
        #     print("some error, try again")
        #     time.sleep(np.random.uniform(2, 6))
        #     continue

        break
    
    return codesavedir

def update_mission(savedirlist:List[str], catLst:List[str]):
    # 根据保存目录中已有的文件夹名，对待下载任务做筛选
    for INPUT in savedirlist:
        Lst = os.listdir(INPUT)
        for code in tqdm.tqdm(Lst, desc=f"{INPUT} updating"):
            if not os.path.isdir(os.path.join(INPUT, code)):
                continue
            
            try:
                catLst.remove(code)
                tqdm.tqdm.write(f"found existing code {code}, skipped.")
                continue
            except:
                pass


def run(cfgs:dict, pattern:str, catalog_csv:str, savedirlist:List[str], updatesavedirlist:List[str], minfreeGB:float, update:bool):
    mdl = MassDownloader(providers=cfgs['providers'])  # 包括IRISPH5

    domaincfgs = cfgs['domain']
    if domaincfgs['domainshape'] == 'Rectangular':
        domain = RectangularDomain(
            minlatitude=domaincfgs['minlatitude'], maxlatitude=domaincfgs['maxlatitude'],
            minlongitude=domaincfgs['minlongitude'], maxlongitude=domaincfgs['maxlongitude'],
        )
    elif domaincfgs['domainshape'] == 'Circular':
        domain = CircularDomain(
            latitude=domaincfgs['latitude'], longitude=domaincfgs['longitude'],
            minradius=domaincfgs['minradius'], maxradius=domaincfgs['maxradius'],
        )
    else:
        raise NotImplementedError(f"{domaincfgs['domainshape']} is not supported.")
    
    Restrictions_kwargs:dict = cfgs['Restrictions']
    start_shift = Restrictions_kwargs['starttime']
    end_shift = Restrictions_kwargs['endtime']
    Restrictions_kwargs.pop('starttime')
    Restrictions_kwargs.pop('endtime')

    savedirlist_bak = deepcopy(savedirlist)
    if update:
        savedirlist = updatesavedirlist
    
    # 转为绝对路径
    savedirlist = [os.path.abspath(p) for p in savedirlist]
    savedir = savedirlist.pop(0)
    os.makedirs(savedir, exist_ok=True)

    # 读取事件目录
    catLst = pd.read_csv(catalog_csv, usecols=['code'], dtype=str).to_dict('list')['code']
    catLst = [s for s in catLst if fnmatch.fnmatch(s, pattern)]

    # 去重，排序
    catLst = list(set(catLst))
    catLst.sort()
    if update:
        update_mission(savedirlist_bak, catLst)

    # 台站元信息保存
    stations_dir = cfgs['stationxml_path']

    for iev, code in enumerate(catLst):
        # 判断是否要切换目录
        freeGB = get_free_GB(savedir)
        if freeGB <= minfreeGB:
            log(f"{savedir} free space has only {freeGB:.2f} GB, savedir need to be changed.")
            if len(savedirlist)>0:
                savedir = savedirlist.pop(0)
            else:
                while True:
                    try:
                        savedir = input("Enter a new savedir: ")
                        os.makedirs(savedir, exist_ok=True)
                    except:
                        print("wrong input, try again.")
                        continue 
                    break
            os.makedirs(savedir, exist_ok=True)

            log(f"savedir was changed to {savedir}.")

        try:
            codesavedir = download_one_event(code, savedir, mdl, domain, 
                            start_shift, end_shift, 
                            Restrictions_kwargs, stations_dir)
        except Exception as e:
            log(str(e))
            log(f"{code} download failed.")
            continue
        
        if os.path.exists(codesavedir):
            log(f"{code} download successfully, directory: {savedir}, free space: {freeGB:.2f} GB")
        else:
            log(f"{code} download failed.")


        log(f"{code} {iev+1}/{len(catLst)} events done.")


    log('Done')



if __name__ == '__main__':
    import argparse 
    import yaml 
    parser = argparse.ArgumentParser()
    parser.add_argument("configpath")
    args = parser.parse_args()
    configpath = args.configpath

    with open(configpath, "r") as f:
        CFGS = yaml.safe_load(f)
        cfgs = CFGS['Waveform']['obspykeys']
        catalog_csv = CFGS['catalog_csv']
        # 多个本地下载路径，防止磁盘空间不够
        savedirlist = CFGS['waveform_dirs']
        updatesavedirlist = CFGS['Waveform']['update_waveform_dirs']
        # 磁盘最小可容许存储空间GB，此时切换下载路径
        minfreeGB = CFGS['Waveform']['minfreeGB']
        # 是否以更新模式下载
        update = CFGS['Waveform']['update']
        # 通配符
        pattern = CFGS['Waveform']['pattern']

    run(cfgs, pattern, catalog_csv, savedirlist, updatesavedirlist, minfreeGB, update)