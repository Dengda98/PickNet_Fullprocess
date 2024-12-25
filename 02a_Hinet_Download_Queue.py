"""
    :file:     02_Hinet_Download_Queue.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2023-04

    基于HinetPy下载Hinet官网的地震数据，

    实际手动下载会发现，在你本地网速正常的情况下，当你在Hinet官网发出下载请求后
    大部分事件是在等待Hinet服务器整理数据，然后才能下载。

    该脚本通过使用多账号，将下载任务以队列的形式进行多进程并行下载，提高下载效率

"""


import os
import numpy as np
from HinetPy import Client 
from datetime import datetime
import shutil
import multiprocessing as mp
from typing import List
import pandas as pd
import tqdm
from copy import deepcopy
import fnmatch


def log(txt: str):
    string = f"[{datetime.today()}] {txt}"
    print(string)

def get_free_GB(path:str):
    '''获得当前路径下磁盘的可用空间，返回GB'''
    total, used, free = shutil.disk_usage(path)
    return  free / (1024**3)

def download_data(task_queue:mp.Queue, user:str, passwd:str, savedirlist:List[str],
                  minfreeGB:float, span:float, netcode:str):
    '''每个进程的任务函数'''
    client = Client(user, passwd, retries=5, max_sleep_count=60, sleep_time_in_seconds=3)
    savedir = savedirlist.pop(0)

    # 切换到一个临时目录，防止各进程的中间文件互相干扰
    tmp_wdir = f".tmp_{user}"
    if os.path.exists(tmp_wdir):
        shutil.rmtree(tmp_wdir)
    os.makedirs(tmp_wdir, exist_ok=True)
    os.chdir(tmp_wdir)

    while True:
        jpcode:str = task_queue.get()
        if jpcode is None:  # 结束信号
            break
        
        codesavedir = os.path.join(savedir, jpcode)
        
        # 根据可用空间判断是否需要更换存储目录
        freeGB = get_free_GB(savedir)
        if freeGB <= minfreeGB:
            log(f"{user}: {savedir} free space has only {freeGB:.2f} GB, savedir need to be changed.")
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

            log(f"{user}: savedir was changed to {savedir}.")

        
        try:
            log(f"{user}: {jpcode} downloading...")
            _ = client.get_continuous_waveform(
                netcode, 
                jpcode, 
                span, 
                outdir=jpcode,  # 由一些库历史问题（HinetPy中使用了os.rename函数，其兼容性不好），这里先保存到本地，然后再移动
            )
            # 再移动到目标目录
            shutil.move(jpcode, codesavedir)
            # time.sleep(1)
        except Exception as e:
            log(f"{user}: {jpcode} Exception: {str(e)}")
            continue
        
        if os.path.exists(codesavedir):
            log(f"{user}: {jpcode} download successfully, directory: {savedir}, free space: {freeGB:.2f} GB")
        else:
            log(f"{user}: {jpcode} download failed.")

    os.chdir("..")
    shutil.rmtree(tmp_wdir)


def update_mission(savedirlist:List[str], catLst:List[str]):
    # 根据保存目录中已有的文件夹名，对待下载任务做筛选
    for INPUT in savedirlist:
        Lst = os.listdir(INPUT)
        for code in tqdm.tqdm(Lst, desc=f"{INPUT} mission updating"):
            if not os.path.isdir(os.path.join(INPUT, code)):
                continue
            
            try:
                catLst.remove(code)
                # tqdm.tqdm.write(f"found existing code {code}, skipped.")
                continue
            except:
                pass


def run(cfgs:dict, pattern:str, catalog_csv:str, savedirlist:List[str], updatesavedirlist:List[str], minfreeGB:float, update:bool):
    # =====================================
    # 设置基本参数
    # =====================================
    savedirlist_bak = deepcopy(savedirlist)
    if update:
        savedirlist = updatesavedirlist

    # 转为绝对路径
    savedirlist = [os.path.abspath(p) for p in savedirlist]
    # 多个账号和密码
    userinfoLst = []
    for user in cfgs['accounts']:
        userinfoLst.append([user['username'], user['password']])

    # 下载波形时长，分钟
    span = cfgs['span']     
    netcode = cfgs['netcode']
    os.makedirs(savedirlist[0], exist_ok=True)

    # 读取事件目录
    catLst = pd.read_csv(catalog_csv, usecols=['code'], dtype=str).to_dict('list')['code']
    catLst = [s for s in catLst if fnmatch.fnmatch(s, pattern)]

    # 去重，排序
    catLst = list(set(catLst))
    catLst.sort()
    if update:
        update_mission(savedirlist_bak, catLst)


    # 创建队列，并添加任务
    log(f"read in {len(catLst)} events.")
    task_queue = mp.Queue()
    for orig in catLst:
        task_queue.put(orig)  # hinet下载开始时刻只允许精确到分钟

    # 启动多个进程
    processes = []
    for i in range(len(userinfoLst)):  
        user, passwd = userinfoLst[i]
        p = mp.Process(target=download_data, 
                       args=(task_queue, user, passwd, savedirlist.copy(), 
                             minfreeGB, span, netcode))
        p.start()
        processes.append(p)

        # 发送结束信号
    for _ in processes:
        task_queue.put(None)

    # 等待所有任务完成
    task_queue.close()
    task_queue.join_thread()

    for p in processes:
        p.join()
        

    log(f"ALL DONE. ")


if __name__ == '__main__':
    import argparse 
    import yaml 
    parser = argparse.ArgumentParser()
    parser.add_argument("configpath")
    args = parser.parse_args()
    configpath = args.configpath

    with open(configpath, "r") as f:
        CFGS = yaml.safe_load(f)
        cfgs = CFGS['Waveform']['hinetkeys']
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