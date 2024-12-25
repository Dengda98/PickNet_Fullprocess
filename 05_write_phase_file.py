"""
    :file:     05_write_phase_file.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2023-11

    将拾取的震相导出为震相文件，其中使用台站名做标记
"""


import os, glob
import numpy as np
from obspy.geodetics import gps2dist_azimuth
from obspy import UTCDateTime
import pandas as pd
from collections import OrderedDict
from typing import List
import bisect

from io import TextIOWrapper



def process_one_input(
    phases:List[str],
    dirpath:str, outf:TextIOWrapper, maxarcdis:float, maxgap:float, 
    minPpicks:int, minSpicks:int,
    EVdict:dict, iev:int):
    '''
        每个事件的数据以各自文件夹保存，该函数针对某一个事件，决定是否可以写入到震相文件中
    '''
    # 精确到分的事件编号
    code = dirpath.split("/")[-1]

    # 找不到该事件元信息，跳过
    if code not in EVdict.keys():
        print(f"{iev}, {code} not found in EVdict, skipped.")
        return iev
    
    # 读取事件元信息
    event = EVdict[code]
    orig = UTCDateTime(event['orig'])
    evla = event['evla']
    evlo = event['evlo']
    evdp = event['evdp']
    mag = event['mag']

    # 方位角统计 
    azmLst = []
    num_p_picks = num_s_picks = 0
    p_picks_info = s_picks_info = []

    # P 
    if "P" in phases:
        p_stainfo_path = os.path.join(dirpath, "P_input/sta_info.npy")
        p_picks_path = os.path.join(dirpath, "P_input/input_fuse_picks.npy")
        if (not os.path.exists(p_stainfo_path)) or (not os.path.exists(p_picks_path)):
            print(f"{iev}, {code} P-phase picknet results not exists.")
            return iev
        
        p_picks_info = process_one_phase(code, p_stainfo_path, p_picks_path, minPpicks,  # 最少P震相数量
                                        maxarcdis, orig, azmLst, iev)
        num_p_picks = len(p_picks_info)
        # P震相不够，放弃这个事件
        if num_p_picks == 0:
            print(f"{iev}, {code} P-phase ({num_p_picks}) too little, skipped.")
            return iev

    # S
    if "S" in phases:
        s_stainfo_path = os.path.join(dirpath, "S_input/sta_info.npy")
        s_picks_path = os.path.join(dirpath, "S_input/input_fuse_picks.npy")
        if (not os.path.exists(s_stainfo_path)) or (not os.path.exists(s_picks_path)):
            print(f"{iev}, {code} S-phase picknet results not exists.")
            return iev
        s_picks_info = process_one_phase(code, s_stainfo_path, s_picks_path, minSpicks,  # 最少S震相数量
                                        maxarcdis, orig, azmLst, iev)
        num_s_picks = len(s_picks_info)
        # S震相不够，放弃这个事件
        if num_s_picks == 0:
            print(f"{iev}, {code} S-phase ({num_s_picks}) too little, skipped.")
            return iev

    # 方位角覆盖不好，放弃这个事件
    azmgap = azimuth_gap(azmLst)
    if azmgap >= maxgap:
        print(f"{iev}, {code} azimuth gap ({azmgap}) too large, skipped.")
        return iev
    
    #----------------------------------------------------------------------------------------
    iev += 1
    
    # 将震相写到文件中 
    year = orig.year
    month = orig.month
    day = orig.day
    hour = orig.hour
    minute = orig.minute
    second = orig.second + 1e-6 * orig.microsecond

    # 
    outf.write(f"{iev:>5d}{year:>5d}{month:>3d}{day:>3d}"
               f"{hour:>3d}{minute:>3d}{second:>7.2f}{0.0:>6.2f}"
               f"{evla:>10.4f}{0.0:>7.4f}{evlo:>10.4f}{0.0:>7.4f}"
               f"{evdp:>7.2f}{0.0:>5.1f}{mag:>5.1f}"  # evdp使用7.2f,多加一个空格
               f"{num_p_picks:>5d}{num_s_picks:>5d}{num_p_picks+num_s_picks:>5d}\n")
    
    # P 
    if "P" in phases:
        for ipick, pick_info in enumerate(p_picks_info):
            stnm, travt = pick_info 
            outf.write(f"{stnm:<8s}{travt+second:>10.2f} 1\n")
    # S 
    if "S" in phases:
        for ipick, pick_info in enumerate(s_picks_info):
            stnm, travt = pick_info 
            outf.write(f"{stnm:<8s}{travt+second:>10.2f} 2\n")

    outf.flush()

    return  iev

def process_one_phase(
    code:str, phase_stainfo_path:str, phase_picks_path:str, minpicks:int,
    maxarcdis:float, orig:UTCDateTime, azmLst:list, iev:int):
    '''
        从某个事件的某个震相(P/S)的结果中：
            1、计算方位角覆盖
            2、统计震相走时

    '''
    try:
        stainfo = np.load(phase_stainfo_path)
        picks = np.load(phase_picks_path)
    except:
        print(f"{iev}, {code} read error, skipped.")
        return []
    
    # 震相数不够，直接放弃这个事件
    if len(picks) < minpicks:
        return []
    
    # 计算方位角，并统计震相走时
    arcdisLst = []
    picks_info = []
    for s in picks:
        # 有拾取的数据索引 和 拾取的索引点
        dataidx, idx = map(lambda x:int(x), s.split("_"))

        # 该震相的基本信息
        stnm, _, starttime, arcdis, az = stainfo[dataidx].split("@")
        starttime = UTCDateTime(starttime)        

        arcdis = float(arcdis)
        az = np.rad2deg(float(az))

        # 震中距太大，跳过
        if arcdis > maxarcdis:
            continue

        # 方位角
        azmLst.append(az)

        # 统计走时，100表示采样率
        travt = starttime + idx/100 - orig 

        # 插入震中距
        _insert_idx = bisect.bisect_left(arcdisLst, arcdis)
        arcdisLst.insert(_insert_idx, arcdis)

        # 按照震中距排序插入
        picks_info.insert(_insert_idx, [stnm, travt])


    return picks_info


def azimuth_gap(azmLst:List[float]):
    azmLst = azmLst.copy()
    azmLst.extend([0.0, 360.0])
    azmLst.sort()
    azmLst = np.array(azmLst)
    return (azmLst[1:] - azmLst[:-1]).max()


def get_evDict(catalog_csv:str):
    df = pd.read_csv(catalog_csv, 
            dtype={'code':str, 
                'orig':str, 
                'evla':np.float64, 
                'evlo':np.float64, 
                'evdp':np.float64, 
                'mag':np.float64,}
        )
    
    # 删除 code 列中的重复值
    df = df[~df['code'].duplicated()]
    
    return df.set_index('code').to_dict('index')



def run(cfgs:dict, phases:List[str], catalog_csv:str, pattern:str, picknetDir:str, output:str):
    # 读入事件
    EVdict = get_evDict(catalog_csv)

    # 循环文件夹，将震相写到文件中 JP_hinet_2008_M2_ttt
    with open(output, "w") as f:
        dirLst = glob.glob(os.path.join(picknetDir, f"{pattern}"))
        dirLst.sort()

        maxarcdis = cfgs['maxarcdis']
        maxgap = cfgs['maxgap']
        minPpicks = cfgs['minPpicks']
        minSpicks = cfgs['minSpicks']

        iev = 0
        for dirpath in dirLst:
            iev = process_one_input(phases, dirpath, f, maxarcdis, maxgap, minPpicks, minSpicks, EVdict, iev)
            print(f"{iev}/{len(dirLst)}, {dirpath}")

        print(f"number of events: {iev}")



if __name__ == '__main__':
    import argparse 
    import yaml 
    parser = argparse.ArgumentParser()
    parser.add_argument("configpath")
    args = parser.parse_args()
    configpath = args.configpath

    with open(configpath, "r") as f:
        CFGS = yaml.safe_load(f)
        cfgs = CFGS['Phasefile']
        pattern = cfgs['pattern']
        phases = cfgs['phases']
        catalog_csv = CFGS['catalog_csv']
        picknetDir = CFGS['picknet_data_dir']

    run(cfgs, phases, catalog_csv, pattern, picknetDir, cfgs['output'])
