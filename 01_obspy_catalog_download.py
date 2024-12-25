"""
    :file:     01_obspy_catalog_download.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2024-06

    从美国IRIS官网下载地震目录，输出的csv已转为日本时间（东九区）
"""

import numpy as np 
from obspy.clients import fdsn
from obspy import *
import time

def run(cfgs:dict, catalog_csv:str):
    # 写csv文件题头
    with open(catalog_csv, "w") as f:
        f.write(f"code,orig,evla,evlo,evdp,mag\n")

    timezone = cfgs['timezone']
    client = fdsn.Client(cfgs['client'])

    starttime = UTCDateTime(cfgs['starttime']) - timezone*3600
    endtime = UTCDateTime(cfgs['endtime']) - timezone*3600

    minlat = cfgs['minlatitude']
    maxlat = cfgs['maxlatitude']
    minlon = cfgs['minlongitude']
    maxlon = cfgs['maxlongitude']
    mindep = cfgs['mindepth']
    maxdep = cfgs['maxdepth']
    minmag = cfgs['minmagnitude']
    maxmag = cfgs['maxmagnitude']
    
    chunk_length0 = chunk_length = cfgs['daychunk'] * 86400  # Query length in seconds
    nev = 0
    while starttime <= endtime:
        endtime0=min(endtime, starttime + chunk_length0)
        # 时间相同，中止下载
        if starttime==endtime0:
            break 
        
        cat0 = Catalog()
        try:
            cat0 = client.get_events(
                starttime=starttime,
                endtime=endtime0,
                minlatitude=minlat, maxlatitude=maxlat,
                minlongitude=minlon, maxlongitude=maxlon,
                mindepth=mindep, maxdepth=maxdep,
                minmagnitude=minmag, maxmagnitude=maxmag,
                orderby='time-asc',
            )
        except Exception as e:
            print(str(e))
            # 除了无数据
            if "No data available for request." not in str(e):
                print("reducing chunklength, tray again.")
                chunk_length0 = max(86400, chunk_length0/2)
                continue
        
        append_write_csv(catalog_csv, cat0, timezone)

        nev += len(cat0)
        print(f"当前块的事件数/总事件数：{len(cat0)}/{nev}, "
              f"{str(starttime + timezone*3600)}->{str(endtime0 + timezone*3600)}")

        time.sleep(1)
        starttime += chunk_length0

        chunk_length0 = chunk_length


def append_write_csv(outpath:str, cat:Catalog, timezone:int):
    with open(outpath, "a") as f:
        for ev in cat:
            orig = ev.origins[0].time + timezone*3600
            evla = ev.origins[0].latitude
            evlo = ev.origins[0].longitude
            evdp = 0.0 if ev.origins[0].depth is None else ev.origins[0].depth*1e-3
            code = f"{orig.year:4d}{orig.month:0>2d}{orig.day:0>2d}{orig.hour:0>2d}{orig.minute:0>2d}"  # code只精确到分钟
            mag = ev.magnitudes[0].mag
            f.write(f"{code},{str(orig)},{evla:.5f},{evlo:.5f},{evdp:.2f},{mag:.1f}\n")


if __name__ == '__main__':
    import argparse 
    import yaml 
    parser = argparse.ArgumentParser()
    parser.add_argument("configpath")
    args = parser.parse_args()
    configpath = args.configpath

    with open(configpath, "r") as f:
        CFGS = yaml.safe_load(f)
        catalog_csv = CFGS['catalog_csv']
        cfgs = CFGS['Catalog']

    run(cfgs, catalog_csv)