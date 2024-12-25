"""
    :file:     convert.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2024-06


    从hinet官网下载全部hinet台站信息[链接]
    (https://www.hinet.bosai.go.jp/st_info/detail/dataset.php?LANG=en)，
    然后从其中提取适当信息

"""


import pandas as pd 
import numpy as np
import os

df = pd.read_csv("NIED_SeismicStation_20241012.csv")

# 只导出network_id=="'01'"的，即hinet的台
hinetdf = df[df['network_id']=="'01'"]  # 嵌套单引号

with open("hinet_sta", "w") as f:
    for i, (_,ser) in enumerate(hinetdf.iterrows()):
        stnm = ser['station_cd']
        stla = float(ser['latitude'])
        stlo = float(ser['longitude'])
        stel = float(ser['sensor_height(m)']/1e3)
        f.write(f"{i+1:>6d} {stnm:<10s}{stla:>12.5f}{stlo:>15.5f}{stel:>12.5f}\n")

