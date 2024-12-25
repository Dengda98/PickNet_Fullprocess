"""
    :file:     08_gen_tomo_phase.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2024-06

    从已有的震相文件中，根据空间范围筛选出事件的震相，
    并根据设定的台站文件，将台站名更改为索引值。
    此时输出的震相文件为固定格式
"""

import numpy as np
from typing import List


def run(pha_input:str, pha_output:str, staLst:List[str],
        minlatitude:float, maxlatitude:float, 
        minlongitude:float, maxlongitude:float,
        mindepth:float, maxdepth:float, 
        minmagnitude:float, maxmagnitude:float):
    
    with open(pha_input, "r") as fin, open(pha_output, "w") as fout:
        iev = 0
        goodev = True
        Pobslines = []
        Sobslines = []
        evline = ""
        while True:
            line = fin.readline()
            if len(line) > 60 or len(line) == 0:
                goodev = True

                if len(evline)>0 and (len(Pobslines)>0 or len(Sobslines)>0):
                    iev += 1
                    nobsP = len(Pobslines)
                    nobsS = len(Sobslines)
                    nobsPS = nobsP+nobsS
                    evline = f"{iev:>5d}{evline}{nobsP:>5d}{nobsS:>5d}{nobsPS:>5d}\n"
                    fout.write(evline)
                    [fout.write(line0) for line0 in Pobslines]
                    [fout.write(line0) for line0 in Sobslines]
                    print(evline, end="")
                    Pobslines = []
                    Sobslines = []
                    evline = ""

                if len(line) == 0:
                    break

                _, year, month, day, hour, minute, secs, _, \
                    evla, _, evlo, _, evdp, _, evmag, _, _, _ = line.strip().split()
                year, month, day, hour, minute = map(lambda x:int(x), [year, month, day, hour, minute])
                secs, evla, evlo, evdp, evmag = map(lambda x:float(x), [secs, evla, evlo, evdp, evmag])
                if evla < minlatitude or evla > maxlatitude or \
                   evlo < minlongitude or evlo > maxlongitude or \
                   evdp < mindepth or evdp > maxdepth or \
                   evmag < minmagnitude or evmag > maxmagnitude:
                    goodev = False 
                    continue
                
                evline = f"{year:>5d}{month:>3d}{day:>3d}"\
                         f"{hour:>3d}{minute:>3d}{secs:>7.2f}{0.0:>6.2f}"\
                         f"{evla:>10.4f}{0.0:>7.4f}{evlo:>10.4f}{0.0:>7.4f}"\
                         f"{evdp:>6.2f}{0.0:>5.1f}{evmag:>5.1f}"

            elif goodev:
                stnm, seco, pha = line.strip().split()
                seco = float(seco)
                pha = int(pha)
                try:
                    ista = staLst.index(stnm) + 1
                    line0 = f"{ista:>5d}{seco:>10.2f}{pha:>2d}\n"
                    if pha==1:
                        Pobslines.append(line0)
                    elif pha==2:
                        Sobslines.append(line0)
                except Exception as e:
                    pass


if __name__ == "__main__":
    # 输入震相文件
    pha_input = "../phases_2009"
    # 输出震相文件
    pha_output = "filt_phases_2009"
    # 输入台站文件
    sta_input = "filt_sta"
    # 空间范围
    minlatitude = 36.0
    maxlatitude = 45.0
    minlongitude = 138.0
    maxlongitude = 146.0
    mindepth = -999
    maxdepth = 999
    minmagnitude = 2.0
    maxmagnitude = 10.0

    staLst = list(np.loadtxt(sta_input, usecols=1, dtype=str))

    run(pha_input, pha_output, staLst,
        minlatitude, maxlatitude,
        minlongitude, maxlongitude,
        mindepth, maxdepth,
        minmagnitude, maxmagnitude)