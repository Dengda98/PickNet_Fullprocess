"""
    :file:     03_cut_window.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2024-06

    将Hinet自定义的win数据转为sac，再根据理论到时截取时窗
"""

import os
import numpy as np
import numpy.lib.format as npf
import glob
try:
    from HinetPy import win32, Client
except Exception as e:
    print(str(e))
    pass

from obspy import read, UTCDateTime
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from obspy.geodetics import gps2dist_azimuth
from datetime import datetime
import subprocess 
from multiprocessing import Pool
import math
import pandas as pd
from typing import List
from collections import OrderedDict
import fnmatch


SAMPLING_RATE = 100 # Hz


def log(txt: str):
    string = f"[{datetime.today()}] {txt}"
    print(string)


def calc_arcdis_az(lat1, lon1, lat2, lon2): # evla, evlo, stla, stlo
    arcdis, az, baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    arcdis = arcdis/1e3 / 111.194926644

    return arcdis, az, baz


def save_one_sta(args):
    # print("Parameters:")
    # print(args)
    pha, out, code, sta, stainfo, evla, evlo, evdp, orig, input_len, win_before_ref, REFmodel, data_suffix = args

    try: 
        if pha == 'P':
            st = read(os.path.join(out, code, f"{sta}.*[ZU]*.{data_suffix}"), format=data_suffix.upper())
        else:
            if len(glob.glob(os.path.join(out, code, f"{sta}.*E*.{data_suffix}"))) > 0:
                st = read(os.path.join(out, code, f"{sta}.*E*.{data_suffix}"), format=data_suffix.upper()) 
                st += read(os.path.join(out, code, f"{sta}.*N*.{data_suffix}"), format=data_suffix.upper())
            elif len(glob.glob(os.path.join(out, code, f"{sta}.*X*.{data_suffix}"))) > 0:
                st = read(os.path.join(out, code, f"{sta}.*X*.{data_suffix}"), format=data_suffix.upper()) 
                st += read(os.path.join(out, code, f"{sta}.*Y*.{data_suffix}"), format=data_suffix.upper())
            else:
                raise FileNotFoundError
    except Exception as e:
        log(f"read {pha} {sta} failed ({str(e)}). skipped")
        return  None
    
    # 获得台站坐标
    # 始终从台站文件中获得台站坐标
    stla = stainfo['stla']
    stlo = stainfo['stlo']
    # stla = st[0].stats.sac['stla']
    # stlo = st[0].stats.sac['stlo']
    
    arcdis, az, baz = calc_arcdis_az(evla, evlo, stla, stlo)
    az = np.deg2rad(az)
    baz = np.deg2rad(baz)


    if pha == 'P':
        arrivals = REFmodel.get_travel_times(
            source_depth_in_km = evdp, 
            distance_in_degree = arcdis,
            phase_list=['P','p','Pn','Pg'])
    else:
        arrivals = REFmodel.get_travel_times(
            source_depth_in_km = evdp, 
            distance_in_degree = arcdis,
            phase_list=['S','s','Sn','Sg'])
    
    # 找初至
    A1st = 9999999.0
    for arrival_theo in arrivals:
        if A1st > arrival_theo.time:
            A1st = arrival_theo.time

    # 理论到时
    abs_theo_time = orig + A1st

    # 时窗时间点, 左边位置卡死，右边冗余1s
    win_t1 = abs_theo_time - win_before_ref*input_len/SAMPLING_RATE                 
    win_t2 = abs_theo_time + (1-win_before_ref)*input_len/SAMPLING_RATE + 1.0   

    # 数据太短，跳过
    if win_t1 < st[0].stats.starttime or win_t2 > st[0].stats.endtime:
        log(f"{sta}, window out of stream, skip.")
        return None

    begin_point = 0
    st = st.slice(win_t1, win_t2)

    try:
        if st[0].stats.sampling_rate > SAMPLING_RATE:
            st.resample(SAMPLING_RATE)
            # st_filt.resample(SAMPLING_RATE)
        elif st[0].stats.sampling_rate < SAMPLING_RATE:
            st.interpolate(SAMPLING_RATE)
            # st_filt.interpolate(SAMPLING_RATE)
    except:
        return None

    # 保存数据
    def _save_data(stream):
        input_data = np.zeros(input_len) if pha == 'P' else np.zeros([input_len,2])
        try:
            if pha == 'P':
                input_data[:] = stream[0].data[begin_point:begin_point+input_len]
            else:
                input_data[:, 0] = stream[0].data[begin_point:begin_point+input_len]  # E
                input_data[:, 1] = stream[1].data[begin_point:begin_point+input_len]   # N
        except:
            return None

        if pha == 'P':
            #normalize
            input_data -= np.mean(input_data)
            input_data /= np.linalg.norm(input_data,np.inf)
        else:
            input_data_raw = input_data.copy()
            # 仅对NE分量旋转
            # R
            if st[0].stats.channel[-1] == 'E' and st[1].stats.channel[-1] == 'N':
                input_data[:, 0] = - input_data_raw[:,0] * math.sin(baz) - input_data_raw[:,1] * math.cos(baz)
            input_data[:, 0] -= np.mean(input_data[:, 0])
            input_data[:, 0] /= np.linalg.norm(input_data[:, 0],np.inf)
            # T
            if st[0].stats.channel[-1] == 'E' and st[1].stats.channel[-1] == 'N':
                input_data[:, 1] = - input_data_raw[:,0] * math.cos(baz) + input_data_raw[:,1] * math.sin(baz)
            input_data[:, 1] -= np.mean(input_data[:, 1])
            input_data[:, 1] /= np.linalg.norm(input_data[:, 1],np.inf)

        return input_data
    
    input_data = _save_data(st)
    sta_info = '@'.join([sta, st[0].stats.channel, str(win_t1), str(arcdis), str(az)])

    return input_data, sta_info


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


def get_staDict(path:str):
    staDict = {}
    with open(path, "r") as f:
        for line in f.readlines():
            _, stnm, stla, stlo, stel = line.strip().split()
            stla, stlo, stel = map(lambda x:float(x), [stla, stlo, stel])
            staDict[stnm] = {
                'stla': stla,
                'stlo': stlo,
                'stel': stel,
            }
    return staDict
    

def get_pathLst(pattern:str, INPUTLST:List[str], update:bool, OUTPUT:str):
    # 遍历多个存有原始波形文件的文件夹，根据最终code排序
    pathLst = []
    codeLst = []
    for INPUT in INPUTLST:
        Lst = os.listdir(INPUT)
        for code in Lst:
            if not os.path.isdir(os.path.join(INPUT, code)):
                continue
            
            # 不符合表达式
            if not fnmatch.fnmatch(code, pattern):
                continue

            # 结果已存在
            if update and \
               os.path.exists(os.path.join(OUTPUT, code, "P_input/input.npy")) and \
               os.path.exists(os.path.join(OUTPUT, code, "S_input/input.npy")):
                continue
            
            # 目标文件中存在重复，跳过
            if code in codeLst:
                log(f"found duplicate code {code}, skipped.")
                continue

            pathLst.append(os.path.join(INPUT, code))
            codeLst.append(code)
    
    # 根据code做排序
    pathLst[:] = [pathLst[i] for i in np.argsort(codeLst)]

    return pathLst

# ==============================================================================
# ==============================================================================
# ==============================================================================
def run(cfgs:dict, catalog_csv:str, pattern:str, update:bool, INPUTLST:List[str], stapath:str, OUTPUT:str):
    os.makedirs(OUTPUT, exist_ok=True)

    # 以字典式索引找事件信息 
    evDict = get_evDict(catalog_csv)

    # 待处理的路径列表
    pathLst = get_pathLst(pattern, INPUTLST, update, OUTPUT)
    NpathLst = len(pathLst)
    log(f"get {NpathLst} events to be extracted.")

    # 判断待处理数据的后缀名，存在.cnt文件表示从日本Hinet下载，需要转为sac文件
    download_from_JP = False 
    data_suffix = ""
    if len(glob.glob(os.path.join(pathLst[0], '*.cnt'))) > 0:
        download_from_JP = True
        data_suffix = "SAC"
    else:
        data_suffix = glob.glob(os.path.join(pathLst[0], '*'))[0].split(".")[-1]

    # 台站文件
    if not os.path.exists(stapath):
        raise OSError(f"{stapath} not exists.")
    
    staDict = get_staDict(stapath)
    stakeys = list(staDict.keys())


    REFmodel = TauPyModel(cfgs['refmodel'])
    for ipath, path in enumerate(pathLst):
        suffix, code = path.split('/')[-2:]
        if code not in evDict.keys():
            log(f"[{ipath+1}/{NpathLst}] {code} not exists in catalog, skipped.")
            continue

        # 先复制到本地
        P = subprocess.Popen(f"cp {path} {OUTPUT}/ -r", shell=True)
        P.wait()

        # Hinet数据转为sac
        if download_from_JP:
            try:
                win32.extract_sac(
                    data=glob.glob(os.path.join(OUTPUT, code, "*.cnt"))[0],
                    ctable=glob.glob(os.path.join(OUTPUT, code, "*.ch"))[0],
                    outdir=os.path.join(OUTPUT, code),
                    processes=None)
            except Exception as e:
                log(f"[{ipath+1}/{NpathLst}] {code} win32 error ({str(e)}). skipped.")
                continue

            log(f"[{ipath+1}/{NpathLst}] {code} from {suffix} sac done.")


        # 获得台站列表
        staLst = list(set([".".join(nm.split('/')[-1].split(".")[:2]) 
                           for nm in glob.glob(os.path.join(OUTPUT, code, f'*.{data_suffix}'))]))
        staLst.sort()

        evla = evDict[code]['evla']
        evlo = evDict[code]['evlo']
        evdp = evDict[code]['evdp']
        orig = UTCDateTime(evDict[code]['orig'])


        try:
            # 根据理论到时
            # P
            args = [('P', OUTPUT, code, sta, staDict[sta], evla, evlo, evdp, orig, 1200, 0.5, REFmodel, data_suffix) 
                    for sta in staLst if sta in stakeys]
            P = Pool(cfgs['nproc'])
            log(f"[{ipath+1}/{NpathLst}] {code} from {suffix} P slice start.")
            resLst = P.map(save_one_sta, args)
            log(f"[{ipath+1}/{NpathLst}] {code} from {suffix} P slice done.")
            save_data_list = list()
            sta_info = list()
            for res in resLst:
                if res is None:
                    continue 
                save_data_list.append(res[0])
                sta_info.append(res[1])

            P.close()
            P.join()
            
            if len(save_data_list) > 0:
                _dir = os.path.join(OUTPUT, code, "P_input")
                os.makedirs(_dir, exist_ok=True)
                np.save(f'{_dir}/input.npy', save_data_list)
                np.save(f'{_dir}/sta_info.npy', sta_info)
        except Exception as e:
            log(f"[{ipath+1}/{NpathLst}] {code} P slice error ({str(e)}). skipped.")
            continue
        
        try:
            # S
            args = [('S', OUTPUT, code, sta, staDict[sta], evla, evlo, evdp, orig, 1600, 0.5, REFmodel, data_suffix) 
                    for sta in staLst if sta in stakeys]
            P = Pool(cfgs['nproc'])
            log(f"[{ipath+1}/{NpathLst}] {code} from {suffix} S slice start.")
            resLst = P.map(save_one_sta, args)
            log(f"[{ipath+1}/{NpathLst}] {code} from {suffix} S slice done.")
            save_data_list = list()
            sta_info = list()
            for res in resLst:
                if res is None:
                    continue 
                save_data_list.append(res[0])
                sta_info.append(res[1])

            P.close()
            P.join()
            
            if len(save_data_list) > 0:
                _dir = os.path.join(OUTPUT, code, "S_input")
                os.makedirs(_dir, exist_ok=True)
                np.save(f'{_dir}/input.npy', save_data_list)
                np.save(f'{_dir}/sta_info.npy', sta_info)
        except Exception as e:
            log(f"[{ipath+1}/{NpathLst}] {code} S slice error ({str(e)}). skipped.")
            continue
            
        log(f"[{ipath+1}/{NpathLst}] {code} from {suffix} slice done.")

        # 去除中间文件
        P = subprocess.Popen(f"rm {os.path.join(OUTPUT, code, f'*.*')} -f", shell=True)
        P.wait()
        log(f"[{ipath+1}/{NpathLst}] {code} from {suffix} sac,ch,cnt remove done.")


if __name__ == '__main__':
    import argparse 
    import yaml 
    parser = argparse.ArgumentParser()
    parser.add_argument("configpath")
    args = parser.parse_args()
    configpath = args.configpath

    with open(configpath, "r") as f:
        CFGS = yaml.safe_load(f)
        cfgs = CFGS['Cutwindow']
        pattern = cfgs['pattern']
        update = cfgs['update']
        catalog_csv = CFGS['catalog_csv']
        winSaveDirLst = CFGS['waveform_dirs']
        output = CFGS['picknet_data_dir']

    stapath = ""
    try:
        stapath = CFGS['stationtxt_path']
    except:
        pass

    

    run(cfgs, catalog_csv, pattern, update, winSaveDirLst, stapath, output)
