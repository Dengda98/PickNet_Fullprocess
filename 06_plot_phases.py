"""
    :file:     06_plot_phases.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2024-06

    绘制picknet的拾取情况
"""

import numpy as np 
import matplotlib.pyplot as plt
import os
import glob
import shutil
import subprocess
import pandas as pd


def plot_one_events_P(evinfo:dict, data:np.ndarray, datapicks:np.ndarray, stainfo:np.ndarray, chunk:int, output:str):
    tmpdir = ".tmp_fig"
    os.makedirs(tmpdir, exist_ok=True)

    # 标题行
    title = f"{evinfo['orig']},evloc=({evinfo['evla']:.3f},{evinfo['evlo']:.3f},{evinfo['evdp']:.1f}),M{evinfo['mag']:.1f}"

    ichunk = 0
    while ichunk < len(datapicks) or ichunk == 0:  # 至少画一次
        _picks = datapicks[ichunk:ichunk+chunk]
        figsize = [8, len(_picks)*0.5]
        if figsize[1] < 2:
            figsize[1] = 2
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for ipick in range(len(_picks)):
            dataidx, pickidx = map(lambda x:int(x), _picks[ipick].split("_"))
            # 该震相的基本信息
            stnm, _, starttime, arcdis, az = stainfo[dataidx].split("@")
            arcdis = float(arcdis)

            h, = ax.plot(data[dataidx, :]*0.7 + ipick, zorder=10)
            ax.vlines(pickidx, -0.7 + ipick, 0.7 + ipick, colors='r', zorder=20)

            ax.text(1200, ipick, f"{stnm} {arcdis:.3f}°", color=h.get_color(), fontsize=10, ha='left', va='center')

        ax.set_title(title, fontsize=13)
        ax.set_ylim([-1, len(_picks)])
        ax.set_xlim([0, 1200])
        ax.set_yticks([])
        fig.tight_layout()

        fig.savefig(f"{tmpdir}/tmp_{ichunk:0>5d}.pdf")
        fig.clf()
        plt.close()

        ichunk += chunk

    P = subprocess.Popen(f"pdftk {tmpdir}/*.pdf cat output {output}_P.pdf", shell=True)
    P.wait()
    shutil.rmtree(tmpdir)


def plot_one_events_S(evinfo:dict, data:np.ndarray, datapicks:np.ndarray, stainfo:np.ndarray, chunk:int, output:str):
    tmpdir = ".tmp_fig"
    os.makedirs(tmpdir, exist_ok=True)

    # 标题行
    title = f"{evinfo['orig']},evloc=({evinfo['evla']:.3f},{evinfo['evlo']:.3f},{evinfo['evdp']:.1f}),M{evinfo['mag']:.1f}"

    ichunk = 0
    while ichunk < len(datapicks) or ichunk == 0:  # 至少画一次
        _picks = datapicks[ichunk:ichunk+chunk]
        figsize = [8, len(_picks)*0.5]
        if figsize[1] < 2:
            figsize[1] = 2
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for ipick in range(len(_picks)):
            dataidx, pickidx = map(lambda x:int(x), _picks[ipick].split("_"))
            # 该震相的基本信息
            stnm, _, starttime, arcdis, az = stainfo[dataidx].split("@")
            arcdis = float(arcdis)
            
            h, = ax.plot(data[dataidx, :, 0]*0.5 + ipick+0.2, zorder=10)
            ax.plot(data[dataidx, :, 1]*0.5 + ipick-0.2, c=h.get_color(), zorder=10)
            ax.vlines(pickidx, -0.5 + ipick, 0.5 + ipick, colors='r', zorder=20)
            ax.text(1600, ipick, f"{stnm} {arcdis:.3f}°", color=h.get_color(), fontsize=10, ha='left', va='center')

        ax.set_title(title, fontsize=13)
        ax.set_ylim([-1, len(_picks)])
        ax.set_xlim([0, 1600])
        ax.set_yticks([])
        fig.tight_layout()

        fig.savefig(f"{tmpdir}/tnp_{ichunk:0>5d}.pdf")
        fig.clf()
        plt.close()

        ichunk += chunk

    P = subprocess.Popen(f"pdftk {tmpdir}/*.pdf cat output {output}_S.pdf", shell=True)
    P.wait()
    shutil.rmtree(tmpdir)


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


def run(cfgs:dict, catalog_csv:str):
    evDict = get_evDict(catalog_csv)

    os.makedirs(cfgs['output'], exist_ok=True)
    for path0 in cfgs['datalst']:
        pathLst = glob.glob(os.path.join(path0))
        pathLst.sort()
        for path in pathLst:
            code, pha = path.split("/")[-2:]
            pha = pha[0]
            data = np.load(os.path.join(path, "input.npy"))
            datapicks = np.load(os.path.join(path, "input_fuse_picks.npy"))
            stainfo = np.load(os.path.join(path, "sta_info.npy"))

            evinfo = evDict[code]

            if pha == 'P':
                plot_one_events_P(evinfo, data, datapicks, stainfo, 10, os.path.join(cfgs['output'], code))
            elif pha == 'S':
                plot_one_events_S(evinfo, data, datapicks, stainfo, 10, os.path.join(cfgs['output'], code))

            print(f"{path} done, npicks={len(datapicks)}.")


if __name__ == '__main__':
    import argparse 
    import yaml 
    parser = argparse.ArgumentParser()
    parser.add_argument("configpath")
    args = parser.parse_args()
    configpath = args.configpath

    with open(configpath, "r") as f:
        CFGS = yaml.safe_load(f)
        cfgs = CFGS['Plotphases']
        catalog_csv = CFGS['catalog_csv']

    run(cfgs, catalog_csv)
