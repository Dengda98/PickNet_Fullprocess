"""
    :file:     02b2_convert_stations_xml.py.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2024-06

    使用obspy下载的台站元数据转为简单的文本格式，保留基本信息
"""

import obspy
import glob
import os


def run(stationxml_path:str, stationtxt_path:str):
    outputdir = os.path.dirname(stationtxt_path)
    os.makedirs(outputdir, exist_ok=True)

    with open(stationtxt_path, "w") as f:
        counter = 1
        staxmlLst = glob.glob(os.path.join(stationxml_path, "*.xml"))
        staxmlLst.sort()
        for sta_name in staxmlLst:
            try:        
                inv = obspy.read_inventory(str(sta_name))
            except:
                continue
            sta_ID = '{}'.format(counter).rjust(6)
            try:
                sta_name = '{}'.format(inv.networks[0].code + '.' + inv.networks[0].stations[0].code).ljust(10)
                sta_lat = '{:.5f}'.format(inv.networks[0].stations[0].latitude).rjust(12)
                sta_lon = '{:.5f}'.format(inv.networks[0].stations[0].longitude).rjust(12)
                sta_elev = '{:.5f}'.format(inv.networks[0].stations[0].elevation/1000.0).rjust(12)
                write_line = sta_ID + ' ' + sta_name + sta_lat + sta_lon + sta_elev
            except:
                print(str(sta_name)+'ERROR')
                continue

            f.write(f"{write_line}\n")
            print(write_line)
            counter += 1


if __name__ == '__main__':
    import argparse 
    import yaml 
    parser = argparse.ArgumentParser()
    parser.add_argument("configpath")
    args = parser.parse_args()
    configpath = args.configpath

    with open(configpath, "r") as f:
        CFGS = yaml.safe_load(f)
        stationtxt_path = CFGS['stationtxt_path']
        stationxml_path = CFGS['Waveform']['obspykeys']['stationxml_path']

    run(stationxml_path, stationtxt_path)
    
