"""
    :file:     07_gen_tomo_station.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2024-06

    从已有的台站文件中，根据空间范围筛选出台站
"""


def run(sta_input:str, sta_output:str,
        minlatitude:float, maxlatitude:float, 
        minlongitude:float, maxlongitude:float):
    with open(sta_input, "r") as fin, open(sta_output, "w") as fout:
        ista = 0
        for line in fin.readlines():
            idx, stnm, stla, stlo, stel = line.strip().split()
            stla, stlo, stel = map(lambda x:float(x), [stla, stlo, stel])
            if stla > maxlatitude or stla < minlatitude or \
               stlo > maxlongitude or stlo < minlongitude:
                print(f"{stnm} out of bound.")
                continue
               
            ista += 1
            fout.write(f"{ista:>6d} {stnm:<10s}{stla:>12.5f}{stlo:>15.5f}{stel:>12.5f}\n")


if __name__ == '__main__':
    # 输入台站文件
    sta_input = "../JPHinet_stations/hinet_sta"
    # 输出台站文件
    sta_output = "filt_sta"
    # 空间范围
    minlatitude = 36.0
    maxlatitude = 45.0
    minlongitude = 138.0
    maxlongitude = 146.0

    run(sta_input, sta_output, 
        minlatitude, maxlatitude,
        minlongitude, maxlongitude)