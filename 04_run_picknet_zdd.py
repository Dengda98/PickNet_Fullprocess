"""
    :file:     04_run_picknet_zdd.py  
    :author:   Zhu Dengda (zhudengda@mail.iggcas.ac.cn)  
    :date:     2023-11

    对于有多个数据的情况，模型只要导入一次，剩下的只需要循环逐个配置即可
"""


import os
# 这样建立session是没有cuda日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 WARNING 和 ERROR 级别的日志
    
import tensorflow as tf

from picknet.fcn.tester import FCNTester
from picknet import DEFAULT_P_CONFIG, DEFAULT_S_CONFIG, P_WAVE_MODEL, S_WAVE_MODEL

import glob
from typing import List
import numpy as np
import numpy.lib.format as npf
import tqdm
import shutil
import fnmatch


def get_session(gpu_fraction = None, gpuid = 0):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    if num_threads is None:
        return tf.Session(config=tf.ConfigProto())
    else:
        num_threads = int(num_threads)
        
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def get_free_GB(path:str):
    '''获得当前路径下磁盘的可用空间，返回GB'''
    total, used, free = shutil.disk_usage(path)
    return  free / (1024**3)


def merge_all_data(phase:str, pathLst:List[str], merge_data_path:str):
    # 内存映射读取所有数据
    mmap_data = [np.load(path, mmap_mode="r") for path in pathLst]
    refshape = mmap_data[0].shape
    nrows = sum(data.shape[0] for data in mmap_data)

    shape = (nrows, *refshape[1:]) 
    merge_data = np.lib.format.open_memmap(merge_data_path, "w+", 'float64', shape)

    # 写入合并数据
    irow = 0
    for data in tqdm.tqdm(mmap_data, desc=f"Merging {phase} Data"):
        nrow = data.shape[0]
        merge_data[irow:irow+nrow, ...] = data
        irow += nrow 



def run(phase:str, merge:bool, batchsize:int, gpuid:int, pathLst:List[str], update:bool, raw_output:bool=False):
    print('Setting Tester')

    # 读入基本模板的yaml
    DEFAULT_CONFIG = DEFAULT_P_CONFIG if phase=='P' else DEFAULT_S_CONFIG
    MODEL_PATH = P_WAVE_MODEL if phase=='P' else S_WAVE_MODEL
    tester = FCNTester(DEFAULT_CONFIG)
    # 修改模型路径
    tester.cfgs['save_dir'] = MODEL_PATH
    # 修改批量大小
    tester.cfgs['batch_size_test'] = batchsize
    
    npath = len(pathLst)
    print("len(npath)=", npath)

    # 判断存储空间是否足够
    # free = get_free_GB(".")
    # need_space = 0.0
    # for path in pathLst:
    #     with open(path, "rb") as fp:
    #         nsta = npf.read_magic(fp)[0][0]
    #     if (not update) or (not os.path.exists(path[:-4]+"_fuse_picks.npy")):
    #         need_space += nsta*1  # 1是估计拾取结果字符串的保存
    #     if raw_output and ((not update) or (not os.path.exists(path[:-4]+"_output.npy"))):
    #         need_space += nsta*8  # 8是双精度浮点数

    # need_space *= 1200 if phase=='P' else 1600*2
    # need_space /= 1024**3
    # print(f"Need Space: {need_space:.2f} GB, Free Space: {free:.2f} GB")

    # if need_space >= free:
    #     raise ValueError(f"free space ({free:.2f} GB) <= ({need_space+1:.2f} GB), please release space or change output directory.")


    # 每次推理前重置默认图，释放之前的图结构
    tf.reset_default_graph()

    # 提前建立一次session
    session = get_session(0.5, gpuid)
    tester.setup(session)

    # 是否合并数据
    if merge:
        # 建立临时路径
        tmpdir = ".tmp_merge"
        os.makedirs(tmpdir, exist_ok=True)
        merge_data_path = os.path.join(tmpdir, f"merge_{phase}.npy")

        Lst0 = []
        cumnrows = []
        nrows = 0
        for ipath, path in enumerate(pathLst):

            nrows += np.load(path, mmap_mode="r").shape[0]
            cumnrows.append(nrows)
            Lst0.append(path)
            print(f"{ipath+1}/{npath}", path[:-4] )
            # 数据不够，继续加
            if cumnrows[-1] < batchsize and ipath != npath-1:
                continue
        
            merge_all_data(phase, Lst0, merge_data_path)

            # 更改关键路径
            tester.cfgs['testing']['filename'] = merge_data_path[:-4] 
            # 适当修改批量大小
            tester.cfgs['batch_size_test'] = cumnrows[-1]

            tester.run(session, raw_output)

            # 将合并的结果整理到各个文件夹中
            split_merge_data_result(Lst0, cumnrows, merge_data_path)

            Lst0 = []
            cumnrows = []
            nrows = 0
            os.remove(merge_data_path)


        # 最终删除临时目录
        shutil.rmtree(tmpdir)

    else:
        for ipath, path in enumerate(pathLst):

            print(f"{ipath+1}/{npath}", path[:-4] )
            # 更改关键路径
            tester.cfgs['testing']['filename'] = path[:-4] 

            tester.run(session, raw_output)


    session.close()

    print('Done Testing')



def split_merge_data_result(pathLst:List[str], cumnrows:List[int], merge_data_path:str):
    fuse_picks = np.load(merge_data_path[:-4]+"_fuse_picks.npy")
    if os.path.exists(merge_data_path[:-4]+"_output.npy"):
        total_output = np.load(merge_data_path[:-4]+"_output.npy")
        total_output = total_output.reshape(-1, *total_output.shape[2:])
        idx = 0
        for i, n in enumerate(cumnrows):
            np.save(pathLst[i][:-4]+'_output.npy', total_output[idx:n, ...]) 
            idx = n


    resLst = []
    ipath = 0
    lastnrows = 0
    for res in fuse_picks:
        dataidx, pickidx = map(lambda x:int(x), res.split("_"))
        if dataidx >= cumnrows[ipath]:
            np.save(pathLst[ipath][:-4]+'_fuse_picks.npy', resLst)
            ipath += 1
            resLst = []
            lastnrows = cumnrows[ipath-1]

        resLst.append(f"{dataidx - lastnrows}_{pickidx}")

    if len(resLst) > 0:
        np.save(pathLst[ipath][:-4]+'_fuse_picks.npy',resLst)
        

        
if __name__ == '__main__':
    import argparse 
    import yaml 
    parser = argparse.ArgumentParser()
    parser.add_argument("configpath")
    args = parser.parse_args()
    configpath = args.configpath

    # 读入路径列表
    with open(args.configpath, "r") as f:
        CFGS = yaml.safe_load(f)
        cfgs = CFGS['Picknet']
        pattern = cfgs['pattern']
        picknetDir = CFGS['picknet_data_dir']
        gpuid = cfgs['gpuid']
        phases = cfgs['phases']
        update = cfgs['update']
        raw_output = cfgs['raw_output']

    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)

    merge = cfgs['merge']
    batchsize = cfgs['batchsize']
    for pha in phases:
        pathLst = glob.glob(os.path.join(picknetDir, f"{pattern}/{pha}_input/input.npy"))
        pathLst.sort()

        # 筛选pathLst
        if update:
            Lst0 = []
            for path in pathLst:
               if os.path.exists(path[:-4]+"_fuse_picks.npy") and \
                  (not raw_output or (os.path.exists(path[:-4]+"_output.npy"))):
                    print(path[:-4], ", result exists, skipped.")
                    continue 
               Lst0.append(path)
            pathLst = Lst0

        run(pha, merge, batchsize, int(gpuid), pathLst, update, raw_output)        # [28600:28700]





"""

def merge_all_data2(phase:str, pathLst:List[str], merge_data_path:str):
    '''省内存的写入方法'''
    nrows = 0
    # 打开一个新的 .npy 文件用于写入
    with open(merge_data_path, 'wb') as output_f:
        first_file = True  # 标志是否为第一个文件
        
        for npy_file in tqdm.tqdm(pathLst, desc=f"Merging {phase} Data"):
            # 打开并读取当前文件
            with open(npy_file, 'rb') as input_f:
                # 读取头部信息
                magic, version = npf.read_magic(input_f) # 需要加这一行，不然下一行报错...
                _header = npf.read_array_header_1_0(input_f)
                header = {
                    'shape': _header[0],
                    'fortran_order': _header[1],
                    'descr': npf.dtype_to_descr(_header[2]),
                }
                # 仅处理第一个文件时，获取并保存头部信息
                if first_file:
                    # 将头部写入新的文件
                    npf.write_array_header_1_0(output_f, header)
                    first_file = False
                
                # 读取文件中的数据并写入到目标文件
                # 获取数据的shape和dtype
                shape = header['shape']
                dtype = header['descr']
                nrows += shape[0]
                
                # 将数据内容写入目标文件
                data = np.fromfile(input_f, dtype=dtype)
                data = data.reshape(shape)
                data.tofile(output_f)

        # 写头文件
        output_f.seek(0) # 文件指针回头部
        header['shape'] = (nrows, *header['shape'][1:])
        npf.write_array_header_1_0(output_f, header)
"""