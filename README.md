<a href="https://doi.org/10.5281/zenodo.15559586"><img src="https://zenodo.org/badge/907793418.svg" alt="DOI"></a>

+ Author: Zhu Dengda  
+ Email: zhudengda@mail.iggcas.ac.cn

这是我写的一套使用PickNet拾取地震波形震相的全流程脚本，
包括地震目录下载，地震波形下载，波形截取，PickNet拾取，震相文件导出等，
以及一系列的画图脚本。

所有流程均依靠一个yaml配置文件，且以下所有流程均可以使用该配置文件进行测试运行。

所有流程的python脚本运行格式均为  
```
    python -u [python脚本名] [yaml配置文件]
```

---

## 关于PickNet程序包

以下拾取中要求先安装[PickNet_package](https://github.com/Dengda98/PickNet_package)，
这是我对Picknet源码做了适当修改，使之可以以python包的形式安装，而不再要求在Picknet源码
目录下运行。具体安装及说明详见上述链接。


---

## 脚本以及运行示例 

在当前目录下可看到两个配置文件，`JP_example.yaml`和`YS_example.yaml`，
前者在**处理日本Hinet数据时适用**，后者在**处理从obspy下载的波形时适用**。 
建议将两个配置文件都试着跑一遍程序。 

**运行前请仔细阅读配置文件中各参数的说明！**  

在配置文件中设置好参数，逐一运行脚本即可。  

+ **下载地震目录** --- `01_obspy_catalog_download.py`  
  从美国IRIS官网下载地震目录，基于obspy工具。其中注意`timezone`参数的设置，
  基本只有设置`0`或`9`的可能。


+ **下载地震波形**（选择其一即可）
  - **从日本Hinet下载** --- `02a_Hinet_Download_Queue.py`    
    基于队列的数据结构，使用多账户同步下载  

  - **使用obspy下载其它地区** --- `02b1_obspy_download_waveform.py`  
    下载时对应的台站`.xml`文件会保存在设定的`stationxml_path`目录下。
    下载结束后，再运行`02b2_convert_stations_xml.py`整理得到存有
    台站基本信息的文本文件，保存在`stationtxt_path`。

+ **切数据** --- `03_cut_window.py`   
  根据PickNet的数据输入要求，截取对应波形，以npy格式保存
  在每个事件目录下。对于日本自定义数据，还需要将Hinet自定
  义的波形数据转为sac格式，这会在脚本中自行完成，要求提前
  安装好日本Hinet提供的`win32tools`。

+ **运行picknet** --- `04_run_picknet_zdd.py`  
  使用Picknet做逐个事件的拾取。这里不采取PickNet原程序中将所有数据
  打包成一个大文件的做法。或可以使用`merge`选项加快，这将适当对多个
  事件的数据合并，以发挥`batchsize`的作用。

+ **整理成震相文件** --- `05_write_phase_file.py`  
  将Picknet拾取结果整理成震相文件（自由格式，各元素之间至少
  有一个空格隔开），其中对应震相记录使用台站名做标记

+ **绘制拾取结果** --- `06_plot_phases.py`(可选)   
  可绘制一些事件的拾取情况，每10条波形为一个pdf，要求系统有安装`pdftk`
  以合并多个pdf，例如ubuntu可以通过运行以下命令安装`pdftk`
  ```
    sudo apt install pdftk
  ```

后续可以自己用脚本实现对台站和事件的自由筛选，以及自定义的文件
格式转换。


---

## 其它辅助脚本  
+ `./gmt_scripts/`  
  - `plot_events.gmt`   
    可绘制事件水平分布

+ `./other_pyscripts/`  
  - `07_gen_tomo_station.py`  
    根据空间范围筛选台站  

  - `08_gen_tomo_phase.py`    
    根据空间范围及震级范围筛选事件，并根据台站文件整理成反演输入的震相文件
