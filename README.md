# iciar2018-challenge

方法介绍与[代码说明](#code)


### 目录结构

```
iciar2018-challenge
│   README.md
│   data_preprocessing.py
│   model.py
│   cnn_predict_svs.py
│
└───utils
│   │   generators.py
│   │   class4_preview.py
│   │   get_colormap_img.py
│   
└───demo
│   data_processing_demo.py
│   file022.txt
│   
└───data
│   └───train
│   │   Normal
│   │   ...
│   └───test
│   │   Normal
│   │   ...
│   
└───pics
│   A08_thumb.png
│   ...
```



### how to run

* 下载整个项目[代码](https://github.com/xjunming/iciar2018-challenge)

* 在Terminal终端运行`virtualenv venv`，创建一个虚拟环境
* `source venv/bin/activate`，激活虚拟环境
* `pip install -r requirement.txt`，安装相关依赖包
* `python data_preprocessing.py`，进行数据预处理
* `python model.py`，训练模型(在1080ti的速度大概1小时/epoch)
* `python cnn_predict_svs.py`，对大图进行预测

注意，运行了`data_preprocessing.py`之后要记得把文件夹名字变成0、1、2、3，使其分别对应Normal、Benign、In Situ、Invasive。

## 附录

(10张svs预测结果)![A03_resnet50_colormap](./pics/coloermap/A03_resnet50_colormap.png)

![A04_resnet50_colormap](./pics/coloermap/A04_resnet50_colormap.png)

![A05_resnet50_colormap](./pics/coloermap/A05_resnet50_colormap.png)

![A06_resnet50_colormap](./pics/coloermap/A06_resnet50_colormap.png)

![A07_resnet50_colormap](./pics/coloermap/A07_resnet50_colormap.png)

![A08_resnet50_colormap](./pics/coloermap/A08_resnet50_colormap.png)

![A09_resnet50_colormap](./pics/coloermap/A09_resnet50_colormap.png)

![A10_resnet50_colormap](./pics/coloermap/A10_resnet50_colormap.png)

![A01_resnet50_colormap](./pics/coloermap/A01_resnet50_colormap.png)

![A02_resnet50_colormap](./pics/coloermap/A02_resnet50_colormap.png)
