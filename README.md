# iciar2018-challenge



## 项目背景和必要性

乳腺癌是全球癌症死亡的主要原因之一。用苏木精-伊红(hematoxylin-eosin)染色图像诊断活检组织是非常重要的，但是专家们往往不同意最终的诊断。

计算机辅助诊断系统有助于降低成本并提高该过程的效率。传统的分类方法依赖于基于领域知识为特定问题设计的特征提取方法。为了克服基于特征的方法的许多困难，深度学习方法正成为重要的替代方案。提出了一种使用卷积神经网络（CNN）对苏木精和伊红染色的乳房活检图像进行分类的方法。

我们希望能设计一个**泛化**能力强的模型，在病理图片上的乳腺癌区域而做精确的四分类。模型网络的体系结构旨在检索不同尺度的信息，包括细胞核和整体组织结构。该设计允许将所提出的系统扩展到整个滑动组织学图像。



## 2018 bach challenge数据集

1、**Part A**: 显微镜图：

图像分为四类，正常组织，良性病变，原位癌和浸润性癌；亦可分为癌和非癌两类。根据每个图像中的主要癌症类型，将显微镜图像标记为正常，良性，原位癌或浸润性癌。注释由两位医学专家和图像执行，如果存在分歧则被丢弃。

<table id="table_dataset">
	<tbody>
		<tr>
			<td>
				<img src=".\pics\normal.png" style="width: 180pt; height: 160pt;"></td>
			<td>
				<img src=".\pics\benign.png" style="width: 180pt; height: 160pt;"></td>
			<td>
				<img src=".\pics\in_situ.png" style="width: 180pt; height: 160pt;"></td>
			<td>
				<img src=".\pics\invasive.png" style="width: 180pt; height: 160pt;"></td>
		</tr>
		<tr>
			<td>
				Normal</td>
			<td>
				Benign</td>
			<td>
				<em>in_situ</em> carcinoma</td>
			<td>
				Invasive carcinoma</td>
		</tr>
	</tbody>
</table>


The dataset contains a total of 400 microscopy images, distributed as follows:

- Normal: 100
- Benign: 100
- *in situ* carcinoma: 100
- Invasive carcinoma: 100

2、**Part B** :10张有标注的svs图片，图片大小在500M以上：![A08_thumb](.//pics//A08_thumb.png)

真实场景测试数据集：

3、100张无标签的测试tif图片；

4、10张无标注的测试用svs图片。



### 数据集分析

**Part A**

的数据集是组织经过H&E染色，并在在显微镜下显示的图片，因此往往存在染色差异，亮度差异等区别。而在进行图片分类问题中，模型容易会学习到其颜色特征，和我们的实际应用场景不相符。实际的应用场景下，我们为了得到更好的鲁棒性，得到的模型需要尽可能的去除其颜色对分类的影响。

原tif图片的像素约为2048x1536，由于图像尺寸太大，不能直接输送到模型里，所以需要对其进行采集。数据集图像分析表明，核半径在3至11像素（范围：1.26微米至4.62 微米）。此外，在我们的初步观察中，我们假设大约128×128像素的斑块应足以覆盖相关的组织结构。但是，在我们的数据集中，标签被分配给2040×1536像素的整个图像，这意味着不能保证小区域包含相关的诊断信息。因此我们使用512×512像素的较大图像块以确保可以为每个图像块提供更可靠的标签。最后得到像素为512*512图片。

与其他分类问题相比，使用过的数据集的样本数量较少。因此，网络可能容易过度拟合。对图像进行增广，可以增加数据集的复杂性和维度。通过颜色变换、旋转和镜像进行数据扩充可进一步改善数据集。这是可能的，因为所研究的问题是旋转不变的，即，医生可以从不同方向研究乳腺癌组织学图像而不改变诊断。因此，颜色变换、旋转和镜像允许增加数据集的大小而不会降低其质量。注意，旋转只需进行-180°，-90°， 0°， 90°， 180°的随机旋转，镜像只需进行水平和垂直镜像，不然需要对图片进行像素填充。

**Part B**

对于svs大图，可不断地滑动窗口，把大图分为不同的小图patch，利用Part A得到的模型，对每一个patch进行分类操作，这样，便可以得到大图片的图像分割。



## 工作流程图

```flow
st=>start: Start|past:>https://github.com/xjunming/iciar2018-challenge[blank]
e=>end: End|future:>
op1=>operation: Data preprocessing|past
op2=>operation: train model|current
op3=>operation: predict Data|past
op1_sub1=>subroutine: Get patch & stain transformation|invalid

io=>inputoutput: catch something...|future

st->op1->op2->op3->e


```



## 方法

### 数据预处理

数据预处理在深度学习中尤为重要，数据集往往是决定模型上限的，而模型既是不停地靠近模型。

数据预处主要有两部分，一是对原数据集的划分，二是采集图片并进行染色增强。在数据集的划分上，为了提高模型的泛化能力，把颜色鲜艳的作为测试集，若模型在val_loss或val_acc表现较好，一定程度上可以说明模型具有一定的泛化能力。在原图片切割方面，根据数据集的分析可知，512*512大小的图片可以很好地表现特征，另一方面，为了降低颜色对cnn模型的影响，进一步提高泛化能力，因此再对切割后的图片进行染色变换，变换的幅度较大，变换通道为h, s, v三个通道。

- 数据集划分：
     统计400张tif图的h, s, v的均值及其方差，用其作为特征进行k-means聚类，最后把原数据集分为了两个，一个颜色较为鲜艳高亮度，另一个较为低亮度低饱和度，将颜色较为鲜艳的分为测试集，剩余的图片则为训练集。划分的数据集分布如下表所示：
|       | **Normal** | **Benign** | **Insitu** | **Invasive** |
| ----- | ---------- | ---------- | ---------- | ------------ |
| Train | 61         | 84         | 64         | 68           |
| Test  | 39         | 16         | 36         | 32           |

- 采集图片

  对像素为2048x1536的tif图片以512*512的patch进行采集。

  在采集的同时，需要增强数据集。对H，S，E通道上进行随机颜色转换，才进行一个图片保存操作，在测试集上不需对其进行颜色变换。

  在训练模型时，进行随机旋转和镜像操作，以增加训练集的随机性。旋转只用$\pi/2$的整数倍，镜像有水平镜像和垂直镜像。

- 采集图片相关参数

| 数据集 | 1        |
| ---- | -------- |
| step | 512 |
| patch size | 512 |
| scale range | 0.6 |
| stain channels | h, s, v |
| aug num | 2 |

注：具体参数定义为

step: 截图步长； patch size：截图大小； scale range：颜色变换范围为[1.08-scale, 1.08+scale]； stain channels：染色通道； aug num：颜色增强数量。



### Data Generator

该生成器是自动获取路径中的图片，以rgb的格式读取其数组，然后进行数据标准化，再进行旋转及镜像等变换。

#### 数据标准化

由于我们导入的是imageNet，因此我们在输入的image上除以255后，还要额外进行去均值和标准差。在ImageNet数据集里，rgb的均值和标准差分别为mean = [0.485, 0.456, 0.406]， std = [0.229, 0.224, 0.225]。具体参考[波哥分享link1](http://192.168.3.126/Kwong/totem_begin/tree/master/tricks_in_processing_and_training), [官网link2](https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/keras/applications/imagenet_utils.py) 。在我的*Generator.py*里，原来的顺序是先进行自定义的标准化，再进行rescale，因此我修改了顺序，把自定义的标准化在最后再进行。

#### 旋转及镜像

为了进一步防止过拟合，增加训练集的随机性，我们需要进行旋转及镜像操作，而这些操作并不会影响医生的判断结果。考虑到旋转角度为$\pi /2$才不会对图片进行填充，因此我们允许的旋转角度为$\{ -\pi, -\pi/2, \pi/2, \pi/2\}​$ （注意，旋转的角度我是修改了源码*Generator.py*），镜像为水平镜像和垂直镜像。



### 模型

ResNet已经在很多分类问题上得到很好的应用，因此我们使用ResNet进行训练模型。

在导入ResNet50预训练的框架基础上，我们加了一个维度为32的隐藏层，和以4分类的结果输出。



#### 模型参数

| 模型 | ResNet50 |
| ---- | -------- |
| epoch | 50 |
| lr | 0.0001 |
| batch size | 128                                           |
| optimizers  | Adam   |
| loss fun | weight_categorical_crossentropy(自定义的loss) |
| resize | 224  |

注：具体参数定义为

模型：主要的模型框架， epoch：训练的回合， lr：学习率， batch size：每次计算loss的batch，optimizers：优化器，loss fun：选用的损失函数， resize：最终输入到的模型的图片大小。

* 为什么要让lr为0.0001，batch size为128：经过我多次实验，如果设置为大于0.0001，batch size小于128，训练了7、8个epoch后，loss依然停留在大于1，而在之后loss出现波动情况。这个要感谢世伟，这个参数是世伟指导的。
* 为什么要自定义loss funtion：为了提高Invasive(乳腺癌最坏的情况)的召回率，尽可能地检测到Invasive，波哥写了一个自定义的loss funtion。



## 结果分析

### 混淆矩阵(confusion matrix)(测试集上)

| test data<br/>(model ResNet50) |          |          |        |        |        |
| ---------- | -------- | -------- | ------ | ------ | ------ |
|            |          | Predict  |        |        |        |
|            |          | Invasive | Insitu | Benign | Normal |
| TRUE       | Invasive |       |     |     |     |
|            | Insitu     |       |       |     |      |
|            | Benign     |        |        |     |      |
|            | Normal     |       |       |     |     |

### 预测精确率

| 实验 | 1 |
| ---- | -------- |
| 模型 | ResNet50 |
| accuracy |   |
| precision |   |
| recall |      |
| F1-score |       |
| precision(Invasive) |     |
| recall(Invasive) |   |
| F1-score(Invasive) |  |

注：precision(Invasive)、recall(Invasive)、F1-score(Invasive)的计算方式就是分为(Invasive，非 Invasive)画的混淆矩阵。



## 总结





## 附录

(10张svs预测结果)
