#                            **人工智能原理实验（综合应用）- 个人作业**

**学号：20354140**            **— 姓名：杨庆城**            **— 时间：2022.10.28**            **— 主题：目标检测 TinySSD**

**注：作业要求中的第1-5部分均在本ReadMe.md文件。**



## 0. 初始准备

#### 【文件解压】

**注：从Github上下载文件则需要先执行文件解压，直接从邮件下载20354140-杨庆城-TinySSD个人作业.zip 压缩则直接跳过此步。**

​        下载后，解压所有压缩包（共4个，包括 **detection.zip、pretrained.zip、test_result.zip、training_loss_change.zip** ）至 **./** 路径，也即**与所有代码文件同一级目录的本级路径**下。

​        解压完后文件、文件夹排列如下图所示：

<img src="https://raw.githubusercontent.com/Tinchill/YangQingcheng_Homework/main/readme_figures/file_structure.png" alt="file_structure" width="285" />

#### 【检查配置】   

​        检查是否配齐本文件第2部分-【环境配置】中所述版本的软件包。



## 1. 文件说明

#### 【代码文件】

**create_train.py**			准备训练集         

​        将 **./detection/background** 中的JPG背景与目标检测对象 **./detection/target** 中的JPG文件结合。

​        运行此代码后，形成 **./detection/sysu_train/images **中的训练集图片（**./detection/sysu_train/label.csv** 记录图片的标签）。

​        对目标进行适当旋转、透明化等操作，可以得到 **./detection/sysu_train_merged/images** 下的训练集图片，具体代码同本 .py 文件类似。

**load_data.py**			加载数据

​        定义Dataset类，加载训练集数据。

**model.py**				模型定义

​        从各模块开始，一级级向上定义TinySSD模型。

**train.py**					网络训练

​        训练模型，每隔10个epoch保存一次模型至./pretrained路径下。

**test.py**					网络测试

​        定义得到并处理边界框的各类函数，为可视化锚框作准备。

**visualize.py**				可视化结果

​        预测边界框将其可视化。

**test_port.py**				测试接口

​        用预训练模型中误差最小的一个，将对./detection/test下图片所预测的目标边界框、置信度等结果可视化。



#### 【文件夹】

**./detection**

​        目标检测任务的数据集文件夹。

**./pretrained**

​        保存预训练的模型参数。

​        模型命名规则：net_ + 训练所用优化器名称 + _lr= + 训练步长 + _ + 保存模型参数时的epoch数 + .pkl。

​        例：'net_SGD_lr=0.2_40.pkl' 代表用SGD迭代优化器，在0.2的训练步长下，训练到epoch=40时所保存的.pkl文件。

**./training_loss_change**

​        保存训练过程中损失函数值的变化曲线图。

​        每隔5个epoch，记录训练过程中 bbox mae（L1范数损失） 和 class error（分类损失，使用交叉熵损失函数衡量）的变化。损失函数曲线图的命名规则：优化器名称 + _ + lr= + 训练步长 + _ + 最大训练轮数 + _ loss_change.jpg。

​        例：'SGD_lr=0.2_50_loss_change.jpg'代表用SGD迭代优化器，在0.2的训练步长下，最大训练到epoch=50时，每隔5个epoch的损失函数值变化情况。

**./test_result**

​        保存对测试图片的锚框预测结果图。

​        图片命名格式：net_ + 训练所用优化器名称 + _lr= + 训练步长 + _ + 保存模型参数时的epoch数 + _ + test + _ 测试图片序号（从0开始） + _result.jpg。

​        例：'net_SGD_lr=0.2_40_test_1_result.jpg'，代表使用SGD迭代优化器在0.2的训练步长下，训练到epoch时所保存的.pkl，来预测序号为1的测试图片所得到的锚框预测结果。

**./readme_figures**

​       *保存本 README 文件所用的图片。*



#### 【其它】


## 2. 环境配置和训练流程

#### 【环境配置】

​        所用到的软件库以及对应版本如下：

​        matplotlib == 3.5.1

​        numpy == 1.21.5

​        opencv-python == 4.5.5.64

​        pandas == 1.3.0

​        pytorch == 1.10.2

​        torchvision == 0.11.3



#### 【训练流程】

​        整体训练框架如图所示：

![figure_1](https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_1.png)

**训练流程：**

​        打开代码文件 **train.py**。

**（1） 调整信息和超参数。**

​        ○ 根据情况修改训练所用设备变量 **_device** 为 **'cpu ' **或 **'cuda'**。（第17行）

```
_device = 'cpu'            # 若使用 GPU, 则修改为 'cuda'
```

​        ○ 根据需要修改迭代步长变量  **learning_rate** 。（第167行）

```
learning_rate = 0.2
```

​        ○ 根据需要调整优化器 **trainer** 。

​        若使用 **SGD优化器** 进行迭代优化，则 **trainer** 定义如下：  (第168行)

```
trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-4)
```

​        如果要使用其它优化器，比如：使用 **Adam优化器** 进行迭代优化，则 **trainer** 定义如下：（替换第168行为以下内容）

```
trainer = torch.optim.Adam(net.parameters(), lr=0.2, weight_dacay=5e-4)
```

​        ○ 根据需要调整训练轮数变量 **num_epochs** ： （第171行）

```
num_epochs = 51
```

**（2）训练和保存模型**

​        运行 **train.py** ，在 **./pretrained** 文件夹下查看所保存的.pkl文件。

​        在本实验中，使用了 **SGD** 优化器，设置**学习步长**为 **0.2**，设置**最大训练轮数**为 **50**。

​        同时，将**每隔 5 个 epoch** 的损失函数值予以记录并保存其变化曲线至 **./training_loss_change** 路径下：

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_2.jpg" alt="figure_2" width="575" />



## 3. 简易测试及预训练模型

#### 【简易测试】

​          代码中默认选用的预训练模型为： **./pretrained/net_SGD_lr=0.2_50.pkl**

​          运行 **test_port.py** ，即可查看测试效果。同时可以在 **./test_result** 中查看可视化结果。

#### 【预训练模型】

​          在 **./pretrained** 路径下，可以查看各预训练模型的.pkl文件。

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_3.png" alt="figure_3" width="675" />

​        如果要使用其它预训练模型进行测试，则打开 **test_port.py** 文件，修改 **model_pkl** 变量为所选择的预训练.pkl文件的名称（注意不要带.pkl后缀）。

​        例：如果希望选择下图红框框住的预训练模型文件

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_4.png" alt="figure_4" width="675" />

​        则修改 **test_port.py** 中的 **model_pkl** 变量：（第18行）

```
model_pkl = 'net_SGD_lr=0.2_30'
```



## 4. 检测效果

####  【原测试图】

（见 **./detection/test** 文件夹）

​        ○ 测试图片 0

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_5.jpg" alt="figure_5" width=350/>

​        ○ 测试图片 1

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_6.jpg" alt="figure_6" width=350/>

#### 【效果可视化】

（见 **./test_result** 文件夹）

​        使用  **./pretrained/net_SGD_lr=0.2_50.pkl** 的检测结果：

​        ● 测试图片 0 的检测效果

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_7.jpg" alt="figure_7" width=635/>

​        ● 测试图片 1 的检测效果

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_8.jpg" alt="figure_8" width=635/>



## 5. 效果提升

​        **○ 原训练方案A：**使用 **SGD** 迭代优化器，学习步长设置为 **0.2**。

​        **○ 提升方案B：**仍使用 **SGD** 迭代优化器，将学习步长调整为 **0.01**；

​        **○ 提升方案C：**使用 **Adagrad** 迭代优化器，设置学习步长为 **0.01**，权重衰减为 **0.0005**。

​        训练至 **epoch=50**，查看并对比结果。

#### 【损失函数指标对比】

#####                                                                                                            □ bbox mae    /×10^(-3)

|       | epoch=0 | epoch=10 | epoch=20 | epoch=30 |  epoch=40  |  epoch=50  |
| :---: | :-----: | :------: | :------: | :------: | :--------: | :--------: |
| **A** |  4.68   |   3.61   |   2.82   |   2.38   |    2.31    |  **2.17**  |
| **B** |  3.94   |   1.47   |   1.44   |   1.39   |    1.36    | **1.34 ↓** |
| **C** |  4.94   |   2.16   |   1.77   |   2.15   | **1.36 ↓** |    3.06    |

#####                                                                                                            □ class error    /×10^(-3)

|       | epoch=0 | epoch=10 | epoch=20 | epoch=30 |  epoch=40  |  epoch=50  |
| :---: | :-----: | :------: | :------: | :------: | :--------: | :--------: |
| **A** |  10.86  |   2.55   |   2.24   |   2.03   |    1.96    |  **1.78**  |
| **B** |  10.80  |   1.42   |   1.29   |   1.16   |    1.14    | **1.07 ↓** |
| **C** |  12.70  |   2.57   |   1.97   |   2.01   | **1.35 ↓** |    3.52    |

#### 【可视化检测结果对比】

**注：使用训练过程中损失函数值最小时保存的.pkl**

○ 原训练方案使用预训练模型 **./pretrained/net_SGD_lr=0.2_50.pkl**，可视化结果见本 README 文件第4部分。

○ 对于提升方案B，采用预训练模型 **./pretrained/net_Adam_lr=0.001_50.pkl**。

​        ● 测试图片 0 的检测效果

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_11.jpg" alt="figure_11" width=635/>

​        ● 测试图片 1 的检测效果

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_12.jpg" alt="figure_12" width=635/>

○ 对于提升方案C，采用预训练模型 **./pretrained/net_Adagrad_lr=0.01_40.pkl**。

​        ● 测试图片 0 的检测效果

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_9.jpg" alt="figure_9" width=635/>

​        ● 测试图片 1 的检测效果

<img src="https://media.githubusercontent.com/media/Tinchill/YangQingcheng_Homework/main/readme_figures/figure_10.jpg" alt="figure_10" width=635/>

​        可以看出，采用B、C方案训练得到的模型，其预测结果的锚框范围比原训练方案得到的最优预训练模型更贴近目标。
