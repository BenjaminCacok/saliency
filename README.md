# 数图第二次大作业

数图第二次大作业，内容为带文本的图像显著性预测，下面将会介绍本次实验的环境配置和文件构成。

------

### 环境配置

```powershell
timm == 0.9.12
torch >= 1.9.1+cu111
easydict == 1.11
opencv-python == 4.8.1
bert-encoder == 0.1.1
```

其中`bert-encoder`的安装参照https://github.com/4AI/bert-encoder

------

### 文件构成

`dataset`文件夹中为csv文件，为训练过程提供文件名数据，这次实验主要用到的是`SALICON`和`SJTU-TIS`两个数据集，csv文件中的X代表原图，Y代表显著性预测图。

`logs`文件夹中为训练和验证阶段的数据和保存的模型，由于该文件夹太大，因此上传到百度网盘中，链接为：https://pan.baidu.com/s/1gMCStZuYv0pZkW96OUhaMQ?pwd=sjtu 

`output`文件夹存储训练结果和测试结果可视化后的图片。

`test`文件夹存储的是模型输出的显著性预测图，一共有10000张，均为`SALICON`数据集中`train`中对应的部分，这些模型预测图上传到百度网盘中，链接为：https://pan.baidu.com/s/1igR-yy31mrLhs9EC-pB_fw?pwd=seie 

`models`文件夹中的文件搭建了实验所需的`GSGNet`网络。

`config.py`为实验参数配置文件，`engine_train.py`为单轮训练代码，`inference.py`为测试代码，`loss.py`为loss计算代码，`sAUC.py`为sAUC计算代码，`utils.py`为数据集定义代码，`train.py`为训练代码。

`ready.ipynb`为划分数据集以及生成csv文件的代码，`visulization.ipynb`为数据可视化的代码。

**注意！**

其中的一些主要参数，例如数据集的位置，存储模型的位置以及不同数据集对应的数据集类的选取是需要手动调整的。

如在`config.py`中

```python
cfg.DATA.SALICON_ROOT = "C:/Users/18336/Documents/SJTUTIS/ours/type3"
cfg.DATA.SALICON_TRAIN = "./dataset/type3_train.csv"
cfg.DATA.SALICON_VAL = "./dataset/type3_val.csv"

cfg.SOLVER.LR = 1e-4
cfg.SOLVER.MIN_LR = 1e-8
cfg.SOLVER.MAX_EPOCH = 30
```

数据集结构应该是（具体应参照给出的csv文件）

dataset
|--type0
   |--images
   |--maps
|--type1
|--type2
|--type3
|--type4

------

### 实验流程

#### 训练

```powershell
python train.py
```

其中

```python
experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_type = "type3_"
experiment_dir = experiment_type + experiment_time
```

可以修改来定义保存log和模型的文件名。

#### 测试

```powershell
python inference.py --path path/to/images --weight_path path/to/model --format format/of/images
python sAUC.py
```

前者用于生成预测的显著图，后者用于计算值。

```python
cv2.imwrite(os.path.join("C:/Users/18336/Documents/GSGNet-main/test/output", filename + ".png"), pred_map)
```

`inference.py`中的这一行代码用于定义生成预测的显著图的位置。

预训练的模型在百度网盘链接中链接：https://pan.baidu.com/s/15PIatX95OEH4rGNzjZU5ag?pwd=eiee 
