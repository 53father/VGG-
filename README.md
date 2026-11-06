# 基于 VGG16 的花卉图像分类

使用 PyTorch 实现 VGG16 迁移学习，对 5 类花卉进行图像分类，最佳验证集准确率达到 **92.0%**。

## 项目简介

本项目在经典花卉数据集 `flower_photos` 上训练 VGG16 网络，通过加载 ImageNet 预训练权重进行迁移学习，实现对以下 5 个类别的识别：

| 索引 | 类别 |
|---|---|
| 0 | daisy（雏菊） |
| 1 | dandelion（蒲公英） |
| 2 | roses（玫瑰） |
| 3 | sunflowers（向日葵） |
| 4 | tulips（郁金香） |

## 实验结果

| 指标 | 数值 |
|---|---|
| 训练集样本 | 3170 张 |
| 验证集样本 | 500 张 |
| 最佳验证准确率 | **92.0%**（第 10、11、12、14 轮） |
| 训练轮数 | 20（计划） |
| 单轮耗时 | 约 15-20 分钟（纯 CPU） |

完整训练日志见 [`训练结果.txt`](训练结果.txt)，预测示例见 [`预测结果.png`](预测结果.png)。

> 注：日志中第 15 轮因手动中断（KeyboardInterrupt）提前结束，模型权重以历史最佳准确率为准保存为 `vgg.pth`。

## 项目结构

```
VGG/
├── model.py            # VGG 网络结构定义（支持 vgg11/13/16/19）
├── train.py            # 训练脚本（含预训练权重加载、最佳模型保存）
├── predict.py          # 批量预测脚本（读取 test/ 目录）
├── split_data.py       # 数据集划分脚本（train / val）
├── class_indices.json  # 类别索引映射，预测时使用
├── 训练结果.txt         # 训练过程日志
├── 预测结果.png         # 预测结果示意图
├── .gitignore
└── README.md
```

> 模型权重（`vgg.pth`、`vgg16.pth`）与数据集目录体积过大，未纳入版本控制。

## 环境依赖

- Python 3.8+
- PyTorch 1.x / 2.x（CPU 或 CUDA 均可）
- torchvision
- Pillow

```bash
pip install torch torchvision pillow
```

## 快速开始

### 1. 准备数据

下载 [flower_photos 数据集](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) 并解压，使目录结构如下：

```
flower_photos/
├── daisy/          # 原始分类文件夹
├── dandelion/
├── roses/
├── sunflowers/
└── tulips/
```

### 2. 划分训练集与验证集

```bash
python split_data.py
```

脚本会为每个类别随机抽取 100 张作为验证集，结果写入 `flower_photos/train` 与 `flower_photos/val`。

### 3. 准备预训练权重（可选但强烈建议）

从 [PyTorch 官方模型库](https://download.pytorch.org/models/vgg16-397923af.pth) 下载 ImageNet 预训练权重，重命名为 `vgg16.pth` 放到项目根目录。

> 未提供该文件时脚本会从零开始训练，收敛很慢且效果较差。

### 4. 训练

```bash
python train.py
```

关键训练配置（见 `train.py`）：

| 项目 | 设置 |
|---|---|
| 输入尺寸 | 224 × 224 |
| 归一化 | mean=0.5, std=0.5 |
| 数据增强 | RandomResizedCrop、RandomHorizontalFlip |
| 损失函数 | CrossEntropyLoss |
| 优化器 | Adam，lr = 0.0002 |
| Batch size | 32 |
| Epochs | 20 |

训练过程中会实时打印 Loss 与验证准确率，并以 `vgg.pth` 保存历史最佳模型。

### 5. 预测

把待预测图片放入 `test/` 文件夹，执行：

```bash
python predict.py
```

输出示例：

```
图片: 123.jpg |  预测:【sunflowers】 | 确信度: 98.72%
```

## 说明与注意事项

- `train.py` 使用当前工作目录（`os.getcwd()`）拼接数据路径，**请在项目根目录下运行**，否则会提示找不到数据集。
- 训练集与验证集的预处理方式不同：训练集带随机增强，验证/预测使用 `Resize((224, 224))`，三者需保持一致以免精度偏差。
- VGG16 参数量约 138M，CPU 训练极慢（本项目单轮约 15-20 分钟）；建议使用 GPU 加速。
- 若换用其他数据集，只需重新运行 `train.py` 即可自动更新 `class_indices.json`。

## License

本项目仅供学习交流使用。
