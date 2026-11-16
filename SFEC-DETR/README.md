# M7

## 安装

使用 Python 3.8–3.11，在本目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python smoke_test.py
```

本目录自带运行所需的 `ultralytics` 源码，请使用这里的版本。
`requirements.txt` 固定 PyTorch 2.0.0 / torchvision 0.15.1。
使用 GPU 时，请为这组版本安装与运行环境匹配的 PyTorch 构建。

## 数据

修改 `data.yaml` 中的 `path`，使用数据集根目录的绝对路径。
图像及标注按以下格式放置，图像与同名 `.txt` 文件一一对应：

```text
VisDrone/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

每行标注为 `class_id x_center y_center width height`，坐标归一化至 `[0, 1]`，
类别编号从 0 开始。VisDrone 原始标注需先转为此格式。
示例配置包含 10 个类别；使用其他数据集时修改 `names`。
训练会根据数据集的类别数调整检测头，模型默认配置位于 `M7.yaml`。

## 训练

```bash
python train.py --data data.yaml --device 0 --epochs 150 --imgsz 640 --batch 4
```

可通过 `--weights /path/to/checkpoint.pt` 加载本地初始化权重；
不传此参数则从头训练。训练参数可通过 `python train.py --help` 查看和调整。
续训使用 `python train.py --data data.yaml --resume runs/train/M7/weights/last.pt --device 0`。
结果默认保存在 `runs/train/M7`；重名目录会自动递增。

## 验证与推理

```bash
python val.py --weights runs/train/M7/weights/best.pt --data data.yaml --device 0
python predict.py --weights runs/train/M7/weights/best.pt --source /path/to/images --device 0
```

CPU 运行使用 `--device cpu`。验证默认输出 Precision、Recall、mAP50 和 mAP50–95。
需要可视化图表时，在训练或验证命令中添加 `--plots`。
推理结果默认保存在 `runs/predict/M7`。

频域运算默认使用 FP32，训练关闭 AMP，验证及推理关闭 FP16。
数据集和训练权重需另行准备。

## 许可证

AGPL-3.0，见 `LICENSE` 和 `NOTICE`。
