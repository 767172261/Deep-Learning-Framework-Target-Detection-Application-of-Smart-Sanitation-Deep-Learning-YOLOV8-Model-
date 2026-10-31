深度学习框架目标检测在智慧化智能环卫的应用 深度学习YOLOV8模型训练垃圾溢满检测数据集 智慧城市、AI环卫监控
@[toc]


数据集描述：


垃圾桶溢满数据集。目标检测的数据集。三类：垃圾、溢满的垃圾桶、未溢满的垃圾桶。
分为训练集、验证集、测试集，训练集2679张图片,验证集502张图片,测试集168张图片。数据集标注准确
标准的yolo格式的数据集，包含类别标签文件，可以直接用于训练模型，无需任何数据处理。
### 🗑️ **垃圾桶溢满检测数据集（Overflowing Trash Bin Detection Dataset）**

| 项目 | 说明 |
|------|------|
| **数据集名称** | 垃圾桶溢满检测数据集 |
| **适用任务** | 目标检测（Object Detection） |
| **适用模型** | YOLO 系列（特别是 YOLOv5、YOLOv8 等） |
| **图像总数** | 3,349 张（训练 + 验证 + 测试） |
| **数据划分** | - 训练集（train）：2,679 张<br> - 验证集（val）：502 张<br> - 测试集（test）：168 张 |
| **图像分辨率** | 未指定（通常为常见监控或手机拍摄分辨率，如 1080×1920 或 720×1280 等） |
| **标注格式** | 标准 **YOLO 格式**（每张图像对应一个 `.txt` 文件）<br>格式：`class_id center_x center_y width height`（归一化到 [0,1]） |
| **类别数量** | 3 类 |
| **类别列表** | 0: 垃圾（garbage）<br>1: 溢满的垃圾桶（overflowing_trash_bin）<br>2: 未溢满的垃圾桶（non_overflowing_trash_bin） |
| **标注质量** | 标注准确，边界框紧密贴合目标 |
| **性能实测** | 使用 **YOLOv8n** 模型在该数据集上实测：<br>**mAP@50 = 91.7%**，表明数据集质量高、模型可学习性强 |
| **文件结构** | ```
TrashOverflow_Dataset/
├── images/
│   ├── train/      # 2679 张
│   ├── val/        # 502 张
│   └── test/       # 168 张
├── labels/
│   ├── train/      # 对应 .txt 标签
│   ├── val/
│   └── test/
└── classes.txt     # 包含三类名称
```|


- `0`: 垃圾（garbage）
- `1`: 溢满的垃圾桶（overflowing_bin）
- `2`: 未溢满的垃圾桶（non_overflowing_bin）

共 **3349 张图像**（训练 2679 + 验证 502 + 测试 168），且已为 **YOLOv8 格式**，可直接训练。

从 **CUDA 驱动安装 → Anaconda → 虚拟环境 → 训练/推理/评估代码** 的 **完整端到端流程**。

---

## ✅ 一、环境搭建（从零开始）

### 1. 确认 CUDA 驱动（GPU 加速）

打开终端或命令行：
```bash
nvidia-smi
```
查看 CUDA 版本（如 12.1、11.8）。若无 GPU，也可使用 CPU，但训练较慢。

> ✅ 推荐：CUDA 11.8 或 12.1

---

### 2. 安装 Anaconda（Python 包管理器）

- 下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- 安装完成后，打开 **Anaconda Prompt**（Windows）或终端（Mac/Linux）

---

### 3. 创建 Python 虚拟环境

```bash
# 创建名为 trash-det 的虚拟环境
conda create -n trash-det python=3.9

# 激活环境
conda activate trash-det
```

---

### 4. 安装必要依赖

```bash
# 安装 PyTorch（以 CUDA 11.8 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 Ultralytics YOLOv8
pip install ultralytics

# 安装 OpenCV（图像处理）
pip install opencv-python

# 其他工具（可选）
pip install matplotlib pillow tqdm
```

✅ 验证安装：
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 应输出 True（有 GPU 时）
```

---

## ✅ 二、数据集结构（YOLO 格式）

### 文件目录结构

```
datasets/
└── trash_overflow/
    ├── images/
    │   ├── train/      # 2679 张
    │   ├── val/        # 502 张
    │   └── test/       # 168 张
    ├── labels/
    │   ├── train/      # 对应 .txt
    │   ├── val/
    │   └── test/
    └── data.yaml
```

> ✅ 数据集已划分好，无需处理，只需按此结构组织文件。

---

### 1. 创建 `data.yaml`

在 `datasets/trash_overflow/` 下创建 `data.yaml`：

```yaml
# data.yaml
train: ../datasets/trash_overflow/images/train
val: ../datasets/trash_overflow/images/val
test: ../datasets/trash_overflow/images/test

nc: 3
names: ['garbage', 'overflowing_bin', 'non_overflowing_bin']
```

---

## ✅ 三、训练代码（调用 YOLOv8 官方预训练模型）

### 📌 `train.py` —— 使用 `yolov8n` 或 `yolov8s` 平衡速度与精度

```python
# train.py
from ultralytics import YOLO
import torch

# 检查 GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 推荐使用 yolov8n（轻量）或 yolov8s（精度更高）
model = YOLO('yolov8n.pt')  # 可选: yolov8s.pt

# 开始训练
results = model.train(
    data='datasets/trash_overflow/data.yaml',  # 数据配置文件
    epochs=150,                                # 训练轮数
    imgsz=640,                                 # 输入尺寸
    batch=32,                                  # 批次大小（根据显存调整）
    name='trash_yolov8n',                      # 实验名称
    device=device,                             # 使用 GPU
    workers=4,                                 # 数据加载线程
    optimizer='AdamW',                         # 收敛稳定
    lr0=0.001,                                 # 初始学习率
    weight_decay=0.0005,
    patience=20,                               # 早停：20轮无提升停止
    project='runs/trash',                      # 保存路径
    amp=True,                                  # 自动混合精度
    close_mosaic=10,                           # 最后10轮关闭 Mosaic
    # --- 数据增强 ---
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.2,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.5,
    degrees=10.0,
)
```

📌 **说明**：
- `yolov8n`：轻量级，适合边缘部署
- `batch=32`：适合 16G 显存；若显存不足，可降为 16 或 8
- `epochs=150`：数据量中等，足够收敛

---

## ✅ 四、推理代码（检测新图像）

### 📌 `predict.py`

```python
# predict.py
from ultralytics import YOLO

# 加载训练好的最佳模型
model = YOLO('runs/trash/trash_yolov8n/weights/best.pt')

# 推理单张图像
results = model.predict(
    source='trash_scene.jpg',                # 输入图像路径
    save=True,                               # 保存带框图像
    conf=0.3,                                # 置信度阈值
    iou=0.45,
    show=False,
    project='runs/predict',
    name='trash_test',
    imgsz=640
)

# 打印结果
for result in results:
    boxes = result.boxes
    print(f"共检测到 {len(boxes)} 个目标：")
    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        cls_name = model.names[cls_id]
        print(f"  {cls_name}: {conf:.3f}")
```

---

## ✅ 五、评估模型性能（在验证集和测试集）

### 📌 `evaluate.py`

```python
# evaluate.py
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/trash/trash_yolov8n/weights/best.pt')

# 在验证集上评估
metrics_val = model.val(
    data='datasets/trash_overflow/data.yaml',
    split='val',
    imgsz=640,
    batch=32,
    conf=0.001,
    iou=0.6,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project='runs/eval',
    name='val'
)

print("=== Validation Set ===")
print(f"mAP@0.5:  {metrics_val.box.map:.4f}")
print(f"mAP@0.5:0.95: {metrics_val.box.map50_95:.4f}")
print(f"Precision: {metrics_val.box.p:.4f}")
print(f"Recall:    {metrics_val.box.r:.4f}")

# 在测试集上评估（YOLOv8 支持 test split）
metrics_test = model.val(
    data='datasets/trash_overflow/data.yaml',
    split='test',
    imgsz=640,
    batch=32,
    conf=0.001,
    iou=0.6,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project='runs/eval',
    name='test'
)

print("\n=== Test Set ===")
print(f"mAP@0.5:  {metrics_test.box.map:.4f}")
print(f"mAP@0.5:0.95: {metrics_test.box.map50_95:.4f}")
print(f"Precision: {metrics_test.box.p:.4f}")
print(f"Recall:    {metrics_test.box.r:.4f}")

# 按类别输出
for i, name in enumerate(model.names):
    print(f"{name}: mAP={metrics_test.box.ap[i]:.4f}")
```

> ⚠️ 注意：YOLOv8 默认 `val`，但支持 `split='test'`（需在 `data.yaml` 中定义 `test:`）

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/655ed4787b7a4f439bedec2822298b0c.png)


## ✅ 六、训练结果可视化

训练完成后，查看以下文件：

- `runs/trash/trash_yolov8n/results.png`：训练曲线（loss, mAP）
- `confusion_matrix.png`：是否混淆“溢满”与“未溢满”
- `PR_curve.png`：各类别 Precision-Recall
- `labels.jpg`：可视化标签是否准确

---



---



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a2519d0239bf468281a22e868020ba8f.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/99927b4ea93b40cbafadabb1c328f108.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/420bbc43a1b34db993456c7218102976.png)
1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/339f364f9be742fc821a757fef42e8fb.png)

1


