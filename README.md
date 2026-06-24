# NAFNet — 图像去噪 (Nonlinear Activation-Free Network)

> **NAFNet (Nonlinear Activation-Free Network)** 图像去噪实现。由旷视研究院（Megvii Research）提出，发表于 ECCV 2022。
> 核心创新：移除传统激活函数（ReLU/GELU），提出 SimpleGate 和 Simplified Channel Attention，在 SIDD 数据集达到 SOTA。

---

## 📋 成果 | Results

| 数据集 | 任务 | PSNR | SSIM |
|--------|------|------|------|
| SIDD | 图像去噪 | **40.30 dB** | **0.963** |
| GoPro | 图像去模糊 | **33.58 dB** | **0.972** |

---

## 💡 核心创新 | Key Innovations

### 1. SimpleGate — 零参数非线性变换

替代传统激活函数，将特征沿通道维度一分为二，逐元素相乘。**零参数量**，计算效率极高。

```
SimpleGate(X) = X[:, :C/2] ⊙ X[:, C/2:]
```

### 2. Simplified Channel Attention (SCA)

相比传统 SE 模块：移除两层 MLP → 单层 1×1 卷积，移除激活函数 → 直接 Sigmoid。**参数量减少 75%**。

### 3. NAFBlock 设计

```
LayerNorm → Conv1×1 → Conv3×3 → SimpleGate → SCA → Conv3×3 → Conv1×1 + Learnable β/γ
```

### 4. 模型变体 | Model Variants

| 模型 | 参数量 | NAFBlocks | 通道数 | FLOPs |
|------|--------|-----------|--------|-------|
| NAFNet-S | 2.1M | [2,2,4,2,2] | 32 | 8G |
| NAFNet-B | 18.5M | [4,8,20,8,4] | 64 | 68G |
| NAFNet-W32 | 5.3M | [4,8,20,8,4] | 32 | 17G |

---

## 🚀 快速开始 | Quick Start

```bash
# 训练
cd basicsr
python train.py -opt ../options/train/SIDD/NAFNet-SIDD-width32.yml

# 单图去噪
python demo.py --input noisy.png --output clean.png

# 批量推理
python nafinfer.py --input_dir ./datasets/val --output_dir ./datasets/clean
```

---

## 📁 项目结构 | Project Structure

```
├── basicsr/
│   ├── models/archs/NAFNet_arch.py   # 核心 NAFNet 架构
│   ├── train.py / demo.py / test.py
│   ├── data/                         # 数据集加载器
│   └── metrics/                      # PSNR, SSIM, NIQE, FID
├── options/                          # YAML 训练配置
├── nafinfer.py                       # 独立推理脚本
└── README.md
```

---

## 📄 引用 | Citation

```bibtex
@inproceedings{chen2022nafnet,
  title={NAFNet: Nonlinear Activation-Free Network for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  booktitle={ECCV}, year={2022}
}
```

## 📄 许可证 | License

MIT
