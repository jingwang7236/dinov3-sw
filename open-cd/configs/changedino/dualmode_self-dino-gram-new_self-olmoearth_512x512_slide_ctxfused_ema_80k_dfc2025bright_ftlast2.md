# 前光后 SAR 异构变化检测 — 技术文档

> 配置文件：`dualmode_self-dino-gram-new_self-olmoearth_512x512_slide_ctxfused_ema_80k_dfc2025bright_ftlast2.py`
> 数据集：DFC2025 BRIGHT　|　任务：4 类建筑物损毁分级（前光学 + 后 SAR）

---

## 1. 任务概述

### 1.1 任务定义
DFC2025 BRIGHT 是一个**异构双时相建筑物损毁分级**数据集：
- **灾前（pre-event）**：光学图像（3 通道 RGB）
- **灾后（post-event）**：SAR 图像（单通道，自动复制为 3 通道）
- **标签**：4 值图 `0/1/2/3`，分别表示
  - `0 = background`（背景/未损毁）
  - `1 = intact`（建筑完好）
  - `2 = damaged`（受损）
  - `3 = destroyed`（毁坏）

这与传统"同源双时相（光学+光学 / SAR+SAR）变化检测"有本质区别：两时相传感器**物理模态不同**（光学 vs 微波），不能用对称孪生网络直接相减，必须分别用各自模态的强预训练 backbone 编码，再在特征空间做异构融合。

### 1.2 本配置的目标
在冻结两个 backbone 的基础上，通过 **两阶段训练 + FreezeScheduleHook** 在后半段解冻双 backbone 最后 2 层做轻量微调，提升域适配能力，弥补"全冻结导致各类 IoU 普遍偏低（欠拟合）"的问题。

---

## 2. 整体网络架构

### 2.1 数据流总览

```
输入: 灾前光学 (B,3,H,W)  +  灾后SAR (B,3,H,W)
        │                          │
        │   (拼接成 B,6,H,W 输入)   │
        ▼                          ▼
┌──────────────────┐      ┌──────────────────┐
│ backbone_opt      │      │ backbone_sar      │
│ DINOv3Adapter     │      │ OlmoEarthSAR      │
│ (ViT-L, 冻结)     │      │ (ViT-B, 冻结)     │
└──────────────────┘      └──────────────────┘
        │ [p2,p3,p4,p5]          │ [p2,p3,p4,p5]
        │  (128ch, /4 /8 /16 /32) │  (128ch, 同尺度)
        └──────────┬─────────────┘
                   ▼
   ┌────────────────────────────────────────┐
   │  ChangeDinoHybridCrossAttnDecoder       │
   │  (差异分支 + 上下文分支, 门控融合)        │
   │  → 多尺度 logits [p2,p3,p4,p5]          │
   └────────────────────────────────────────┘
                   │
        训练: 多尺度 Focal+Dice+Lovász
        推理: 多尺度融合 + TTA + 滑窗
                   ▼
            4 类分割图 (H×W)
```

### 2.2 关键设计思想
1. **异构双流**：光学走 DINOv3（强自然图像表征），SAR 走 OlmoEarth（遥感 SAR 专用预训练），各取所长。
2. **非对称但输出对齐**：两路输出尺度与通道完全一致（4 个尺度 × 128ch），便于在解码器做逐尺度融合。
3. **冻结预训练 + 轻量适配层**：主干冻结保留强表征，仅训练 adapter / 投影 / 解码器；后半段选择性解冻。

---

## 3. 各模块详解

### 3.1 数据预处理 — `DualInputSegDataPreProcessor`

```python
mean = [73.17, 82.67, 86.77,  56.38, 56.38, 56.38]   # 前3=光学, 后3=SAR
std  = [53.27, 58.12, 67.28,  44.49, 44.49, 44.49]
bgr_to_rgb=True, size_divisor=32, seg_pad_val=255
```

- **6 通道均值/方差**：光学（前 3）与 SAR（后 3）各用独立的统计量，因为两者数值分布差异巨大（光学反射率 vs SAR 后向散射强度）。
- `size_divisor=32`：保证空间尺寸是 32 的倍数，适配 ViT patch 化与多尺度下采样（/32）。
- `seg_pad_val=255`：标签填充用 255，配合 `ignore_index=255` 在损失中忽略。

### 3.2 数据集 — `DFC2025BRIGHTDataset`

| 项 | 说明 |
|----|------|
| 目录 | `pre-event/`(光学) · `post-event/`(SAR) · `target/`(标签) |
| `binarize_label=False` | 保留 4 类损毁分级（不合并为二分类） |
| 标签合并机制 | 通过 `label_map` 而非阈值法，避免 `<128→0` 误判 |
| 输出 | `data_list[i].img_path = [光学路径, SAR路径]`，由 `MultiImg*` 流水线成对加载 |

> 注意：SAR 本是单通道，`MultiImgLoadImageFromFile` 默认 `color_type='color'` 会将其复制为 3 通道，与光学侧对齐后再拼成 6 通道。

### 3.3 光学编码器 — `DINOv3AdapterBackbone`

**结构**：DINOv3 ViT-L16（24 层，embed_dim=1024） + ViT-Adapter + 4 个投影 SepAdapterBlock。

**核心组件**：
1. **ViT-L 主干**：完全自研预训练权重（`self_trained`），处理 512×512 输入 → patch16 → 32×32 token 网格。使用 RoPE 位置编码、memory-efficient attention（不物化注意力矩阵，省显存）。
2. **ViT-Adapter**（`DINOv3_Adapter`）：把纯 ViT 的单尺度语义特征转换为密集多尺度特征，关键在 **多尺度可变形注意力（MSDeformAttn）**：
   - `spm`（Semantic Prediction Module）：从输入先产生粗多尺度语义先验 c1~c4。
   - `interactions`：在 ViT 第 `[5,11,17,23]` 层注入交互，用 deformable attn 把 ViT 浅层细节"喂回"给语义先验，逐步精化。
   - 最终输出 4 尺度特征，再经 4 个 `SepAdapterBlock` 把 1024→128 通道。
3. **冻结模式**：`freeze_mode='frozen'`（ViT 主干 `requires_grad=False`），adapter/投影层可训练。

**输出**：`[p2, p3, p4, p5]`，通道 128，空间 /4 /8 /16 /32。

### 3.4 SAR 编码器 — `OlmoEarthSAREncoder`

**结构**：OlmoEarth-base ViT（12 层，embed_dim=768） + Composite Encodings + 多尺度特征提取器 + 输入通道适配器。

**核心组件**：
1. **ViT 主干**：遥感 SAR 专用预训练（`olmoearth10m_base`），patch8，`native_size=384`（输入先下采样到 384×384 再过 ViT，避免 token 数爆炸）。注意力用 `F.scaled_dot_product_attention`（SDPA，显存友好）。
2. **Composite Encodings**：4 段编码拼接加到 token 上，覆盖遥感特有维度：
   - `channel embed`：模态通道编码
   - `time encoding`：时相编码（单时相）
   - `month embed`：月份编码（季节/物候）
   - `spatial sincos`：按地面分辨率（10m）缩放的二维位置编码
3. **输入通道适配器（`input_adapter`）**：BRIGHT SAR 是 3 通道而预训练是 2 通道，用 1×1 可训练 conv 适配。
4. **MultiScaleFeatureExtractor**：从 ViT 输出 token 网格用 1×1 conv + avg_pool 提取 `/4 /8 /16 /32` 四尺度，均投影到 128ch。

**新增方法 `set_freeze_mode`**（本配置新增）：支持 `frozen` / `full_finetune` / `unfreeze_last_n`，签名与 DINOv3 侧一致，供 `FreezeScheduleHook` 调用。

### 3.5 解码器 — `ChangeDinoHybridCrossAttnDecoder`（核心）

继承链：`ChangeDinoDecoder` → `ChangeDinoCrossAttnDecoder` → **`ChangeDinoHybridCrossAttnDecoder`**。

#### (a) 基类 `ChangeDinoDecoder`：自顶向下 FPN 融合
- `FuseGated` 门控融合：`fused = x2 + sigmoid(gate(cat(x1,x2))) * x1`，再做 3×3 conv+BN+SiLU。用于 p5→p4→p3→p2 自顶向下传递。
- 每尺度 `TransformerBlock`（CDA/OCDA 注意力）精化特征。
- 4 个尺度各有预测头 `p2/p3/p4/p5_head`（1×1 conv → 4 类）。

#### (b) `CrossAttnFusion`：双向交叉注意力（替代朴素差分）
纯差分 `|t1−t2|` 信号弱。本模块在窗口内做**对称交叉注意力并取差**：
```
out = Proj( XAttn(t1→t2) − XAttn(t2→t1) )
```
显式保留"变化"语义。窗口注意力（`window_size=8`）兼顾效率与局部性。

#### (c) Hybrid 解码器：差异分支 + 上下文分支（★本配置核心创新）
**动机**：纯差异分支对**未变化类**（BRIGHT 的 `intact` 完好建筑）信号极弱，导致其 IoU 显著偏低。

**做法**：每个尺度同时计算两路：
- **change 分支**：`CrossAttnFusion(t1, t2)` → 差异特征
- **context 分支**：`ContextFuse(t1, t2)` = 把双时相特征通道拼接后投影（保留联合绝对外观，让"未变化"区域也有可分辨信号）

两路通过**可学习门控**加权相加：
```
diff = diff + sigmoid(gate) * context_weight * context_feat
gate_init = -1.0   # 初始 sigmoid≈0.27, 训练自适应
```
这样既保持变化检测能力，又提升未变化类可分性，直接拉高 mIoU。

#### (d) 损失计算
对 p2/p3/p4/p5 四个尺度分别计算并加权求和：
- **Focal Loss**（`gamma=4.0`，类别加权 `alpha=[0.5,1.0,0.75,0.75]`）→ 抑制易类、增强稀有类
- **Dice Loss**（各尺度权重 0.5）→ 缓解类别不平衡
- **Lovász Softmax Loss**（`weight=1.0`，深层尺度权重递减）→ 直接优化 IoU 的凸代理

### 3.6 检测器包装 — `DualModeBranchEncoderDecoder`
- 继承 `ChangeDinoEncoderDecoder`，光学 backbone 作为 `self.backbone`，SAR 作为 `self.backbone_sar`。
- `extract_feat`：把 6 通道输入按通道切成光/SAR 两路分别编码。
- 复用父类的 loss / refiner / 后处理；额外实现：
  - **滑窗推理**（`mode='slide'`）
  - **TTA**（`tta_flips`）
  - **多尺度 logits 融合**（`ms_inference`）

---

## 4. 核心 Trick 详解

### Trick ① 两阶段冻结微调（FreezeScheduleHook）★本配置主线

**问题**：双 backbone 全冻结 → 模型有效容量受限 → 4 类 IoU 普遍偏低（欠拟合/域适配不足）。

**方案**：两个 `FreezeScheduleHook` 实例分别驱动两个 backbone：

| 阶段 | iter 区间 | 光学 ViT | SAR ViT | 学习率 |
|------|-----------|----------|---------|--------|
| 阶段1 | 0 ~ 40k | 全冻结 | 全冻结 | base lr=5e-4 |
| 阶段2 | 40k ~ 80k | 解冻最后 2 层 +norm | 解冻最后 2 层 +norm | 重新 warmup |

**机制要点**：
- Hook 通过**通用属性路径遍历**（`getattr`）定位 backbone：`backbone_attr='backbone'`（光学）、`backbone_attr='backbone_sar'`（SAR），不依赖硬编码分支。
- Hook 在 `iter=40000` 调用 `backbone.set_freeze_mode('unfreeze_last_n', 2)`，把最后 2 个 transformer block + 最终 norm 的 `requires_grad=True`。
- **无需重建 optimizer**：PyTorch optimizer 在初始化时已包含所有参数，冻结参数 `grad=None` 被跳过，解冻后梯度自动计算更新。
- **配合 `paramwise_cfg`**：backbone 的 ViT blocks 用 `lr_mult=0.1`（5e-5），避免大 lr 破坏预训练表征。

**两段式 scheduler**（防止解冻瞬间发散）：
```
阶段1: 0-5k warmup → 5k-40k cosine 到 5e-5
阶段2: 40k-43k 重新 warmup → 43k-80k cosine 到 1e-6
```

### Trick ② EMA 权重平滑（EMAHook）
- `momentum=2e-4`（影子权重 = 0.9998×影子 + 0.0002×当前权重）。
- `update_buffers=True`：连 BN running stats 一起平滑。
- val/test 自动换入 EMA 权重评估，通常均匀提升各类 IoU +0.3~1.0。
- `priority='LOWEST'`：保证在其它 hook 之后执行。

### Trick ③ 多尺度推理融合（ms_inference）
推理时把 p2/p3/p4/p5 各预测头 logits 上采样到同尺寸后加权求和：
```python
ms_inference_weights = [0.5, 0.3, 0.1, 0.1]  # p2 主导, 高层补充语义
```
- 改善边界与碎小目标。
- 仅影响推理，训练不变。

### Trick ④ TTA（Test-Time Augmentation）
```python
tta_flips = [3, 2]   # 水平(dim=3) + 垂直(dim=2)翻转
```
对每个翻转版本推理后翻转回来取均值，稳拿 +0.2~0.5 mIoU，几乎零成本。

### Trick ⑤ 滑窗推理（slide inference）
```python
test_cfg = dict(mode='slide', crop_size=(512,512), stride=(256,256))
```
大图按 512×512 窗口、步长 256 滑动，重叠区域取均值。避免整图 resize 破坏分辨率。

### Trick ⑥ 损失组合（直接优化 mIoU + 稀有类）
| 损失 | 作用 | 权重 |
|------|------|------|
| Focal | 难例挖掘 + 类别加权 | 各尺度 1.0 |
| Dice | 缓解不平衡 | 各尺度 0.5 |
| Lovász | IoU 凸代理 | p2=1.0, p3=0.5, p4=0.3, p5=0.1 |

类别加权 `focal_alpha=[0.5,1.0,0.75,0.75]`：背景降权，完好建筑与损毁类加权。

### Trick ⑦ 学习率分组（paramwise_cfg）
```python
paramwise_cfg = dict(custom_keys={
    'backbone.adapter.backbone.blocks': dict(lr_mult=0.1),  # 光学 ViT
    'backbone_sar.blocks':            dict(lr_mult=0.1),  # SAR ViT
})
```
按参数名匹配分组，解冻后自动以 1/10 lr 微调，保护预训练表征。

---

## 5. 训练流程

### 5.1 数据增强（train_pipeline）
```
Load → RandomRotate(20°) → RandomCrop(512, cat_max_ratio=0.75)
     → RandomFlip(水平) → RandomFlip(垂直) → PhotoMetricDistortion
     → PackSegInputs
```
双时相**同步增强**（`MultiImg*` 前缀保证两时相一致），光度扭曲仅作用于光学通道。

### 5.2 优化器与调度
- **AdamW**：`lr=5e-4, betas=(0.9,0.999), weight_decay=5e-4`
- **IterBasedTrainLoop**：`max_iters=80000, val_interval=8000`
- **checkpoint**：`save_best='mIoU', max_keep_ckpts=5`

### 5.3 评估
`mmseg.IoUMetric`，指标 `['mFscore', 'mIoU']`，ignore_index=255。

---

## 6. 关键超参数速查表

| 超参 | 值 | 说明 |
|------|----|------|
| crop_size | 512×512 | 训练/滑窗尺寸 |
| batch_size | 8 | 单卡 |
| SAR native_size | 384 | SAR ViT 处理分辨率 |
| extract_ids | [5,11,17,23] | ViT-L 交互层 |
| fpn_channels | 128 | 解码器通道 |
| cross_num_heads | 4 | 交叉注意力头数 |
| window_size | 8 | 窗口注意力 |
| gate_init | -1.0 | 上下文门控初值 |
| focal_gamma | 4.0 | Focal 难度 |
| lovasz_weight | 1.0 | Lovász 权重 |
| EMA momentum | 2e-4 | 影子权重动量 |
| 解冻 iter | 40000 | 阶段2 切换点 |
| unfreeze_last_n | 2 | 解冻层数 |
| backbone lr_mult | 0.1 | 微调学习率系数 |
| ms_weights | [0.5,0.3,0.1,0.1] | 多尺度融合权重 |

---

## 7. 训练命令

```bash
OPENCV_LOG_LEVEL=ERROR CUDA_VISIBLE_DEVICES=4 \
python tools/train.py \
  configs/changedino/dualmode_self-dino-gram-new_self-olmoearth_512x512_slide_ctxfused_ema_80k_dfc2025bright_ftlast2.py
```

多卡 DDP：
```bash
CUDA_VISIBLE_DEVICES=0,1,4 NPROC_PER_NODE=3 \
bash tools/dist_train.sh \
  configs/changedino/dualmode_self-dino-gram-new_self-olmoearth_512x512_slide_ctxfused_ema_80k_dfc2025bright_ftlast2.py <NGPUS>
```

---

## 8. 显存与调优建议

### 8.1 显存特征
- 两 backbone 注意力均为显存友好型（DINOv3 MemEff + SAR SDPA），**不物化 N×N 注意力矩阵**。
- 阶段2 解冻后额外显存主要来自：解冻层的反向激活（SAR token=2304 是大头）+ 梯度/AdamW 状态，估算约 **1.5–3 GB**。
- `use_checkpoint=False`（保速度）关闭了梯度检查点，显存压力由 bs / native_size 承担。

### 8.2 OOM 应对（按优先级）
1. `batch_size 8 → 6`：激活降 ~25%，最稳。
2. `SAR native_size 384 → 320`：token 2304→1600，SAR 激活降 ~30%。
3. `unfreeze_last_n 2 → 1`：解冻层激活减半。
4. 仅解冻 SAR、光学保持冻结（SAR 域适配收益通常更关键）。

### 8.3 提升 mIoU 的后续方向
- **focal_gamma 4.0 → 2.0**：4.0 过度抑制易类，可能让 background/intact 学不动。
- 浅层 aux 权重提高（p2/p3 加权），强化细节边界。
- 滑窗 stride 降到 (192,192)，减少边缘伪影。
- decoder `n_layers [1,1,1,1] → [2,2,2,2]`，backbone 冻结时提升 decoder 容量。
- 统计真实类别频率，按 `1/√freq` 重设 focal_alpha。
- 增加多尺度 TTA（0.75×/1.0×/1.25×/1.5×）。
- 后处理：形态学开运算 + 连通域过滤小碎片。

---

## 9. 模块文件索引

| 模块 | 文件路径 |
|------|----------|
| 检测器包装 | `opencd/models/change_detectors/dual_mode_encoder_decoder.py` |
| 父类检测器 | `opencd/models/change_detectors/changedino_encoder_decoder.py` |
| 解码器 | `opencd/models/change_detectors/changedino_decoder.py` |
| 光学编码器 | `opencd/models/backbones/dinov3_adapter.py` |
| SAR 编码器 | `opencd/models/backbones/sar_olmoearth_encoder.py` |
| 冻结调度 Hook | `opencd/models/change_detectors/freeze_schedule_hook.py` |
| 损失函数 | `opencd/models/losses/{focal_loss,lovasz_loss}.py` |
| 数据集 | `opencd/datasets/dfc2025_bright_dataset.py` |
| Adapter block | `opencd/models/blocks/adapter.py` |
| 可变形注意力 | `opencd/models/blocks/ms_deform_attn.py` |
