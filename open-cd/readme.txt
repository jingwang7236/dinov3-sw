# Open-CD 变化检测项目代码与训练启动命令说明
## 一、项目代码存放路径
基于 OpenMMLab 框架开发的遥感变化检测代码路径：
`/mnt/ht2-nas2/00-model/00-wj/Codes/open-cd`

## 二、模型训练启动脚本
### 1. 纯光学影像变化检测模型（ChangeDINO 简化版）
适用场景：仅输入前后时相光学遥感图像完成变化检测
GPU分配：单卡0号卡
启动命令：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/changedino/self-dino_standard_light_adapter_freeze_256x256_80k_sysucd.py
```
配置说明：输入256×256、ChangeDINO 去除轻量化分支、自研Dinov3权重冻结训练、80k迭代，训练数据集为SYSU-CD

### 2. 光学+SAR双模态变化检测模型（ChangeDINO简化版+OlmoEarth）
适用场景：融合前时相光学影像、后时相SAR影像开展跨模态变化检测
GPU分配：单卡1号卡
启动命令：
```bash
OPENCV_LOG_LEVEL=ERROR CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/changedino/dualmode_self-dino_self-olmoearth_512x512_slide_ctxfused_ema_80k_dfc2025bright.py
```
配置说明：512×512输入、自研Dinov3权重和自研Olmoearth权重 冻结训练、EMA权重平滑、80k迭代，适配DFC2025 Bright数据集

## 备注
1. 代码根目录统一为上述NAS路径，所有训练、测试脚本均在`tools/`文件夹下；
2. 所有模型配置文件集中存放于`configs/changedino/`目录，文件名包含输入尺寸、训练策略、数据集标识，可快速区分实验版本；
3. 配置文件名中包含`self-`字段的是使用自研模型的配置，`dualmode_`开头的配置文件是前光后SAR任务，其他的都是纯光变化检测任务。
