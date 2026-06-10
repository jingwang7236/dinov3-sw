import re
import matplotlib.pyplot as plt
import argparse

# ===================== 配置项 =====================
LOSS_NAMES = [
    "total_loss",       # dinov3 原有loss
    "loss",             # opencd/mmengine 新日志loss
]
# ==================================================

def parse_dinov3_log(line):
    """解析旧版 DINOv3 格式日志行"""
    step_pattern = re.compile(r"\[\s*(\d+)/\d+\]")
    loss_pattern = re.compile(r"(\w+):\s*([\d.]+)\s*\([\d.]+\)")
    
    step_match = step_pattern.search(line)
    if not step_match:
        return None, None
    
    step = int(step_match.group(1))
    loss_matches = loss_pattern.findall(line)
    loss_dict = {k: float(v) for k, v in loss_matches}
    return step, loss_dict

def parse_opencd_log(line):
    """解析新版 OpenCD/MMEngine 格式日志行"""
    # 匹配 Iter(train) [  19/40000]
    step_pattern = re.compile(r"Iter\(train\)\s*\[\s*(\d+)/\d+\]")
    # 匹配 key: value （无括号平均值）
    loss_pattern = re.compile(r"([\w\.]+):\s*([\d.]+)")
    
    step_match = step_pattern.search(line)
    if not step_match:
        return None, None
    
    step = int(step_match.group(1))
    loss_matches = loss_pattern.findall(line)
    loss_dict = {k: float(v) for k, v in loss_matches}
    return step, loss_dict

def parse_log(log_path):
    """自动识别两种日志格式，统一解析"""
    steps = []
    losses = {name: [] for name in LOSS_NAMES}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            step = None
            loss_dict = None
            
            # 自动判断日志类型并解析
            if "Training" in line:
                step, loss_dict = parse_dinov3_log(line)
            elif "Iter(train)" in line:
                step, loss_dict = parse_opencd_log(line)
            
            # 无效行跳过
            if step is None or loss_dict is None:
                continue
            
            steps.append(step)
            
            # 收集所有配置的loss
            for name in LOSS_NAMES:
                if name in loss_dict:
                    losses[name].append(loss_dict[name])

    print(f"✅ 解析完成，共读取 {len(steps)} 个训练点")
    return steps, losses

def plot_loss_curve(steps, losses, save_path="loss_curve.png", task_name="Training Loss"):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(13, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # 绘制所有有效loss曲线
    for i, name in enumerate(LOSS_NAMES):
        if len(losses[name]) == len(steps) and len(losses[name]) > 0:
            ax.plot(steps, losses[name], label=name, linewidth=2.2, color=colors[i % len(colors)])

    # 【关键优化】自动缩放Y轴，微小变化也清晰可见
    all_values = []
    for name in LOSS_NAMES:
        if len(losses[name]) > 0:
            all_values.extend(losses[name])
    
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        margin = (y_max - y_min) * 0.02
        ax.set_ylim(y_min - margin, y_max + margin)

    # 样式美化
    ax.set_xlabel("Training Steps", fontsize=13, weight='bold')
    ax.set_ylabel("Loss Value", fontsize=13, weight='bold')
    ax.set_title(task_name, fontsize=15, weight='bold')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存高清图片
    plt.savefig(save_path, dpi=350, bbox_inches="tight")
    print(f"📊 曲线已保存：{save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="兼容 DINOv3 + OpenCD 双格式 loss 曲线绘制")
    parser.add_argument("--log", required=True, help="日志文件路径")
    parser.add_argument("--save", default="loss_curve.png", help="保存图片路径")
    parser.add_argument("--task_name", default="Training Loss Curve", help="训练任务名称")
    args = parser.parse_args()

    steps, losses = parse_log(args.log)
    plot_loss_curve(steps, losses, args.save, args.task_name)

if __name__ == "__main__":
    main()