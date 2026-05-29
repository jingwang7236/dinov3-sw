import re
import matplotlib.pyplot as plt
import argparse

# ===================== 配置项 =====================
LOSS_NAMES = [
    "total_loss",
    # "dino_local_crops_loss",
    # "dino_global_crops_loss",
    # "ibot_loss",
    # "koleo_loss"
]
# ==================================================

def parse_log(log_path):
    steps = []
    losses = {name: [] for name in LOSS_NAMES}

    step_pattern = re.compile(r"\[\s*(\d+)/\d+\]")
    loss_pattern = re.compile(r"(\w+):\s*([\d.]+)\s*\([\d.]+\)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Training" not in line:
                continue

            step_match = step_pattern.search(line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            steps.append(step)

            loss_matches = loss_pattern.findall(line)
            loss_dict = {k: float(v) for k, v in loss_matches}

            for name in LOSS_NAMES:
                if name in loss_dict:
                    losses[name].append(loss_dict[name])

    print(f"✅ 解析完成，共读取 {len(steps)} 个训练点")
    return steps, losses

def plot_loss_curve(steps, losses, save_path="loss_curve.png"):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(13, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # 绘制所有loss曲线
    for i, name in enumerate(LOSS_NAMES):
        if len(losses[name]) == len(steps):
            ax.plot(steps, losses[name], label=name, linewidth=2.2, color=colors[i % len(colors)])

    # ===================== 【关键优化】自动缩放Y轴，放大微小变化 =====================
    all_values = []
    for name in LOSS_NAMES:
        all_values.extend(losses[name])
    
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        # 上下留 2% 余量，不贴边
        margin = (y_max - y_min) * 0.02
        ax.set_ylim(y_min - margin, y_max + margin)
    # ============================================================================

    # 样式增强
    ax.set_xlabel("Training Steps", fontsize=13, weight='bold')
    ax.set_ylabel("Loss Value", fontsize=13, weight='bold')
    ax.set_title("DINOv3 Training Loss (Auto-scaled for small changes)", fontsize=15, weight='bold')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存高清图
    plt.savefig(save_path, dpi=350, bbox_inches="tight")
    print(f"📊 曲线已保存：{save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="绘制DINOv3训练loss曲线（自动放大微小变化）")
    parser.add_argument("--log", required=True, help="日志文件路径")
    parser.add_argument("--save", default="loss_curve.png", help="保存图片路径")
    args = parser.parse_args()

    steps, losses = parse_log(args.log)
    plot_loss_curve(steps, losses, args.save)

if __name__ == "__main__":
    main()