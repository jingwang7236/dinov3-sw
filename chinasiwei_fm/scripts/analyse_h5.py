import h5py
import numpy as np
import torch

# ======================
# 1. 【核心】分析 H5 文件结构（最常用）
# ======================
def inspect_h5_structure(file_path, max_depth=5):
    """
    遍历并打印 HDF5 文件的完整结构：
    - 所有组 (group)
    - 所有数据集 (dataset)
    - 形状、数据类型、维度、大小
    """
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            print(f"📊 数据集: {name}")
            print(f"   形状 (shape): {node.shape}")
            print(f"   数据类型 (dtype): {node.dtype}")
            print(f"   大小 (size): {node.size}")
            print(f"   维度 (ndim): {node.ndim}\n")
        else:
            print(f"📂 组: {name}\n")

    with h5py.File(file_path, 'r') as f:
        print("=" * 80)
        print(f"📦 HDF5 文件路径: {file_path}")
        print(f"📦 文件根目录 keys: {list(f.keys())}")
        print("=" * 80)
        print("\n【完整文件结构】\n")
        f.visititems(visitor_func)

    return list(f.keys())


# ======================
# 2. 读取指定 key 的数据（返回 numpy）
# ======================
def read_h5_dataset(file_path, key):
    with h5py.File(file_path, 'r') as f:
        if key not in f:
            raise ValueError(f"❌ 键 {key} 不存在！文件内有效 keys: {list(f.keys())}")
        data = f[key][:]  # 读取全部数据
    return data


# ======================
# 3. 读取所有数据（自动转成字典）
# ======================
def load_all_h5_data(file_path):
    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data_dict[key] = f[key][:]
    return data_dict


# ======================
# 4. 数据检查：是否适合模型训练（非常实用）
# ======================
def check_data_for_training(data, data_name="data"):
    print(f"\n===== 【{data_name} 训练可用性检查】 =====")
    print(f"形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"最小值: {np.min(data)}")
    print(f"最大值: {np.max(data)}")
    print(f"均值: {np.mean(data):.2f}")
    print(f"是否包含 NaN: {np.isnan(data).any()}")
    print(f"是否包含 Inf: {np.isinf(data).any()}")

    # 常见问题提示
    if np.min(data) < 0 or np.max(data) > 255:
        print("⚠️  提示：数据范围不在 0~255，可能需要归一化")
    if len(data.shape) == 3:
        print("✅ 形状为 (H, W, C)，适合语义分割/图像任务")
    if len(data.shape) == 4:
        print("✅ 形状为 (N, H, W, C)，批量数据，可直接训练")
    print("=" * 50)


# ======================
# 5. 转 PyTorch Tensor（训练用）
# ======================
def to_tensor(data):
    tensor = torch.from_numpy(data).float()
    return tensor


# ======================
# ========== 主程序 ==========
# ======================
if __name__ == '__main__':
    # 替换成你的 H5 文件路径
    H5_FILE = "your_file.h5"

    # --------------------
    # 第一步：分析文件结构（必须先运行！）
    # --------------------
    keys = inspect_h5_structure(H5_FILE)

    # --------------------
    # 第二步：读取你需要的数据（根据上面打印的 key）
    # --------------------
    # 示例：假设你的 H5 里面有 'image' 和 'label'
    try:
        images = read_h5_dataset(H5_FILE, "image")
        labels = read_h5_dataset(H5_FILE, "label")

        # --------------------
        # 第三步：检查数据是否适合训练
        # --------------------
        check_data_for_training(images, "图像数据")
        check_data_for_training(labels, "标签数据")

        # --------------------
        # 第四步：转为 tensor 用于模型训练
        # --------------------
        images_tensor = to_tensor(images)
        labels_tensor = to_tensor(labels)

        print("\n🎉 数据读取完成！可直接送入模型训练")
        print(f"图像张量形状: {images_tensor.shape}")
        print(f"标签张量形状: {labels_tensor.shape}")

    except Exception as e:
        print(f"\n❌ 读取失败: {e}")