import os
import json
import h5py
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ===================== 配置项（改成你的路径） =====================
INPUT_FOLDER = "/mnt/ht2-nas2/EO_test/dataset/geo-bench-1.0/segmentation_v1.0/m-NeonTree"  # 放h5和json的目录
OUTPUT_FOLDER = "/mnt/ht2-nas2/00-model/00-wj/Data/m-NeonTree"    # 输出保存路径
json_file = os.path.join(INPUT_FOLDER, "default_partition.json")
# ==================================================================

def main():
    # 1. 查找所有 hdf5 文件
    h5_files = list(Path(INPUT_FOLDER).glob("**/*.hdf5"))
    if not h5_files:
        print("❌ 未找到任何 .h5 文件！")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        train_names = json_data["train"]
        test_names = json_data["test"]
        # import pdb;pdb.set_trace()

    # 3. 创建输出目录

    for mode in ["train", "test"]:
        save_dir = f"{OUTPUT_FOLDER}/images/{mode}"
        save_dir_mask = f"{OUTPUT_FOLDER}/masks/{mode}"
        save_vis_mask = f"{OUTPUT_FOLDER}/vis/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir_mask, exist_ok=True)
        os.makedirs(save_vis_mask, exist_ok=True)


    # 4. 批量处理所有 h5 文件
    for h5_path in h5_files:
        task_name = h5_path.stem
        print(f"\n======== 处理任务：{task_name} ========")
        h5_name = os.path.basename(h5_path).split(".")[0]
        print("h5_name: ", h5_name)
        if h5_name in train_names:
            mode = "train"
        elif h5_name in test_names:
            mode = "test"
        else:
            print(f"❌ 未找到 {h5_name} 在 json 中的信息！")
            continue

        with h5py.File(h5_path, "r") as f:
            # 读取三通道
            R = f["Red"][:]
            G = f["Green"][:]
            B = f["Blue"][:]
            label = f["label"][:]
            #TODO: 将1像素变成其他颜色，方便可视化,1像素看不清,得到新的mask变量:vis_label
            
            # 查看label中有多少种数值
            # print(np.unique(label))
            import pdb;pdb.set_trace()
            vis_label = np.where(label==1,255,label)

            # 堆叠成 (3, H, W)
            rgb = np.stack([R, G, B], axis=0)  # (3,400,400)

            # 保存TIFF
            tif_path = os.path.join(save_dir, f"{task_name}.tif")
            save_tiff(rgb, tif_path)

            # 保存PNG mask
            png_path = os.path.join(save_dir_mask, f"{task_name}.png")
            save_mask_png(label, png_path)

            vis_path = os.path.join(save_vis_mask, f"{task_name}.png")
            save_mask_png(vis_label, vis_path)

            print(f"✅ {task_name}.tif + .png save successfully!")

    print("\n🎉 全部处理完成！")
    print(f"📂 输出路径：{OUTPUT_FOLDER}")

def save_tiff(bands, save_path):
    """保存多通道遥感图像为 TIFF 格式"""
    c, h, w = bands.shape
    # rasterio 需要 (C, H, W) 格式
    with rasterio.open(
        save_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=bands.dtype,
    ) as dst:
        dst.write(bands)

def save_mask_png(mask, save_path):
    """保存单通道掩码为 PNG"""
    mask = mask.astype(np.uint8)
    Image.fromarray(mask, mode="L").save(save_path)


# 改成你的 .h5 文件路径
h5_path = "m-neontree-seg.h5"

def print_h5_structure(name, node):
    """递归打印 h5 结构"""
    print(name)
    if isinstance(node, h5py.Dataset):
        print(f"  → 数据类型: {node.dtype}")
        print(f"  → 形状(shape): {node.shape}")
        print(f"  → 属性(attrs): {dict(node.attrs)}")
    print("-" * 50)



if __name__ == "__main__":
    h5_sample_path = os.path.join(INPUT_FOLDER, "2019_OSBS_5_405000_3287000_image_crop2_02_01.hdf5")
    # 打开并遍历
    with h5py.File(h5_sample_path, "r") as f:
        print("📂 HDF5 文件根目录：")
        f.visititems(print_h5_structure)
    main()
