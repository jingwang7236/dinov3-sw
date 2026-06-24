"""
LEVIR-CD 数据集滑窗裁剪工具 (一键执行版)

将 1024x1024 的原始图像裁剪为 256x256 的图像块，保持原始目录结构。
自动适配 445/64/128 张图像 → 7,120/1,024/2,048 张切片。

输入目录结构 (原始):
    data_root/
    ├── train/
    │   ├── T1/  (时相1)
    │   ├── T2/  (时相2)
    │   └── GT/  (标签)
    ├── val/
    │   ├── T1/
    │   ├── T2/
    │   └── GT/
    └── test/
        ├── T1/
        ├── T2/
        └── GT/

输出目录结构 (保持不变):
    data_root_crop_256/
    ├── train/
    │   ├── T1/
    │   ├── T2/
    │   └── GT/
    ├── val/
    │   ├── T1/
    │   ├── T2/
    │   └── GT/
    └── test/
        ├── T1/
        ├── T2/
        └── GT/
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


class LEVIRCDCropper:
    """LEVIR-CD 数据集滑窗裁剪器 - 一键执行版"""
    
    def __init__(self, data_root: str, crop_size: int = 256, output_root: str = 'output'):
        """
        Args:
            data_root: 输入数据根目录 (包含 train/val/test 子目录)
            crop_size: 裁剪尺寸
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.crop_size = crop_size
        self.stride = crop_size  # 无重叠滑窗
        
        # 子目录名称 (保持原始命名)
        self.phases = ['train', 'val', 'test']
        self.subdirs = ['T1', 'T2', 'GT']
        self.suffix = '.png'
        
        # 统计信息
        self.stats = {}
    
    def _get_image_files(self, phase: str, subdir: str):
        """获取指定阶段和子目录的所有图像文件"""
        img_dir = self.data_root / phase / subdir
        if not img_dir.exists():
            return []
        return sorted(img_dir.glob(f'*{self.suffix}'))
    
    def _crop_image(self, img: np.ndarray):
        """将图像裁剪为 crop_size x crop_size 的块"""
        h, w = img.shape[:2]
        crops = []
        positions = []
        
        for y in range(0, h - self.crop_size + 1, self.stride):
            for x in range(0, w - self.crop_size + 1, self.stride):
                crop = img[y:y+self.crop_size, x:x+self.crop_size]
                crops.append(crop)
                positions.append((y, x))
        
        return crops, positions
    
    def _get_crop_name(self, stem: str, row_idx: int, col_idx: int) -> str:
        """生成裁剪后的文件名"""
        # 原始格式: 000001.png
        # 裁剪后: 000001_0_0.png
        return f"{stem}_{row_idx}_{col_idx}{self.suffix}"
    
    def _save_crops(self, crops, output_dir, stem, positions):
        """保存裁剪后的图像块"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (crop, (y, x)) in enumerate(zip(crops, positions)):
            row_idx = y // self.stride
            col_idx = x // self.stride
            crop_name = self._get_crop_name(stem, row_idx, col_idx)
            cv2.imwrite(str(output_dir / crop_name), crop)
    
    def _process_phase(self, phase: str) -> dict:
        """处理一个阶段 (train/val/test)"""
        # 获取三个子目录的文件
        t1_files = self._get_image_files(phase, 'T1')
        t2_files = self._get_image_files(phase, 'T2')
        gt_files = self._get_image_files(phase, 'GT')
        
        # 验证文件数量
        n_files = len(t1_files)
        if n_files == 0:
            return {'pairs': 0, 'crops': 0}
        
        if not (len(t2_files) == n_files and len(gt_files) == n_files):
            print(f"⚠️ {phase}: 文件数量不一致 T1={len(t1_files)}, T2={len(t2_files)}, GT={len(gt_files)}")
            return {'pairs': 0, 'crops': 0}
        
        # 创建输出目录
        out_dirs = {
            'T1': self.output_root / phase / 'T1',
            'T2': self.output_root / phase / 'T2',
            'GT': self.output_root / phase / 'GT',
        }
        
        total_crops = 0
        
        # 遍历处理
        desc = f"📁 {phase.upper()} ({n_files} 对)"
        for t1_path, t2_path, gt_path in tqdm(
            zip(t1_files, t2_files, gt_files), 
            desc=desc, 
            total=n_files,
            leave=False
        ):
            # 读取图像
            img_t1 = cv2.imread(str(t1_path))
            img_t2 = cv2.imread(str(t2_path))
            img_gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            
            if img_t1 is None or img_t2 is None or img_gt is None:
                print(f"⚠️ 读取失败: {t1_path.name}")
                continue
            
            # 裁剪
            crops_t1, positions = self._crop_image(img_t1)
            crops_t2, _ = self._crop_image(img_t2)
            crops_gt, _ = self._crop_image(img_gt)
            
            # 保存
            stem = t1_path.stem
            self._save_crops(crops_t1, out_dirs['T1'], stem, positions)
            self._save_crops(crops_t2, out_dirs['T2'], stem, positions)
            self._save_crops(crops_gt, out_dirs['GT'], stem, positions)
            
            total_crops += len(positions)
        
        return {'pairs': n_files, 'crops': total_crops}
    
    def run(self):
        """一键执行裁剪"""
        print("=" * 65)
        print("🔪 LEVIR-CD 滑窗裁剪工具 (一键执行)")
        print("=" * 65)
        print(f"📂 输入目录: {self.data_root}")
        print(f"📂 输出目录: {self.output_root}")
        print(f"📐 裁剪尺寸: {self.crop_size}x{self.crop_size}")
        print(f"📐 滑窗步长: {self.stride}")
        print(f"📁 每张图切片数: 16 (4x4)")
        print("=" * 65)
        
        # 预期数量
        expected = {'train': (445, 7120), 'val': (64, 1024), 'test': (128, 2048)}
        
        print("\n📊 预期输出:")
        for phase, (pairs, crops) in expected.items():
            print(f"  {phase.upper()}: {pairs} 对 → {crops} 张切片")
        print("-" * 65)
        
        # 创建输出根目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 处理各阶段
        total_pairs = 0
        total_crops = 0
        
        for phase in self.phases:
            result = self._process_phase(phase)
            self.stats[phase] = result
            total_pairs += result['pairs']
            total_crops += result['crops']
            
            # 打印阶段结果
            expected_pairs, expected_crops = expected.get(phase, (0, 0))
            status = "✅" if result['pairs'] == expected_pairs else "⚠️"
            print(f"  {status} {phase.upper()}: {result['pairs']} 对 → {result['crops']} 张切片 (预期: {expected_crops})")
        
        # 汇总统计
        print("\n" + "=" * 65)
        print("📊 裁剪完成汇总")
        print("=" * 65)
        print(f"  总图像对数: {total_pairs} (预期: 637)")
        print(f"  总切片数:   {total_crops} (预期: 10,192)")
        print(f"  切片放大比: {total_crops / total_pairs:.1f}x")
        print("-" * 65)
        print(f"✅ 输出目录: {self.output_root}")
        print("=" * 65)
        
        # 验证与论文一致
        if total_pairs == 637 and total_crops == 10192:
            print("\n🎉 切片数量与论文完全一致!")
            print("   LEVIR-CD 论文: 7,120/1,024/2,048 = 10,192 张")
        else:
            print("\n⚠️ 切片数量与预期不符，请检查原始数据")


def main():
    # ============================================================
    # 🔧 只需要修改这里的路径即可
    # ============================================================
    DATA_ROOT = "/mnt/ht2-nas2/EO_test/dataset/ChangeDetection/LEVIR-CD"
    OUTPUT_ROOT = "/mnt/ht2-nas2/EO_test/dataset/ChangeDetection/LEVIR-CD-Patch"
    CROP_SIZE = 256
    # ============================================================
    
    cropper = LEVIRCDCropper(data_root=DATA_ROOT, crop_size=CROP_SIZE, output_root=OUTPUT_ROOT)
    cropper.run()


if __name__ == '__main__':
    main()