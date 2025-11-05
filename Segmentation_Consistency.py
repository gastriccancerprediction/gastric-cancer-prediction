# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
from medpy.metric import binary
from scipy.spatial.distance import directed_hausdorff

# ======== xx ========
folder = r"C:\xx\DSC"
# ==============================

def hausdorff_95(surface1, surface2, spacing):
    """95% Hausdorff"""
    from scipy.ndimage import distance_transform_edt
    dt1 = distance_transform_edt(1 - surface1, sampling=spacing)
    dt2 = distance_transform_edt(1 - surface2, sampling=spacing)
    sds1 = dt2[surface1 > 0]
    sds2 = dt1[surface2 > 0]
    hd95 = np.percentile(np.hstack((sds1, sds2)), 95)
    return hd95

def average_surface_distance(mask1, mask2, spacing):
    """（ASD）"""
    from scipy.ndimage import distance_transform_edt
    dt1 = distance_transform_edt(1 - mask1, sampling=spacing)
    dt2 = distance_transform_edt(1 - mask2, sampling=spacing)
    sds1 = dt2[mask1 > 0]
    sds2 = dt1[mask2 > 0]
    asd = (np.mean(sds1) + np.mean(sds2)) / 2.0
    return asd

def compute_metrics(mask1, mask2, spacing=(1,1,1)):
    """"""
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    dice = binary.dc(mask1, mask2)
    jaccard = binary.jc(mask1, mask2)
    hd95 = hausdorff_95(mask1, mask2, spacing)
    asd = average_surface_distance(mask1, mask2, spacing)
    return dice, jaccard, hd95, asd

results = []
for f in os.listdir(folder):
    if f.endswith('.nii.gz') and '(1)' not in f:
        base = f.replace('.nii.gz', '')
        f1 = os.path.join(folder, f)  # A
        f2 = os.path.join(folder, base + '(1).nii.gz')  # B
        if not os.path.exists(f2):
            print(f"⚠️ 未找到匹配文件: {f2}")
            continue
        
        #
        nii1 = nib.load(f1)
        nii2 = nib.load(f2)
        mask1 = nii1.get_fdata()
        mask2 = nii2.get_fdata()
        spacing = nii1.header.get_zooms()[:3]  
        
        dice, iou, hd95, asd = compute_metrics(mask1, mask2, spacing)
        results.append([base, dice, iou, hd95, asd])
        print(f"{base}: Dice={dice:.4f}, mIoU={iou:.4f}, HD95={hd95:.2f}mm, ASD={asd:.2f}mm")

# 汇总结果
import pandas as pd
df = pd.DataFrame(results, columns=["ID", "Dice", "mIoU", "HD95(mm)", "ASD(mm)"])
save_path = os.path.join(folder, "Segmentation_Consistency.csv")
df.to_csv(save_path, index=False)
print("\n✅ 所有结果已保存至：", save_path)
