# -*- coding: utf-8 -*-
"""
分割骨骼肌、肌间脂肪（IMAT）、内脏脂肪，并二值化保存
每个组织单独保存为nii.gz，0/1表示
"""

import os
import nibabel as nib
import numpy as np

input_folder = r"F:\zhangx_GastricCancer\Dataset_Processed\L3_Img_2D"
output_folder = r"C:\Users\ZHANGXIAO\Desktop\chenqiu2-20250402\binary_seg_20251011"

os.makedirs(output_folder, exist_ok=True)

# 灰度区间
muscle_range = (-29, 150)
imat_range = (-190, -30)
vat_range = (-150, -50)

for filename in os.listdir(input_folder):
    if filename.endswith(".nii.gz"):
        filepath = os.path.join(input_folder, filename)
        img = nib.load(filepath)
        data = img.get_fdata()
        
        # 二值化掩膜
        muscle_mask = ((data >= muscle_range[0]) & (data <= muscle_range[1])).astype(np.uint8)
        imat_mask = ((data >= imat_range[0]) & (data <= imat_range[1])).astype(np.uint8)
        vat_mask = ((data >= vat_range[0]) & (data <= vat_range[1])).astype(np.uint8)

        # 保存骨骼肌
        nib.save(nib.Nifti1Image(muscle_mask, affine=img.affine, header=img.header),
                 os.path.join(output_folder, filename.replace(".nii.gz", "_muscle.nii.gz")))
        
        # 保存肌间脂肪
        nib.save(nib.Nifti1Image(imat_mask, affine=img.affine, header=img.header),
                 os.path.join(output_folder, filename.replace(".nii.gz", "_IMAT.nii.gz")))
        
        # 保存内脏脂肪
        nib.save(nib.Nifti1Image(vat_mask, affine=img.affine, header=img.header),
                 os.path.join(output_folder, filename.replace(".nii.gz", "_VAT.nii.gz")))
        
        print(f"已处理并保存：{filename}")

print("\n✅ 所有文件处理完成！")
