# -*- coding: utf-8 -*-
"""
@author: xx
"""
import os
import shutil
import pandas as pd

excel_path = "C:/Users/xx.xlsx"  
data_folder = "F:/xx"  
backup_folder = "C:/xx"  

os.makedirs(backup_folder, exist_ok=True)

# Excel
df = pd.read_excel(excel_path, sheet_name='Test', dtype=str) 

for _, row in df.iterrows():
    old_name = f"{row['L3_Index']}.nii.gz"  
    new_name = f"{row['Index']}.nii.gz"  
    
    old_path = os.path.join(data_folder, old_name)
    new_path = os.path.join(data_folder, new_name)
    backup_path = os.path.join(backup_folder, old_name)

    if os.path.exists(old_path):
        shutil.copy(old_path, backup_path)  
        os.rename(old_path, new_path) 
        print(f"已重命名: {old_name} -> {new_name}")
    else:
        print(f"文件未找到: {old_name}")

print("文件重命名完成！所有原始文件已备份到 data2D_backup 文件夹。")
