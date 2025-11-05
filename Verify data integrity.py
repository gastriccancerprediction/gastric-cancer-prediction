# -*- coding: utf-8 -*-
"""
@author: xx
"""

import os
import pandas as pd

# Excel
excel_path = "C:/Users/xx.xlsx"  
data_folder = "F:/xx"  

df = pd.read_excel(excel_path,sheet_name='Test',dtype={'Index': str})  
index_names = set(df['Index'].astype(str))  

# 
folder_files = set(f.split('.nii.gz')[0] for f in os.listdir(data_folder) if f.endswith('.nii.gz'))

# 
matched_files = index_names & folder_files  
only_in_excel = index_names - folder_files  
only_in_folder = folder_files - index_names 

print(f"匹配的文件数量: {len(matched_files)}")
print(f"仅在Excel中的文件数量: {len(only_in_excel)}")
print(f"仅在data_GC中的文件数量: {len(only_in_folder)}")

# 可选：将结果保存到CSV
result_df = pd.DataFrame({
    "Matched Files": list(matched_files) + [None] * (max(len(only_in_excel), len(only_in_folder)) - len(matched_files)),
    "Only in Excel": list(only_in_excel) + [None] * (max(len(matched_files), len(only_in_folder)) - len(only_in_excel)),
    "Only in Folder": list(only_in_folder) + [None] * (max(len(matched_files), len(only_in_excel)) - len(only_in_folder))
})

result_df.to_csv("C:/Users/xx.csv", index=False)
print("结果已保存到 comparison_results.csv")
