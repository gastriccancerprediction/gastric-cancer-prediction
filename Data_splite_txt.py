# -*- coding: utf-8 -*-
"""
### Train-Val-Test，->txt
@author: xx
"""

import numpy as np
import pandas as pd

def extractor_excelIndex(xlsx):
    m,n = xlsx.shape
    Idx = np.zeros((m,1))
    Idx = np.array(xlsx['Index'])
    return Idx

def extractor_name(idx_temp):
    nameFiles = []
    for i in range(len(idx_temp)):
        names = idx_temp[i]
        nameFiles.append(names)
    return nameFiles


if __name__ == '__main__':

    xls_train = pd.read_excel('C:/Users/xx.xlsx',sheet_name='Train')
    xls_val = pd.read_excel('C:/Users/xx.xlsx',sheet_name='Val')
    xls_test = pd.read_excel('C:/Users/xx.xlsx',sheet_name='Test')
    
    idx_train = extractor_excelIndex(xls_train)
    idx_val = extractor_excelIndex(xls_val)
    idx_test = extractor_excelIndex(xls_test)
    
    name_train = extractor_name(idx_train)
    name_val = extractor_name(idx_val)
    name_test = extractor_name(idx_test)

    file_train = "C:/Users/xx.txt"
    file_val = "C:/Users/xx.txt"
    file_test = "C:/Users/xx.txt"
    
    ## 保存数据
    np.savetxt(file_train,name_train,fmt='%s')
    np.savetxt(file_val,name_val,fmt='%s')
    np.savetxt(file_test,name_test,fmt='%s')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    