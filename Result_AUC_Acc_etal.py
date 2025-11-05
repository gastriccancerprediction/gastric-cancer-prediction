# -*- coding: utf-8 -*-
"""
@author: xx
"""

import numpy as np
import random
import os
import torch
import pandas as pd
from sklearn.metrics import auc,roc_auc_score,roc_curve,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
from lifelines.utils import concordance_index

def delong_roc_test(y_true, prob_1, prob_2):
    """
     DeLong test 
    """
    def compute_auc_variance(y_true, prob):
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_value = auc(fpr, tpr)
        n1 = sum(y_true == 1)
        n2 = sum(y_true == 0)
        v = (auc_value * (1 - auc_value) + (n1 - 1) * (auc_value / (2 - auc_value))**2 +
             (n2 - 1) * ((1 - auc_value) / (1 + auc_value))**2) / (n1 * n2)
        return auc_value, v

    auc1, var1 = compute_auc_variance(y_true, prob_1)
    auc2, var2 = compute_auc_variance(y_true, prob_2)
    z = (auc1 - auc2) / np.sqrt(var1 + var2)
    p_value = 2 * (1 - norm.cdf(abs(z)))  
    return p_value

def extractor_excel_RFS(xlsx):
    RFS_Clinical = np.array(xlsx['Clinical model'])
    RFS_SM = np.array(xlsx['SM model'])
    RFS_Tumor = np.array(xlsx['Tumor model'])
    RFS_CT = np.array(xlsx['CT model'])
    RFS_SMT = np.array(xlsx['SMT model'])
    RFS_CSMT = np.array(xlsx['CSMT model'])
    RFS = np.array(xlsx['RFS'])
    return RFS_Clinical,RFS_SM,RFS_Tumor,RFS_CT,RFS_SMT,RFS_CSMT,RFS



if __name__ == '__main__':
    ###
    xlsx_path = 'C:/Users/xx.xlsx'
    xlsx_train = pd.read_excel(xlsx_path, sheet_name='Train')
    xlsx_val = pd.read_excel(xlsx_path, sheet_name='Val')
    xlsx_test = pd.read_excel(xlsx_path, sheet_name='Test')
    
    RFS_Clinical,RFS_SM,RFS_Tumor,RFS_CT,RFS_SMT,RFS_CSMT,RFS = extractor_excel_RFS(xlsx_test)
  
    label = RFS
    #######   #########
    prob_score =  RFS_CSMT   
    
    prob = np.column_stack((1-prob_score, prob_score))
    pred = np.argmax(prob, axis = 1)
    
    c_index = concordance_index(label,prob[:,1])
    print("C_Index:", c_index)
    
    
    
    
    

























