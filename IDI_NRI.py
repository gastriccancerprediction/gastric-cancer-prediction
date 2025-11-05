# -*- coding: utf-8 -*-
"""
# IDI-NRI
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


def calculate_nri_idi(y_true, prob_old, prob_new, threshold=0.5):
    """   
    - y_true: （0 - 1）
    - prob_old: 
    - prob_new: 
    - threshold: 0.5
    
    return：
    - NRI: 
    - IDI: 
    """
    y_true = np.array(y_true)
    prob_old = np.array(prob_old)
    prob_new = np.array(prob_new)
    
    #  NRI
    reclass_up_event = np.sum((prob_new >= threshold) & (prob_old < threshold) & (y_true == 1))  
    reclass_down_event = np.sum((prob_new < threshold) & (prob_old >= threshold) & (y_true == 1)) 
    reclass_up_nonevent = np.sum((prob_new >= threshold) & (prob_old < threshold) & (y_true == 0))  
    reclass_down_nonevent = np.sum((prob_new < threshold) & (prob_old >= threshold) & (y_true == 0))  
    
    n_event = np.sum(y_true == 1)
    n_nonevent = np.sum(y_true == 0)
    
    nri = (reclass_up_event / n_event - reclass_down_event / n_event) - (reclass_up_nonevent / n_nonevent - reclass_down_nonevent / n_nonevent)
    
    #  IDI
    mean_prob_event_old = np.mean(prob_old[y_true == 1])
    mean_prob_event_new = np.mean(prob_new[y_true == 1])
    mean_prob_nonevent_old = np.mean(prob_old[y_true == 0])
    mean_prob_nonevent_new = np.mean(prob_new[y_true == 0])
    
    idi = (mean_prob_event_new - mean_prob_event_old) - (mean_prob_nonevent_new - mean_prob_nonevent_old)
    
    return nri, idi


def compute_de_long_test(y_true, pred1, pred2):

    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    
    # DeLong 
    def delong_variance(y_true, pred):
        n1 = sum(y_true == 1)
        n2 = sum(y_true == 0)
        v = np.var(pred[y_true == 1]) / n1 + np.var(pred[y_true == 0]) / n2
        return v

    var1 = delong_variance(y_true, pred1)
    var2 = delong_variance(y_true, pred2)
    cov = np.cov(pred1, pred2)[0, 1] / len(y_true)  

    #  DeLong
    delta_auc = auc1 - auc2
    se = np.sqrt(var1 + var2 - 2 * cov)
    z = delta_auc / se
    p_value = 2 * (1 - norm.cdf(abs(z)))  

    return auc1, auc2, p_value

def bootstrap_nri_idi_pvalue(y_true, prob_old, prob_new, threshold=0.5, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    y_true = np.array(y_true)
    prob_old = np.array(prob_old)
    prob_new = np.array(prob_new)
    
    nri_vals = []
    idi_vals = []
    
    for _ in range(n_bootstrap):
      
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        y_sample = y_true[idx]
        prob_old_sample = prob_old[idx]
        prob_new_sample = prob_new[idx]
        
        nri_sample, idi_sample = calculate_nri_idi(y_sample, prob_old_sample, prob_new_sample, threshold)
        nri_vals.append(nri_sample)
        idi_vals.append(idi_sample)

    nri_vals = np.array(nri_vals)
    idi_vals = np.array(idi_vals)
    

    nri_obs, idi_obs = calculate_nri_idi(y_true, prob_old, prob_new, threshold)

    p_value_nri = 2 * min(np.mean(nri_vals >= nri_obs), np.mean(nri_vals <= nri_obs))
    p_value_idi = 2 * min(np.mean(idi_vals >= idi_obs), np.mean(idi_vals <= idi_obs))

    return nri_obs, idi_obs, p_value_nri, p_value_idi


def extractor_excel_RFS(xlsx):
    RFS_Clinical = np.array(xlsx['Clinical model'])
    RFS_SM = np.array(xlsx['SM model'])
    RFS_Tumor = np.array(xlsx['Tumor model'])
    RFS_CT = np.array(xlsx['CT model'])
    RFS_SMT = np.array(xlsx['SMT model'])
    RFS_CSMT = np.array(xlsx['CSMT model'])
    
    RFS_Fat = np.array(xlsx['Fat model'])
    RFS_CSMTFat = np.array(xlsx['CSMT-Fat model'])
    RFS = np.array(xlsx['RFS'])
    return RFS_Clinical,RFS_SM,RFS_Tumor,RFS_CT,RFS_SMT,RFS_CSMT,RFS_Fat,RFS_CSMTFat,RFS



if __name__ == '__main__':

    xlsx_path = 'C:/Users/xx.xlsx'
    xlsx_train = pd.read_excel(xlsx_path, sheet_name='Train')
    xlsx_val = pd.read_excel(xlsx_path, sheet_name='Val')
    xlsx_test = pd.read_excel(xlsx_path, sheet_name='Test')
    
    RFS_Clinical,RFS_SM,RFS_Tumor,RFS_CT,RFS_SMT,RFS_CSMT,RFS_Fat,RFS_CSMTFat,RFS = extractor_excel_RFS(xlsx_test)
    

    y_true = RFS
    pred1 = RFS_CSMTFat
    pred2 = RFS_CSMT
    auc1, auc2, p_value = compute_de_long_test(y_true, pred1, pred2)
    print(f"p-value: {p_value:.4f}")
    
    
    
    y_true = RFS 
    prob_old = RFS_Tumor
    prob_new = RFS_SMT
    nri, idi = calculate_nri_idi(y_true, prob_old, prob_new, threshold=0.5)
    print(f"NRI: {nri:.4f}")
    print(f"IDI: {idi:.4f}")
    
    if nri > 0:
        print("NRI > 0，表明新模型在正确方向上的重新分类改善")
    else:
        print("NRI <= 0，表明新模型在正确方向上的改善有限或变差")

    if idi > 0:
        print("IDI > 0，表明新模型的整体判别能力比旧模型更好")
    else:
        print("IDI <= 0，表明新模型的整体判别能力未改善或变差")
        
    a = 0.0156
    print(a*0.615,a*1.555)


    nri_obs, idi_obs, p_nri, p_idi = bootstrap_nri_idi_pvalue(RFS, prob_old, prob_new, threshold=0.5, n_bootstrap=1000)
    print(f"Bootstrap NRI: {nri_obs:.4f}, p-value: {p_nri:.4f}")
    print(f"Bootstrap IDI: {idi_obs:.4f}, p-value: {p_idi:.4f}")




























