# -*- coding: utf-8 -*-
"""
### prob -> C-index
@author: xx
"""


import numpy as np
import random
import os
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index


def extractor_excelIndex(xlsx):
    prob = np.array(xlsx['Prob3'])
    RFS = np.array(xlsx['RFS5'])
    return prob,RFS


if __name__ == '__main__':
    xlsx_path = 'C:/Users/xx.xlsx'
    xlsx_train = pd.read_excel(xlsx_path, sheet_name='Train')
    xlsx_val = pd.read_excel(xlsx_path, sheet_name='Val')
    xlsx_test = pd.read_excel(xlsx_path, sheet_name='Test')
    
    prob_all,RFS_all = extractor_excelIndex(xlsx_val)
    
    ####################################################
    prob_1 = prob_all
    label = RFS_all
    prob_0 = 1-prob_1
    
    prob = np.column_stack((prob_0, prob_1))
    pred = np.argmax(prob, axis = 1)

    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    

    c_index = concordance_index(label,prob[:,1])
    auc=roc_auc_score(label,prob[:,1]) 
    fpr,tprr,thresholds=roc_curve(label,prob[:,1])  
    acc = accuracy_score(label, pred)
    f1 = f1_score(label, pred)
    specificity = tn / (tn + fp) 
    sensitivity = tp / (tp + fn)  
    ppv = tp / (tp + fp)  
    npv = tn / (tn + fn)  

    print('----------------------------')
    print('C_Index:',c_index)
    print('Auc:',auc)
    print('Acc:',acc)
    print('F1:',f1)
    print('Spec:',specificity)
    print('Sen:',sensitivity)
    print('PPV:',ppv)
    print('NPV:',npv)
    print('----------------------------')
    
    
    plt.rcParams['font.sans-serif'] = ['Arial']  
    plt.figure(figsize=(10,8))
    plt.title('RFS_SMDL - Train',fontsize=14)
    plt.plot(fpr,tprr,'b',linewidth=1.5)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xticks(fontsize=13) 
    plt.yticks(fontsize=13)

    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.show()       

    





