# -*- coding: utf-8 -*-
"""
## Clinic Model-RFS
@author: xx
"""

import numpy as np
import random
import os
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

def extractor_excelIndex(xlsx):
    m,n = xlsx.shape
    clinic = np.zeros((m,6),dtype='float')
    clinic[:,0] = np.array(xlsx['T_stage'])
    clinic[:,1] = np.array(xlsx['N_stage'])
    clinic[:,2] = np.array(xlsx['Position'])
    clinic[:,3] = np.array(xlsx['LNR'])
    clinic[:,4] = np.array(xlsx['CEA'])
    clinic[:,5] = np.array(xlsx['chemotherapy'])
    RFS = np.array(xlsx['RFS'])
    return clinic,RFS


if __name__ == '__main__':
    ### 提取临床指标特征
    xlsx_path = 'C:/Users/xx.xlsx'
    xlsx_train = pd.read_excel(xlsx_path, sheet_name='Train')
    xlsx_val = pd.read_excel(xlsx_path, sheet_name='Val')
    xlsx_test = pd.read_excel(xlsx_path, sheet_name='Test')
    
    clinic_train,RFS_train = extractor_excelIndex(xlsx_train)
    clinic_val,RFS_val = extractor_excelIndex(xlsx_val)
    clinic_test,RFS_test = extractor_excelIndex(xlsx_test)
    
    # # ## ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    model = svm.SVC(kernel='rbf',C=1,gamma='auto',probability=(True)).fit(clinic_train, RFS_train)
    y_pred_train =model.predict_proba(clinic_train)
    y_pred_val =model.predict_proba(clinic_val)
    y_pred_test =model.predict_proba(clinic_test)
    
    
    ### Initialize logistic regression model
    model=LogisticRegression(penalty='l2',C=5.0,solver='lbfgs',max_iter=1000,multi_class='ovr')    # liblinear, lbfgs, newton-cg, sag, saga.  multinomial
    model.fit(clinic_train, RFS_train)
    y_pred_train =model.predict_proba(clinic_train)
    y_pred_val =model.predict_proba(clinic_val)
    y_pred_test =model.predict_proba(clinic_test)
    
    ########### 随机森林
    model = RandomForestClassifier(n_estimators=600,max_depth=6,class_weight='balanced',random_state=42)
    model.fit(clinic_train, RFS_train)
    y_pred_train =model.predict_proba(clinic_train)
    y_pred_val =model.predict_proba(clinic_val)
    y_pred_test =model.predict_proba(clinic_test)
    
    
    ########### Clinic-Test 测量指标  #####################
    prob = y_pred_test
    label = RFS_test
    pred = np.argmax(prob, axis = 1)
    # 计算混淆矩阵
    conf_matrix_train = confusion_matrix(label, pred)
    print("Train Confusion Matrix:")
    print(conf_matrix_train)
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    # 计算各类指标
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
    plt.title('RFS_Idx - Train',fontsize=14)
    plt.plot(fpr,tprr,'b',linewidth=1.5)
    # plt.plot(fpr,tpr,'b',label = 'AUC = %0.3f' % auc,linewidth=1.5)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xticks(fontsize=13)  
    plt.yticks(fontsize=13)

    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.show()       

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    