# -*- coding: utf-8 -*-
"""
### DCA Curve
@author: xx
"""

import sklearn
from sklearn.metrics import roc_curve,auc,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
import pandas as pd
import os

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Candara'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA_1D(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'green', label = 'SMFTC model')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'darkgreen', alpha = 0.2)

    #Figure Configuration
    ax.set_xlim(0,1)
    ax.set_ylim(0,0.5)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.2)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Candara','fontsize': 13}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Candara','fontsize': 13}
        )
    # ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax

def plot_DCA_3D(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'blue', label = 'SMFTC model')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    #Figure Configuration
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'darkblue', alpha = 0.2)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,0.5)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.05)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Candara', 'fontsize': 13}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Candara', 'fontsize': 13}
        )
    # ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax

def plot_DCA_5D(ax, thresh_group, net_benefit_model, net_benefit_all):
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'red', label = 'SMFTC model')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
    
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'darkred', alpha = 0.2)

    ax.set_xlim(0,1)
    ax.set_ylim(0,0.5)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.015)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Candara', 'fontsize': 13}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Candara', 'fontsize': 13}
        )
    # ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax

#############################################################################################



xlsx_path = 'C:/Users/xx/Dataset_calibration_Roc_curve.xlsx'
xlsx1 = pd.read_excel(xlsx_path, sheet_name='Train')
xlsx2 = pd.read_excel(xlsx_path, sheet_name='Val')
xlsx3 = pd.read_excel(xlsx_path, sheet_name='Test')


###### 1 2 3-Train, Val, Test
df1 = xlsx1[['Prob1','RFS1']]
df3 = xlsx1[['Prob2','RFS3']]
df5 = xlsx1[['Prob3','RFS5']]


##################
thresh_group = np.arange(0,1,0.005)
y_label1 = df1['RFS1']
y_pred_score1 = df1['Prob1']

y_label3 = df3['RFS3']
y_pred_score3 = df3['Prob2']

y_label5 = df5['RFS5']
y_pred_score5 = df5['Prob3']

net_benefit_model1 = calculate_net_benefit_model(thresh_group, y_pred_score1, y_label1)
net_benefit_all1 = calculate_net_benefit_all(thresh_group, y_label1)

net_benefit_model3 = calculate_net_benefit_model(thresh_group, y_pred_score3, y_label3)
net_benefit_all3 = calculate_net_benefit_all(thresh_group, y_label3)

net_benefit_model5 = calculate_net_benefit_model(thresh_group, y_pred_score5, y_label5)
net_benefit_all5 = calculate_net_benefit_all(thresh_group, y_label5)


save_dir = 'C:/Users/xx/Figures_DCA_addFat/Train'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the first plot (1D) in PDF format
fig, ax = plt.subplots()
ax1 = plot_DCA_1D(ax, thresh_group, net_benefit_model1, net_benefit_all1)
plot_path1 = os.path.join(save_dir, 'DCA_1D_plot.pdf')
fig.savefig(plot_path1)
plt.close(fig)

# Save the second plot (3D) in PDF format
fig, ax = plt.subplots()
ax3 = plot_DCA_3D(ax, thresh_group, net_benefit_model3, net_benefit_all3)
plot_path3 = os.path.join(save_dir, 'DCA_3D_plot.pdf')
fig.savefig(plot_path3)
plt.close(fig)

# Save the third plot (5D) in PDF format
fig, ax = plt.subplots()
ax5 = plot_DCA_5D(ax, thresh_group, net_benefit_model5, net_benefit_all5)
plot_path5 = os.path.join(save_dir, 'DCA_5D_plot.pdf')
fig.savefig(plot_path5)
plt.close(fig)




















