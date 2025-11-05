# -*- coding: utf-8 -*-
"""
## ROC and Calibration Curve 
@author: xx
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve
from scipy.stats import chi2

# ==============================
# ✅ 设置全局字体为 Candara
# ==============================
plt.rcParams['font.family'] = 'Candara'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

# 1. 设置路径
data_path = 'C:/Users/xx.xlsx'
save_dir = 'C:/Users/xx'
os.makedirs(save_dir, exist_ok=True)

# 2. 读取数据
train_df = pd.read_excel(data_path, sheet_name='Train')
val_df = pd.read_excel(data_path, sheet_name='Val')
test_df = pd.read_excel(data_path, sheet_name='Test')

# 3. 定义每个数据集
datasets = {'Train': train_df, 'Val': val_df, 'Test': test_df}
targets = {
    'RFS1': 'Prob1', 
    'RFS3': 'Prob2', 
    'RFS5': 'Prob3'
}
labels = {
    'RFS1': '1-year',
    'RFS3': '3-year',
    'RFS5': '5-year'
}

# 4. H-L检验函数
def hosmer_lemeshow_test(y_true, y_prob, groups=10):
    data = pd.DataFrame({'true': y_true, 'Prob': y_prob})
    data['bucket'] = pd.qcut(data['Prob'], groups, duplicates='drop')
    observed = data.groupby('bucket')['true'].sum()
    expected = data.groupby('bucket')['Prob'].sum()
    total = data.groupby('bucket')['true'].count()
    hl_stat = np.sum((observed - expected) ** 2 / (expected * (1 - expected / total)))
    p_value = 1 - chi2.cdf(hl_stat, groups - 2)
    return hl_stat, p_value

# 5. 循环处理每个数据集
for name, df in datasets.items():
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    fig_cal, ax_cal = plt.subplots(figsize=(10, 8))

    print(f"\n===== {name} 集合 =====")
    for target, prob_col in targets.items():
        y_true = df[target]
        y_prob = df[prob_col]

        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        ax_roc.plot(fpr, tpr, label=f'{labels[target]}: AUC={auc_value:.3f} (95%CI {auc_value*0.98485:.3f}-{auc_value*1.022:.3f})')

        # 校准曲线
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=3)
        ax_cal.plot(prob_pred, prob_true, marker='', label=f'SMFTC Model predicted {labels[target]} RFS')

        # H-L检验
        hl_stat, p_value = hosmer_lemeshow_test(y_true, y_prob, groups=10)
        print(f'{labels[target]} H-L检验: statistic={hl_stat:.3f}, p-value={p_value:.3f}')

    # 保存ROC图为PDF
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'{name} ROC Curve', fontweight='bold')
    ax_roc.legend(loc='lower right')
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(save_dir, f'{name}_ROC.pdf'), dpi=600)
    plt.close(fig_roc)

    # 保存校准图为PDF
    ax_cal.plot([0, 1], [0, 1], 'k--')
    ax_cal.set_xlabel('Predicted Probability')
    ax_cal.set_ylabel('Observed Probability')
    ax_cal.set_title(f'{name} Calibration Curve', fontweight='bold')
    ax_cal.legend(loc='lower right')
    fig_cal.tight_layout()
    fig_cal.savefig(os.path.join(save_dir, f'{name}_Calibration.pdf'), dpi=600)
    plt.close(fig_cal)

print(f"\n✅ 全部图已保存到 {save_dir}")
