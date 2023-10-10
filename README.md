# Two-phase Predictive Model Pricing
A Framework of Predictive Model Pricing.

Code for implementation of "A Framework of Two-phase Predictive Model Pricing for Revenue Maximization and Distribution Optimization".

## Prerequisites
- Python3, NumPy, Pandas, Scikit-learn, Matplotlib, Pickle

## File Description
**Program**
- TPMP_Build.py: 資料前處理，包含資料樣本分組與訓練模型實例
- TPMP_main.py: 程式主介面，以範例實驗測試函數功能是否正常
- TPMP_Experiments.py: 完整的迭代定價實驗
- TPMP_Preprocessing.py: 前處理相關功能
- TPMP_ModelTraining.py: 模型訓練相關功能
- TPMP_Class.py: TPMP類別與函數
- TPMP_ResultPlot.py: 繪製實驗結果

**Variable**
- V_acc.pkl: 預設的市場調查結果(價格上限函數)
- D_acc.pkl: 預設的市場調查結果(需求分布函數)

## Dataset
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 圖片資料集 (python version, 163 MB)
  - 其包含10種類別各5000筆訓練樣本和1000筆測試樣本
  - 每個樣本為32x32的彩色圖片，即每張圖片具有3072個特徵維度
- x_raw101.csv, y_raw101.csv
  - 測試功能的模擬資料

## Authors
* **Chih-Chien Cheng** - (categoryv@cycu.org.tw)
