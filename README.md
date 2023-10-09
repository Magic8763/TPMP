Two-phase Predictive Model Pricing
=====================================
A Framework of Predictive Model Pricing.

Code for implementation of "A Framework of Two-phase Predictive Model Pricing for Revenue Maximization and Distribution Optimization".

## Prerequisites
- Python3, NumPy, Pandas, Scikit-learn, Matplotlib, Pickle

## File Description

**Program**
- TPMP_main.py: 程式主介面，包含各函數的單獨執行功能與完整的迭代定價實驗
- TPMP_Preprocessing.py: 資料分組前處理
- TPMP_ModelTraining.py: 模型訓練相關功能
- TPMP_Class.py: TPMP類別與函數
- TPMP_ResultPlot.py: 繪製迭代實驗的輸出結果

**Variable**
- V_acc.pkl: 預設的市場調查結果(價格上限函數)
- D_acc.pkl: 預設的市場調查結果(需求分布函數)

**Data**
- x_raw101.csv, y_raw101.csv: 測試功能的模擬資料

## Dataset
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)圖片資料集
  - 包含10個類別各5000筆訓練樣本和1000筆測試樣本
  - 每個樣本為32x32的彩色圖片，即每張圖片具有3072個特徵維度

## Authors
* **Chih-Chien Cheng** - (categoryv@cycu.org.tw)
