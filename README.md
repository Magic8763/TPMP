# Two-phase Predictive Model Pricing
![](https://img.shields.io/github/stars/magic8763/TPMP)
![](https://img.shields.io/github/watchers/magic8763/TPMP)
![](https://img.shields.io/github/forks/magic8763/TPMP)
![shields](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

[「收益最大化和分配最佳化之兩階段預測模型定價框架」](https://hdl.handle.net/11296/4w3p68)的實作程式碼。






## Presentation


## Prerequisites
- Python3, NumPy, Pandas, Scikit-learn, Matplotlib

## Description
**Program**
- `TPMP_Build.py`: 資料前處理，包含資料樣本分組與訓練模型實例
- `TPMP_main.py`: 程式主介面，包含以模擬資料測試函數功能的範例實驗
- `TPMP_Experiments.py`: 完整的迭代定價實驗
- `TPMP_Preprocessing.py`: 前處理相關功能
- `TPMP_ModelTraining.py`: 模型訓練相關功能
- `TPMP_Class.py`: TPMP 類別與函數
- `TPMP_Plot.py`: 實驗結果繪圖

**Variable**
- `V_acc.pkl`: 預設的市場調查結果-模型價格上限
- `D_acc.pkl`: 預設的市場調查結果-市場需求分布

## Dataset
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 圖片資料集 (Python version, 163 MB)
  - 其包含10種類別各5000筆訓練樣本和1000筆測試樣本
  - 每個樣本為32x32的彩色圖片，即每張圖片具有3072個特徵維度
- `x_raw101.csv`, `y_raw101.csv`
  - 測試功能的模擬資料

## Reference
- [Data Shapley: Equitable Valuation of Data for Machine Learning](https://github.com/amiratag/DataShapley)

## Authors
- **[Magic8763](https://github.com/Magic8763)**

## License
This project is licensed under the [MIT License](https://github.com/Magic8763/TPMP/blob/main/LICENSE)
