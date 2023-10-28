# Two-phase Predictive Model Pricing
![](https://img.shields.io/github/stars/magic8763/TPMP)
![](https://img.shields.io/github/watchers/magic8763/TPMP)
![](https://img.shields.io/github/forks/magic8763/TPMP)
![shields](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

用戶

從傳統資料市集上購買資料集來訓練預測模型的的效率遠遠不及將預訓練模型作為商品直接販售的模型市集，但後者普遍採用訂閱制或依次付費等簡易的定價方案，導致銷售方普遍受買方的套利行為所害而

無法進一步提高收益

現今由各家平台提供的AI模型服務仍以訂閱制或依次付費等簡易的定價方案為主

由於許多行業紛紛採用機器學習方法分析數據，各式應用服務平台為此提供資料集或演算法以便用戶訓練預測模型，但是計價普遍仍採用訂閱制或依次付費等簡易方案，這使得用戶為了獲得符合期待的模型須付出龐大的時間與金錢成本，漸漸導致買方購買意願低落也使銷售方不易獲利，甚至處在被套利的風險中。



其中我們關注將預訓練模型作為商品販售的模型市集，提出以模型準確度與訓練集為依據的兩階段預測模型定價框架。第一階段計算在不違反套利的條件下能最大化銷售收益的模型價格；第二階段則依據售出模型之訓練集分配給資料提供者公平的回饋報酬。
![image](https://github.com/Magic8763/TPMP/blob/main/img/模型市集角色互動.jpg)


註：本專案為[「收益最大化和分配最佳化之兩階段預測模型定價框架」](https://hdl.handle.net/11296/4w3p68)的實作程式碼。

## Presentation

我們提出基於約束的迭代定價方法具有線性對數的時間複雜度與收益最大化的保證。
我們提出模型牴觸移除的概念及兩種實現概念的收益優化方法，二者均透過容許移除少量模型實例來達到提高收益的無套利定價，實驗結果顯示收益優化後都提升到最大可能收益的九成以上。
我們提出的支持夏普利值僅考慮參與者貢獻皆非負而且並非全為零的模型實例，據此分配回饋報酬的方法仍可滿足夏普利公平性的規範。

1. 定價階段: CIP
2. 定價階段: 移除特色
3. 分配階段: 支持SV

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
