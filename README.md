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

![image](https://github.com/Magic8763/TPMP/blob/main/img/model_market.jpg)


註：本專案為[「收益最大化和分配最佳化之兩階段預測模型定價框架」](https://hdl.handle.net/11296/4w3p68)的實作程式碼。

## Presentation
**Arbitrage-Free Pricing**

定價階段: CIP

**Revenue Maximization**

**「模型價格必須滿足所有無套利約束條件才會是不可套利的」**，為了在此前提下最大化銷售收益，我們選擇將部分模型實例從銷售清單中移除，藉此減少約束條件對其餘模型價格的限制。我們提出兩種效率與效果兼具的貪婪方法，依據所屬約束條件對價格的限制程度來決定移除何者有助於改善預期收益。
- `MinCover`: 優先移除會形成較多約束條件的模型實例
- `MaxImprove`: 優先移除可改進收益較高的模型實例
  - 可改進收益 = 所屬約束條件造成的收益損失估計值 - 該模型本身價格
![image](https://github.com/Magic8763/TPMP/blob/main/img/revenue.jpg)
![image](https://github.com/Magic8763/TPMP/blob/main/img/revenue_ratio.jpg)

無論是`MinCover`或`MaxImprove`方法，在原始定價`Base`的收益百分比隨著給定模型數增加而逐漸降低的情況下，二者仍能將預期收益穩定維持在較高的狀態，其中又以`MaxImprove`更勝一籌。

**Fair Distribution**

分配階段: 支持SV

我們提出基於約束的迭代定價方法具有線性對數的時間複雜度與收益最大化的保證。
我們提出模型牴觸移除的概念及兩種實現概念的收益優化方法，二者均透過容許移除少量模型實例來達到提高收益的無套利定價，實驗結果顯示收益優化後都提升到最大可能收益的九成以上。
我們提出的支持夏普利值僅考慮參與者貢獻皆非負而且並非全為零的模型實例，據此分配回饋報酬的方法仍可滿足夏普利公平性的規範。

## Prerequisites
- Python3, NumPy, Pandas, Scikit-learn, Matplotlib

## Description
**Program**
- `TPMP_Build.py`: 資料前處理，包含資料樣本分組與訓練模型實例
- `TPMP_main.py`: 程式主介面，包含以模擬資料測試函數功能的範例實驗
- `TPMP_Experiments.py`: 完整的迭代定價實驗
- `TPMP_Preprocessing.py`: 前處理相關功能
- `TPMP_ModelTraining.py`: 支持 SV 類別，包含模型訓練與資料集效用評估函數
- `TPMP_Class.py`: TPMP 類別，包含無套利定價與收益分配函數
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
