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
**Preparation: Model Training**

在固定模型參數的情況下，將數個資料集以聯集組合作為模型實例的訓練集，並在訓練過程中同步計算這些資料集個自的貢獻程度 (utility)。

訓練完畢的模型實例將作為合法商品加入商品清單，但符合以下兩點的模型實例將被排除，因為它們會破壞收益分配的公平性。
1. 訓練集包含負貢獻資料集
2. 訓練集所含資料集的貢獻皆為零

**Phase 1: Arbitrage-Free Pricing**

**「模型價格必須滿足所有無套利約束條件才會是不可套利的」**。每當銷售方有新模型被加入商品清單時，由該模型形成的約束條件將一併套用在所有同樣作為待售商品的模型實例上，例如：準確度較高的模型之價格不得低於準確度較低者；或依據訓練集的組成關係，約束由較多資料訓練而成的模型之價格不得低於資料量較少者。

這些約束條件會迫使部分模型調降其售價以避免買方的套利行為，而在滿足無套利約束的前提下計算具有最大預期收益之模型價格的定價方式稱為**無套利定價**。

TPMP 採用基於個別模型所屬約束條件計算模型價格的貪婪方法`CIP`，透過從價格上限開始迭代調降模型價格的定價方式，找出符合約束條件的最高售價。

![image](https://github.com/Magic8763/TPMP/blob/main/img/arbitrage-free_pricing.jpg)

如圖所示，與採用循序最小平方規劃法`SLSQP`的非線性規劃求解器相比，`CIP`能有效計算出滿足無套利約束的模型價格最佳解，同時它的計算效率又比`SLSQP`快了將近兩個數量級。

**Phase 1+: Revenue Maximization**

為了在滿足無套利約束的前提下最大化銷售收益，我們選擇將部分模型實例從商品清單中移除，藉此減少約束條件對剩餘模型的價格限制。

TPMP 採用兩種效率與效果兼具的貪婪方法，依據所屬約束條件對價格的限制程度來決定移除何者有助於改善預期收益。
- `MinCover`: 優先移除會形成較多約束條件的模型實例
- `MaxImprove`: 優先移除可改進收益較高的模型實例
  - 可改進收益 = 所屬約束條件造成的收益損失估計值 - 該模型本身價格

![image](https://github.com/Magic8763/TPMP/blob/main/img/revenue_maximization.jpg)

無論是`MinCover`或`MaxImprove`方法，在原始定價`Base`的收益百分比隨著給定模型數增加而逐漸降低的情況下，二者仍能將預期收益穩定維持在較高的狀態，其中又以`MaxImprove`更勝一籌。

**Phase 2: Fair Distribution**

在訓練階段產生模型實例時一併求得的資料集個別貢獻將作為模型售出後的收益分配依據，我們將其定義為**支持夏普利值** (*Supported Shapley value*)。與原始的夏普利值 (*Shapley value*) 不同，支持夏普利值僅考慮通過篩選而輸出作為商品的模型實例。

我們在論文裡證明了依據支持夏普利值等比分配的回饋報酬仍能滿足夏普利值保證的所有公平特性，並藉由排除負貢獻與全零貢獻等特例確保這種分配方式是可執行的，解決了夏普利值無法公平地為它們分配收益的實務問題。

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
