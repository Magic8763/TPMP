![image](https://github.com/Magic8763/TPMP/assets/114419914/33ad1363-eecd-470a-b136-0f6f0c2659c0)# Two-phase Predictive Model Pricing
![](https://img.shields.io/github/stars/magic8763/TPMP)
![](https://img.shields.io/github/watchers/magic8763/TPMP)
![](https://img.shields.io/github/forks/magic8763/TPMP)
![shields](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

由於各行各業對於 AI 應用的需求與日俱增，以預訓練模型提供即時 AI 服務的企業平台因其高度的便利性，比起販售資料集的傳統資料市集更受市場用戶喜愛，但前者普遍仍採用訂閱制或依次付費等簡易的計價方案。若模型商品的性能遠高於用戶的期望，將迫使用戶付出高於其預算的金額或是放棄購買。同時銷售方也會受簡單的定價機制限制而無法進一步提高收益，甚至處在可能被套利 (Arbitrage) 的風險中。

[先前的研究](https://lchen001.github.io/papers/2019_Nimbus_SIGMOD.pdf)提出以預訓練模型作為商品販售而非資料集的資料市集變體，稱為模型市集。希望從買賣雙方及市場仲介商這三個角度研究如何防止套利同時最大化銷售收益。

有鑑於上述挑戰，我們提出以模型準確度與訓練集為依據的兩階段預測模型定價框架 TPMP。第一階段計算在不違反套利的條件下能最大化銷售收益的模型價格；第二階段以售出模型之訓練集為依據，分配給所屬資料提供者公平的回饋報酬。

<p align="center"><img src="https://github.com/Magic8763/TPMP/blob/main/img/model_market.jpg"></p>

註：本專案為[「收益最大化和分配最佳化之兩階段預測模型定價框架」](https://hdl.handle.net/11296/4w3p68)的實作程式碼。

## Presentation
**Preparation: Model Training**

採用羅吉斯回歸分類器 (Logistic Regression) 作為預設的建模演算法。

在固定模型參數的情況下，將數個資料集以聯集組合作為模型實例的訓練集，並在訓練過程中同步計算這些資料集個自的貢獻程度 (Utility)。

訓練完畢的模型實例將作為合法商品加入商品清單，但符合以下兩點的模型實例將被排除，因為它們會破壞收益分配的公平性。
1. 訓練集包含負貢獻資料集
2. 訓練集所含資料集的貢獻皆為零

**Phase 1: Arbitrage-Free Pricing**

為了避免模型市集發生套利行為，我們假設模型實例的價格必須滿足基於下列屬性設置的約束條件才能維持市場交易的公平性，這些屬性統稱為無套利性 (*Arbitrage-freeness*)
1. 非負性：模型價格不得為負
2. 單調性：價值相同的模型具有相等的價格，而價值較高的模型價格不得低於價值較低者
3. 次可加性：若兩個低價值的模型進行組合可以得到另一個高價值的模型，前二者的價格加總不得低於後者

**「模型價格必須滿足所有無套利約束條件才會是不可套利的」**。每當銷售方有新模型被加入商品清單時，由該模型形成的約束條件將一併套用在所有同樣作為待售商品的模型實例上，例如：準確度較高的模型之價格不得低於準確度較低者；或依據訓練集的組成關係，約束由較多資料訓練而成的模型之價格不得低於資料量較少者。這些約束條件會迫使部分模型調降其售價以避免買方的套利行為，而據此計算具有最大預期收益之模型價格的定價方式稱為**無套利定價** (*Arbitrage-Free Pricing*)
。

TPMP 採用基於個別模型所屬約束條件計算模型價格的貪婪方法`CIP`，透過從價格上限開始迭代調降模型價格的定價方式，找出符合約束條件的最高售價。

<p align="center"><img src="https://github.com/Magic8763/TPMP/blob/main/img/arbitrage-free_pricing.jpg"></p>

與採用循序最小平方規劃法`SLSQP`的非線性規劃求解器相比，兩者具有相同的無套利價格最佳解，但`CIP`有著將近兩個數量級的計算效率優勢。

**Phase 1+: Revenue Maximization**

為了在滿足無套利約束的前提下最大化銷售收益，我們選擇將部分模型實例從商品清單中移除，藉此減少約束條件對剩餘模型的價格限制。

TPMP 採用兩種效率與效果兼具的貪婪方法，依據所屬約束條件對價格的限制程度來決定移除何者有助於改善預期收益。
- `MinCover`：優先移除會形成較多約束條件的模型實例
- `MaxImprove`：優先移除可改進收益較高的模型實例
  - 可改進收益 = 由模型形成之約束條件造成的收益損失估計值 – 該模型的價格上限

<p align="center"><img src="https://github.com/Magic8763/TPMP/blob/main/img/revenue_maximization.jpg"></p>

如圖所示，在原始定價結果`Base`的收益百分比隨著給定模型數增加而逐漸降低的情況下，上述二方法仍能將預期收益穩定維持在較高的狀態，其中又以`MaxImprove`更勝一籌。

**Phase 2: Fair Distribution**

在訓練階段產生模型實例時一併求得的資料集個別貢獻將作為模型售出後的收益分配依據，其定義為**支持夏普利值** (*Supported Shapley value*)。與原始的夏普利值 (*Shapley value*) 不同，支持夏普利值僅考慮通過篩選而輸出作為商品的模型實例。

TPMP 採用基於支持夏普利值等比分配銷售收益的方式，我們在論文裡證明了據此分配的回饋報酬仍能滿足夏普利值保證的所有公平特性，並藉由排除負貢獻與全零貢獻等特例確保這種分配方式是可執行的，解決了夏普利值無法公平地為它們分配收益的實務問題。

## Prerequisites
- Python3, NumPy, Pandas, Scikit-learn, Matplotlib

## Description
**Program**
- `TPMP_Preprocessing.py`: 資料前處理函數
- `TPMP_ModelTraining.py`: 支持夏普利值類別，包含模型訓練與資料集效用評估的相關功能
- `TPMP_Build.py`: 前處理與模型訓練的執行流程
- `TPMP_Class.py`: TPMP 類別，包含無套利定價與收益分配的相關功能
- `TPMP_main.py`: 以模擬資料測試函數功能的範例實驗
- `TPMP_Experiments.py`: 完整的兩階段定價實驗
- `TPMP_Plot.py`: 實驗結果繪圖

**Variable**
- `V_acc.pkl`: 預設的市場調查結果-模型價格上限
- `D_acc.pkl`: 預設的市場調查結果-市場需求分布

## Dataset
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 圖片資料集 (Python version, 163 MB)
  - 其包含 10 種類別各 5000 筆訓練樣本和 1000 筆測試樣本
  - 每個樣本為 32x32 的彩色圖片，即每張圖片具有 3072 個特徵維度
- `x_raw101.csv`, `y_raw101.csv`
  - 測試功能的模擬資料

## Reference
- [Data Shapley: Equitable Valuation of Data for Machine Learning](https://github.com/amiratag/DataShapley)

## Authors
- **[Magic8763](https://github.com/Magic8763)**

## License
This project is licensed under the [MIT License](https://github.com/Magic8763/TPMP/blob/main/LICENSE)
