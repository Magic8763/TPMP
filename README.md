# Two-phase Predictive Model Pricing
![](https://img.shields.io/github/stars/magic8763/TPMP)
![](https://img.shields.io/github/watchers/magic8763/TPMP)
![](https://img.shields.io/github/forks/magic8763/TPMP)
![shields](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

隨著各行業對 AI 應用的需求與日俱增，以預訓練模型提供即時 AI 服務的企業平台因其高度的便利性，比起販售資料集的傳統資料市集更受市場歡迎。然而，這些平台普遍仍採用簡單的計價方式，如訂閱制或按次支付。若模型性能超出用戶預期，將迫使用戶支付超出預算的金額或放棄購買。同時，簡單的定價機制限制了銷售方提高收益的能力，甚至存在套利風險 (*Arbitrage*)。

為了克服這些挑戰，[先前的研究](https://lchen001.github.io/papers/2019_Nimbus_SIGMOD.pdf)提出以預訓練模型作為商品販售的模型市集概念，從買賣雙方及市場仲介商這三個角度研究如何防止套利並最大化銷售收益。在此背景下，我們提出以模型準確度和訓練集為依據的兩階段預測模型定價框架 TPMP。第一階段基於模型準確度和訓練集計算能防止套利並最大化銷售收益的模型價格。第二階段則基於售出模型的訓練集效用，公平地分配回饋報酬給資料提供者。

<p align="center"><img src="https://github.com/Magic8763/TPMP/blob/main/img/model_market.jpg"></p>

TPMP 的特色如下：
1. 大幅縮短定價時間：TPMP 的定價效率比使用非線性規劃求解器的定價方式快上兩個數量級
2. 極大化預期收益：TPMP 的定價收益可達到最大值的九成以上
3. 公平收益分配：採用客觀公平且可執行的方式分配銷售利潤

#本專案為[「收益最大化和分配最佳化之兩階段預測模型定價框架」](https://hdl.handle.net/11296/4w3p68)的實作程式碼。

#本論文已正式發表於[第27屆人工智慧與應用研討會 (TAAI2022)](https://taai2022.github.io/dprogram.html)。

## Presentation
**Preparation: Model Training**

使用羅吉斯迴歸分類器 (Logistic Regression) 作為預設建模演算法。

在固定模型參數的情況下，將數個資料集聯集為模型實例的訓練集，同時計算各資料集的貢獻程度 (Utility)。

完成訓練的模型實例將被列入商品清單，但為了確保收益分配的公平性，任何符合以下兩條件之一的模型將被排除：
1. 訓練集包含負貢獻資料集
2. 訓練集所含資料集的貢獻皆為零

**Phase 1: Arbitrage-Free Pricing**

為確保模型市集不發生套利，我們假設模型價格需滿足以下無套利性 (*Arbitrage-freeness*) 約束條件
1. 非負性：模型價格不能為負
2. 單調性：價值相同的模型需有相等的價格，價值較高的模型價格不得低於價值較低者
3. 次可加性：若組合兩低價值模型可得一高價值模型，則前者價格總和不得低於後者

**「所有無套利約束條件須得到滿足，方為不可套利」**。每當有新模型被加入商品清單，其相關約束條件將同步套用在所有待售模型上。例如，高準確度模型價格不得低於低準確度者，或基於資料訓練量的約束條件，較多資料訓練的模型價格不得低於資料較少者。這迫使一些模型降價以防止買方套利，據此計算模型價格的定價方式稱為**無套利定價** (*Arbitrage-Free Pricing*)。

TPMP 採用基於約束的迭代定價方法`CIP`，根據個別模型的約束條件，從價格上限開始迭代降低，找出符合約束條件的最高售價。

<p align="center"><img src="https://github.com/Magic8763/TPMP/blob/main/img/arbitrage-free_pricing.jpg"></p>

與採用循序最小平方規劃法`SLSQP`的非線性規劃求解器相比，兩者具有相同的無套利價格最佳解，但`CIP`有著高達兩個數量級的計算效率優勢。

**Phase 1+: Revenue Maximization**

為了在滿足無套利約束的前提下最大化銷售收益，我們選擇移除將部分待售模型，藉此減少對剩餘模型的約束條件。

TPMP 採用兩種兼顧效率與效果的貪婪方法，依據約束條件對價格的限制程度選擇移除那些模型有助於改善預期收益。
- `MinCover`：優先移除形成較多約束條件的模型實例
- `MaxImprove`：優先移除可改進收益較高的模型實例
  - 可改進收益 = 由模型形成之約束條件造成的收益損失估計值 – 該模型的價格上限

<p align="center"><img src="https://github.com/Magic8763/TPMP/blob/main/img/revenue_maximization.jpg"></p>

如圖所示，即使原始定價收益`Base`的收益百分比隨模型數增加而降低，上述方法仍能將預期收益維持在較高的狀態，其中以`MaxImprove`效果更為優越。

**Phase 2: Fair Distribution**

在訓練階段同步計算的資料集個別貢獻將作為模型售出後的收益分配依據，其定義為**支持夏普利值** (*Supported Shapley value*)。與原始的夏普利值 (*Shapley value*) 不同，支持夏普利值僅考慮通過篩選並作為商品售出的模型實例。

TPMP 採用基於支持夏普利值的等比分配方式來分配銷售利潤，我們在論文中證明了這種分配方式滿足夏普利值保證的所有公平特性。同時，支持夏普利值排除了負貢獻和全零貢獻等特殊情況，確保這種分配方式的可行性，從而解決了夏普利值無法公平分配收益的實務問題。

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

#建議使用 **Spyder IDE** 逐段運行

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
