
import os
# os.chdir()
import numpy as np
import TPMP_ModelTraining as MT
import TPMP_Class as TPMP
from copy import deepcopy as dcp

# In[模型參數設置]:

if __name__ == "__main__":
    mode = 'logistic' # 預設以logistic建立分類模型物件clf
    # mode = 'SVC' # 以SVC建立分類模型
    # mode = 'LinearSVC' # 以LinearSVC建立分類模型
    clf = MT.return_model(mode = mode, tol = 0.1, solver = 'liblinear', n_jobs = 1)

# In[取得訓練資料]:

    train_size = 100
    # x_raw, y_raw = MT.Data_generator(model = clf, train_size = train_size, test_size = 1000,
    #                                  target_accuracy = 0.8, randomSort = False, show = True,
    #                                  d = 10, difficulty = 1, important_dims = 1) # 產生模擬資料
    x_raw, y_raw = MT.Read_RawCSV(xfile = 'data/x_raw101.csv', yfile = 'data/y_raw101.csv') # 載入既有資料

# In[建立模型訓練物件]:

    group_size = 10
    cut = np.arange(start = 0, stop = train_size+1, step = train_size//group_size)[1:] # 資料集分割
    Pre_Work = MT.SupportedSV(x_raw = x_raw, y_raw = y_raw, size = train_size, model = clf, cut = cut, K = 1)
    # build_date = '1016'
    # dir_name = 'var'
    # Pre_Work = MT.RW_ClassObj(dir_name = dir_name, name = 'ClassObj',
    #                           date = build_date, batch = 'm0') # 載入個案物件

# In[版本模型訓練與篩選]:

    res = MT.RunMS(ClassObj = Pre_Work, search = 1, # 0: 寬度優先走訪, 1: 深度優先走訪, 2: 完全隨機走訪
                   k = 1, # 輸出模型數, 0~1為百分比(0~100%), 1以上為個數
                   reboot = True) # True: 重啟訓練過程, False: 僅重新篩選模型而非重啟訓練過程(先以參數 "k=1, reboot=True" 執行完畢後才能使用)
    # cmb, cmb_acc, cmb_len, cmb_utility, cmb_marginal, training_time = MT.RunMS(ClassObj = Pre_Work,
    #                                                                            search = 1, # 0: 寬度優先走訪, 1: 深度優先走訪, 2: 完全隨機走訪
    #                                                                            k = 1, # 輸出模型數, 0~1為百分比(0~100%), 1以上為個數
    #                                                                            reboot = True) # True: 重啟訓練過程, False: 僅重新篩選模型而非重啟訓練過程(先以參數 "k=1, reboot=True" 執行完畢後才能使用)
    # Using x_raw101, y_raw101:
    #   >> Trained Models: 1023, Output Models: 310
    #   >> [Model Selection] ML_310: 0.9780740737915039 sec.

# In[MT其他功能]:

    # dir_name = 'var'
    # fm_vals, fm_vals_uf, fm_acc_count = MT.RunSV(ClassObj = Pre_Work, getAC = True) # 計算原始SV
    # MT.RW_ClassObj(Pre_Work, wtire = True, dir_name = dir_name, name = 'ClassObj', date = 'today', batch = '0') # 輸出類別物件
    # Pre_Work = MT.RW_ClassObj(dir_name = dir_name, name = 'ClassObj', date = 'today', batch = '0') # 載入類別物件
    # MT.Write_RawCSV(ClassObj = Pre_Work, xfile = 'x_raw.csv', yfile = 'y_raw.csv') # 輸出訓練資料為.csv檔
    # MT.SurveyExample(budget_reverse = False, demand_reverse = False, disp = 0, style = 1, fontsize = 14) # 市調函數範例繪製

# In[建立TPMP物件]:

    Twophase = TPMP.Twophase_Predictive_Model_Pricing(cmb = res['cmb'], acc = res['cmb_acc'],
                                                      lenGroup = res['cmb_len'],
                                                      utility = res['utility'],
                                                      marginal = res['marginal'])
    Twophase_0, Twophase_3, Twophase_4 = dcp(Twophase), dcp(Twophase), dcp(Twophase)

# In[TPMP定價與分配]:

    # RunMode[定價方法, 定價模式, 有無收益分配, 是否為迭代進行(隱藏多餘的文字輸出)]
    xOpt = TPMP.RunTPMP(ClassObj = Twophase, RunMode = [0, 0, True, False], # SLSQP原始定價
                        irregular = False, ReadSurvey = True, plot = True)
    # Using x_raw101, y_raw101:
    #   >> Arbitrage-free Pricing: SLSQP
    #   >> Optimization problem: Optimization terminated successfully
    #   >> Price Loss = 115321.5800
    #   >> Revenue Loss = 148.8210
    #   >> Total_Revenue: 218.936 / 367.757 (59.533%)
    #   >> Pricing ML_310 P1: 23.898258447647095 sec.

    #   >> Distribution Optimization: SLSQP
    #   >> Error(EC, AC) = 7.6600
    #   >> Loss(EC, AC) = -0.0000
    #   >> Avg. Error(EC, AC) = 3.7420
    #   >> R2(EC, AC) = 0.9750
    #   >> Compensation Error: 7.66 / 218.936 (3.499%)
    #   >> Distribution ML310 P2: 84.18590521812439 sec.

    sm_vioNum = Twophase.Ineq_Verify(sm = True, xOpt = xOpt['ML_price'], verify = True, show = True) # 驗證訓練集單調性
    sa_vioNum = Twophase.Ineq_Verify(sm = False, xOpt = xOpt['ML_price'], verify = True, show = True) # 驗證訓練集次可加性

# In[CIP原始定價]:

    xOpt0 = TPMP.RunTPMP(ClassObj = Twophase_0, RunMode = [1, 0, True, False],
                         irregular = False, ReadSurvey = True, plot = True)
    # Using x_raw101, y_raw101:
    #   >> Arbitrage-free Pricing: CIP
    #   >> Iterations = 2
    #   >> Price Loss = 115321.5800
    #   >> Revenue Loss = 148.8210
    #   >> Total_Revenue: 218.936 / 367.757 (59.533%)
    #   >> Pricing ML310 P1: 0.907768964767456 sec.
      
    #   >> Distribution Optimization: SLSQP
    #   >> Error(EC, AC) = 7.6600
    #   >> Loss(EC, AC) = -0.0000
    #   >> Avg. Error(EC, AC) = 3.7420
    #   >> R2(EC, AC) = 0.9750
    #   >> Compensation Error: 7.66 / 218.936 (3.499%)
    #   >> Distribution ML310 P2: 84.58517169952393 sec.

    sm_vioNum_0 = Twophase_0.Ineq_Verify(sm = True, xOpt = xOpt0['ML_price'], verify = True, show = True) # 驗證訓練集單調性
    sa_vioNum_0 = Twophase_0.Ineq_Verify(sm = False, xOpt = xOpt0['ML_price'], verify = True, show = True) # 驗證訓練集次可加性

# In[CIP定價(MinCover)]:

    xOpt3 = TPMP.RunTPMP(ClassObj = Twophase_3, RunMode = [1, 3, True, False],
                         irregular = False, ReadSurvey = True, plot = True)
    # Using x_raw101, y_raw101:
    #   >> Arbitrage-free Pricing: CIP
    #   >> Iterations = 2
    #   >> Price Loss = 303.4500
    #   >> Revenue Loss = 24.2250
    #   >> Total_Revenue: 343.532 / 367.757 (93.413%)
    #   >> Pricing ML260 P1: 1.5309679508209229 sec.

    #   >> Distribution Optimization: SLSQP
    #   >> Error(EC, AC) = 35.4210
    #   >> Loss(EC, AC) = -0.0000
    #   >> Avg. Error(EC, AC) = 19.6820
    #   >> R2(EC, AC) = 0.9160
    #   >> Compensation Error: 35.421 / 343.532 (10.311%)
    #   >> Distribution ML260 P2: 46.242607831954956 sec.

    sm_vioNum_3 = Twophase_3.Ineq_Verify(sm = True, xOpt = xOpt3['ML_price'], verify = True, show = True) # 驗證訓練集單調性
    sa_vioNum_3 = Twophase_3.Ineq_Verify(sm = False, xOpt = xOpt3['ML_price'], verify = True, show = True) # 驗證訓練集次可加性

# In[CIP定價(MaxImprove)]:

    xOpt4 = TPMP.RunTPMP(ClassObj = Twophase_4, RunMode = [1, 4, True, False],
                         irregular = False, ReadSurvey = True, plot = True)
    # Using x_raw101, y_raw101:
    #   >> Arbitrage-free Pricing: CIP
    #   >> Iterations = 2
    #   >> Price Loss = 501.5000
    #   >> Revenue Loss = 0.9650
    #   >> Total_Revenue: 366.792 / 367.757 (99.738%)
    #   >> Pricing ML257 P1: 1.5352590084075928 sec.
    
    # >> Distribution Optimization: SLSQP
    # >> Error(EC, AC) = 26.2950
    # >> Loss(EC, AC) = -0.0000
    # >> Avg. Error(EC, AC) = 8.4300
    # >> R2(EC, AC) = 0.9620
    # >> Compensation Error: 26.295 / 366.792 (7.169%)
    # >> Distribution ML257 P2: 42.75877237319946 sec.

    sm_vioNum_4 = Twophase_4.Ineq_Verify(sm = True, xOpt = xOpt4['ML_price'], verify = True, show = True) # 驗證訓練集單調性
    sa_vioNum_4 = Twophase_4.Ineq_Verify(sm = False, xOpt = xOpt4['ML_price'], verify = True, show = True) # 驗證訓練集次可加性

# In[定價結果繪圖]:

    Twophase.PlotCurve(xOpt = [xOpt0['ML_price_full'], xOpt3['ML_price_full'], xOpt4['ML_price_full']], RmMode = [0, 3, 4], fontsize = 14)
