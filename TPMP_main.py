
import os
# os.chdir()
import numpy as np
import TPMP_ModelTraining as MT
import TPMP_Class as TPMP

# In[模型參數設置]:

if __name__ == "__main__":
    mode = 'logistic' # 以logistic建立分類模型
    # mode = 'SVC' # 以SVC建立分類模型
    # mode = 'LinearSVC' # 以LinearSVC建立分類模型
    clf = MT.return_model(mode = mode, tol = 0.1, solver = 'liblinear', n_jobs = 1) # 以logistic建立分類模型物件clf
    build_date = '1010'

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
    # dir_name = 'var'
    # Pre_Work = MT.RW_ClassObj(dir_name = dir_name, name = 'ClassObj',
    #                           date = build_date, batch = 'm0') # 載入個案物件

# In[版本模型訓練與篩選]:

    cmb, cmb_acc, cmb_len, cmb_utility, cmb_marginal, training_time = MT.RunMS(ClassObj = Pre_Work,
                                                                               search = 1, # 0: 寬度優先走訪, 1: 深度優先走訪, 2: 完全隨機走訪
                                                                               k = 1, # 輸出模型數, 0~1為百分比(0~100%), 1以上為個數
                                                                               reboot = True) # True: 重啟訓練過程, False: 僅重新篩選模型而非重啟訓練過程(先以參數 "k=1, reboot=True" 執行完畢後才能使用)
    cmb_SV = np.sum(cmb_utility, axis = 0) # 邊際貢獻矩陣utility的行總和 = 版本模型的支持SV

# In[MT其他功能]:

    # dir_name = 'var'
    # fm_vals, fm_vals_uf, fm_acc_count = MT.RunSV(ClassObj = Pre_Work, getAC = True) # 計算原始SV
    # MT.RW_ClassObj(Pre_Work, wtire = True, dir_name = dir_name, name = 'ClassObj', date = 'today', batch = '0') # 輸出類別物件
    # Pre_Work = MT.RW_ClassObj(dir_name = dir_name, name = 'ClassObj', date = 'today', batch = '0') # 載入類別物件
    # MT.Write_RawCSV(ClassObj = Pre_Work, xfile = 'x_raw.csv', yfile = 'y_raw.csv') # 輸出訓練資料為.csv檔
    # MT.SurveyExample(budget_reverse = False, demand_reverse = False, disp = 0, style = 1, fontsize = 14) # 市調函數範例繪製

# In[建立TPMP物件]:

    Twophase = TPMP.Twophase_Predictive_Model_Pricing(cmb = cmb, acc = cmb_acc,
                                                      lenGroup = cmb_len,
                                                      utility = cmb_utility,
                                                      marginal = cmb_marginal)

# In[TPMP定價與分配]:

    xOpt0, ML_price0 = TPMP.RunTPMP(ClassObj = Twophase, RunMode = [1, 0, False, False], irregular = False, ReadSurvey = True, plot = True)
    # xOpt3, ML_price3  = TPMP.RunTPMP(ClassObj = Twophase, RunMode = [1, 3, False, False], irregular = False, ReadSurvey = True, plot = True)
    # xOpt4, ML_price4  = TPMP.RunTPMP(ClassObj = Twophase, RunMode = [1, 4, False, False], irregular = False, ReadSurvey = True, plot = True)
    # Twophase.PlotCurve(xOpt = [ML_price0, ML_price3, ML_price4], RmMode = [0, 3, 4], fontsize = 14)


# In[無套利不等式驗證]:

    sm_vioNum = Twophase.Ineq_Verify(sm = True, xOpt = xOpt0, verify = True, show = True) # 驗證訓練集單調性
    sa_vioNum = Twophase.Ineq_Verify(sm = False, xOpt = xOpt0, verify = True, show = True) # 驗證訓練集次可加性
