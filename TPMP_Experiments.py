
import os
# os.chdir()
import time
import numpy as np
import pandas as pd
import TPMP_ModelTraining as MT
import TPMP_Class as TPMP
import threading

# In[全域變數]:

strategies = ['Base', 'MaxBudget', 'MaxBudget_IneqWeighted', 'MinCover', 'MaxImprove', 'MaxImprove_Uion']
# 定價模式: 0原始定價Base, 1最高收益上限MaxBudget, 2最高收益上限(不等式參與加權)MaxBudget_W,
#          3最小覆蓋集MinCover, 4最大可改進收益MaxImprove, 5最大可改進收益(聯集)MaxImprove_Uion
dir_name = 'var'
date = ['1017'] # 目標的資料分組路徑
minK, datasetNum = 50, 10
thread_count = 1
# BFS: thread_count = 1, p2 = False
#   >> [Removing Effectiveness]
#   >> [pNormal disp=0.0] Runtime: 2229.1386778354645 sec.

# BFS: thread_count = 20, p2 = False
#   >> [Removing Effectiveness]
#   >> [pNormal disp=0.0] Runtime: 2861.1001303195953 sec.

# BFS: thread_count = 1, p2 = True
#   >> [Removing Effectiveness]
#   >> [pNormal disp=0.0] Runtime: 10744.449492454529 sec.

# DFS: thread_count = 1, p2 = True
#   >> [Removing Effectiveness]
#   >> [pNormal disp=0.0] Runtime: 8363.225188732147 sec.

effect_rows = ['Revenue', 'Revenue Ratio (%)', 'Runtime (s)', 'Quantity', 'Iterations', 'Distribution Error (%)', 'Runtime_P2 (s)', 'R2 Score', 'Avg. Distribution Error (%)']                
ineq_rows = ['AccMT', 'SetMT', 'SubAdd', 'SetMT_Reserved', 'SubAdd_Reserved']
dataset_rows = ['d'+str(di) for di in range(0, datasetNum)]+['CorrCoef']
acc_rows = ['0.'+str(di) for di in range(50, 100)]+['CorrCoef']
size_rows = list(np.arange(1, datasetNum+1))+['CorrCoef']
filtered_cols = ['CIP_Unfiltered', 'CIP_Filtered', 'SLSQP_Unfiltered', 'SLSQP_Filtered']
basePricing_cols = ['Trained_Models', 'Training_Time','CIP_R_Ratio', 'CIP_Runtime', 'CIP_Vio', 'SLSQP_R_Ratio', 'SLSQP_Runtime', 'SLSQP_Vio']

# In[]:

def RW_ExpDict(adict, wtire = False, dir_name = '', date = '', batch = ''):
    if wtire:
        for key, value in adict.items():
            MT.RW_ClassObj(obj = value, wtire = True, dir_name = dir_name, name = key, date = date, batch = batch)
    else:
        return {key: MT.RW_ClassObj(wtire = False, dir_name = dir_name, name = key, date = date, batch = batch) for key in adict}

def Reset_ExpDict():
    return {'records': np.zeros((9, total_round, total_Obj), float), # 定價與分配結果(CIP有無移除)
            'basePricing': np.zeros((len(k_list), 8, total_Obj), float), # 原始定價結果(CIP與求解器)
            'Ineq_distribution': np.zeros((5, total_round, total_Obj), float), # 不等式分布(CIP有無移除)
            'FilterRuntime': np.zeros((len(k_list), 4, total_Obj), float), # 過濾多餘不等式的效率改善(CIP與求解器)
            'SupportedSV': np.zeros((datasetNum+1, total_round, total_Obj), float), # 支持SV相關性
            'AccDistribution': np.zeros((51, total_round, total_Obj), float), # 模型準確度分布(0.5~0.99)
            'SizeDistribution': np.zeros((datasetNum+1, total_round, total_Obj), float), # 模型大小分布
            'Participation': np.zeros((datasetNum+1, total_round, total_Obj), float)} # 資料集參與分布

def threadingRun(smt, d, t, stk):
    global expdict, VioMap
    Pre_Work = MT.RW_ClassObj(dir_name = dir_name, name = 'ClassObj', date = date[d], batch = 'm'+str(t)) # 載入模型訓練物件
    for m_idx in range(0, len(k_list)):
        picked = k_list[m_idx]
        if thread_count == 1:
            print('\n'+setup, ' K=', picked, sep = '')
        res = MT.RunMS(ClassObj = Pre_Work, # SMT模型訓練
                       search = smt, # 0: 寬度優先走訪, 1: 深度優先走訪, 2: 完全隨機走訪
                       k = picked, # k: 輸出模型數, 0~1為百分比(0~100%), 1以上為個數
                       reboot = False, iterate = True)
        Twophase = TPMP.Twophase_Predictive_Model_Pricing(cmb = res['cmb'], acc = res['cmb_acc'],
                                                          lenGroup = res['cmb_len'],
                                                          utility = res['utility'],
                                                          marginal = res['marginal']) # 建立TPMP物件
        for rm in range(0, len(RmMode)):
            xOpt = TPMP.RunTPMP(ClassObj = Twophase,
                                RunMode = [1, RmMode[rm], p2, True],
                                ReadSurvey = True) # TPMP定價與分配
            col_idx = m_idx*len(RmMode)+rm
            expdict['records'][0, col_idx, t] = xOpt['r_total']
            expdict['records'][1, col_idx, t] = xOpt['r_ratio']
            expdict['records'][2, col_idx, t] = xOpt['runtime_P1']
            expdict['records'][3, col_idx, t] = len(xOpt['ML_price'])
            expdict['records'][4, col_idx, t] = xOpt['itr_t']
            expdict['Ineq_distribution'][:, col_idx, t] = xOpt['IneqNum']
            if p2:
                expdict['records'][5, col_idx, t] = xOpt['c_error']
                expdict['records'][6, col_idx, t] = xOpt['runtime_P2']
                expdict['records'][7, col_idx, t] = xOpt['R2Score']
                expdict['records'][8, col_idx, t] = xOpt['cAvgError']
            if rm == 0:
                expdict['basePricing'][m_idx, 0, t] = Pre_Work.trained_count
                expdict['basePricing'][m_idx, 1, t] = sum(Pre_Work.trainedTime)
                expdict['basePricing'][m_idx, 3, t] = xOpt['runtime_P1']
                for i in range(0, 2):
                    if i == 1:
                        xOpt = TPMP.RunTPMP(ClassObj = Twophase,
                                            RunMode = [0, 0, False, True],
                                            ReadSurvey = True) # TPMP定價與分配
                        expdict['basePricing'][m_idx, 6, t] = xOpt['runtime_P1']
                    sm_vioNum = Twophase.Ineq_Verify(sm = True, xOpt = xOpt['ML_price'],
                                                     verify = True, show = False) # 驗證訓練集單調性
                    if sm_vioNum > 0:
                        expdict['basePricing'][m_idx, 4+i*3, t] = 1
                        VioMap[t, i, stk] = 1
                    else:
                        sa_vioNum = Twophase.Ineq_Verify(sm = False, xOpt = xOpt['ML_price'],
                                                         verify = True, show = False) # 驗證訓練集次可加性
                        if sa_vioNum > 0:
                            expdict['basePricing'][m_idx, 4+i*3, t] = 1
                            VioMap[t, i, stk] = 1
                        else:
                            expdict['basePricing'][m_idx, 2+i*3, t] = xOpt['r_ratio']
                    if stk == 0:
                         expdict['FilterRuntime'][m_idx, 1+i*2, t] = xOpt['runtime_P1']
            if stk == 0:
                expdict['SupportedSV'][:datasetNum, col_idx, t] = np.sum(Twophase.ML_utility, axis = 0)
                expdict['AccDistribution'][:50, col_idx, t] = Twophase.Get_AccDistribution(stack = True)
                expdict['SizeDistribution'][:datasetNum, col_idx, t] = Twophase.Get_SizeDistribution()
                expdict['Participation'][:datasetNum, col_idx, t] = Twophase.Get_ParticipationCounts()
                if rm == 0:
                    xOpt = TPMP.RunTPMP(ClassObj = Twophase,
                                        RunMode = [1, 0, False, True],
                                        ReadSurvey = True,
                                        IneqFilter = False) # 未過濾多餘不等式的CIP定價
                    expdict['FilterRuntime'][m_idx, 0, t] = xOpt['runtime_P1']
                    xOpt = TPMP.RunTPMP(ClassObj = Twophase,
                                        RunMode = [0, 0, False, True],
                                        ReadSurvey = True,
                                        IneqFilter = False) # 未過濾多餘不等式的求解器定價
                    expdict['FilterRuntime'][m_idx, 2, t] = xOpt['runtime_P1']
            if thread_count == 1:
                print('  [', strategies[RmMode[rm]], '] done!', sep = '')
    baseMax_idx = total_round-len(RmMode)
    for col_idx in range(total_round):
        expdict['SupportedSV'][datasetNum, col_idx, t] = np.corrcoef(expdict['SupportedSV'][:datasetNum, col_idx, t], expdict['SupportedSV'][:datasetNum, baseMax_idx, t])[0][1]
        expdict['AccDistribution'][50, col_idx, t] = np.corrcoef(expdict['AccDistribution'][:50, col_idx, t], expdict['AccDistribution'][:50, baseMax_idx, t])[0][1]
        expdict['SizeDistribution'][datasetNum, col_idx, t] = np.corrcoef(expdict['SizeDistribution'][:datasetNum, col_idx, t], expdict['SizeDistribution'][:datasetNum, baseMax_idx, t])[0][1]
        expdict['Participation'][datasetNum, col_idx, t] = np.corrcoef(expdict['Participation'][:datasetNum, col_idx, t], expdict['Participation'][:datasetNum, baseMax_idx, t])[0][1]
    print(setup, ' ', d*batchObj+t+1, '-th done!', sep = '')

# In[實驗:模型定價與收益分配]:

if __name__ == "__main__":
    k_list = np.append(np.arange(minK, minK*5+1, step = minK), 1)
    RmMode = np.array([0, 3, 4])
    total_round = len(RmMode)*len(k_list)
    # normal = np.array(['pNormal', 'nNormal'])
    # dispMap = np.array([0.0, 0.3])
    normal = np.array(['pNormal']) # 需求函數 = 標準常態分布(正向)
    dispMap = np.array([0.0]) # 需求函數偏移量 = 0
    SMT_pair = {0: ' (BFS)', 1: ' (DFS)'} # 0為BFS, 1為DFS
    # SMT_pair = {0: ' (BFS)'}
    # SMT_pair = {1: ' (DFS)'}
    batchObj = 20
    total_Obj = len(date)*batchObj
    p2 = True # 要進行收益分配
    # p2 = False # 不進行收益分配
    VioMap = np.zeros((total_Obj, 2, 3), int) # 紀錄非最佳解的定價結果
    input_cols = []
    for k in k_list:
        k_str = 'MAX' if k == 1 else str(k)
        for s in RmMode:
            input_cols.append('K='+k_str+':'+strategies[s])
    expdict = Reset_ExpDict() # 實驗數據字典
    stk = 0
    for curve in normal:
        for disp in dispMap:
            for smt, search in SMT_pair.items():
                starttime = time.time()
                setup = '['+curve+' disp='+str(disp)+']'
                if stk > 0:
                    expdict['records'] = np.zeros((9, total_round, total_Obj), float)
                    expdict['basePricing'] = np.zeros((len(k_list), 8, total_Obj), float)
                for d in range(0, len(date)):
                    path_dir = dir_name+'/'+date[d]+'/'+setup
                    if not os.path.exists(path_dir):
                        os.makedirs(path_dir)
                    if not os.path.exists(path_dir+search):
                        os.makedirs(path_dir+search)
                    #expdict = RW_ExpDict(expdict, wtire = False, dir_name = dir_name, date = date[d], batch = setup+search) # 接續紀錄
                    for t in range(0, batchObj, thread_count):
                        if thread_count > 1:
                            threads = []
                            for i in range(thread_count): # 建立子執行緒
                                threads.append(threading.Thread(target=threadingRun, args=(smt, d, t+i, stk)))
                                threads[i].start()
                            for j in range(thread_count): # 等待所有子執行緒結束
                                threads[j].join()
                        else:
                            threadingRun(smt, d, t, stk)
                    records_sub = np.sum(expdict['records'], axis = 2)
                    RMdf = pd.DataFrame(records_sub/total_Obj, index = effect_rows, columns = input_cols)
                    fname = 'Optimizing Effectiveness Test'+search+'.csv'
                    RMdf.to_csv(path_dir+'/'+fname, index = True, header = True)

                    if stk == 0: # 以下屬性不受curve與disp變化而改變, 因此只需輸出一次
                        Ineq_distribution_sub = np.sum(expdict['Ineq_distribution'], axis = 2)
                        IDdf = pd.DataFrame(Ineq_distribution_sub/total_Obj, index = ineq_rows, columns = input_cols)
                        fname = 'Removed Inequalities Distribution Test'+search+'.csv'
                        IDdf.to_csv(path_dir+'/'+fname, index = True, header = True)
                        
                        SupportedSV_sub = np.sum(expdict['SupportedSV'], axis = 2)
                        SVdf = pd.DataFrame(SupportedSV_sub/total_Obj, index = dataset_rows, columns = input_cols)
                        fname = 'Supported SV Test'+search+'.csv'
                        SVdf.to_csv(path_dir+'/'+fname, index = True, header = True)
                
                        Participation_sub = np.sum(expdict['Participation'], axis = 2)
                        PPdf = pd.DataFrame(Participation_sub/total_Obj, index = dataset_rows, columns = input_cols)
                        fname = 'Participation Test'+search+'.csv'
                        PPdf.to_csv(path_dir+'/'+fname, index = True, header = True) 
                        
                        AccDistribution_sub = np.sum(expdict['AccDistribution'], axis = 2)
                        ADdf = pd.DataFrame(AccDistribution_sub/total_Obj, index = acc_rows, columns = input_cols)
                        fname = 'Accuracy Distribution Test'+search+'.csv'
                        ADdf.to_csv(path_dir+'/'+fname, index = True, header = True)
            
                        SizeDistribution_sub = np.sum(expdict['SizeDistribution'], axis = 2)
                        SDdf = pd.DataFrame(SizeDistribution_sub/total_Obj, index = size_rows, columns = input_cols)
                        fname = 'Size Distribution Test'+search+'.csv'
                        SDdf.to_csv(path_dir+'/'+fname, index = True, header = True)
            
                        FilterRuntime_sub = np.sum(expdict['FilterRuntime'], axis = 2)
                        FRdf = pd.DataFrame(FilterRuntime_sub/total_Obj, index = k_list, columns = filtered_cols)
                        fname = 'Filtered Efficiency Test'+search+'.csv'
                        FRdf.to_csv(path_dir+'/'+fname, index = True, header = True)
             
                    basePricing_sub = np.sum(expdict['basePricing'], axis = 2)
                    PCdf = pd.DataFrame(basePricing_sub/total_Obj, index = k_list, columns = basePricing_cols)
                    fname = 'Pricing Efficiency Test'+search+'.csv'
                    PCdf.to_csv(path_dir+'/'+fname, index = True, header = True)
    
                runtime = time.time()-starttime
                print('[Removing Effectiveness]\n', setup, ' Runtime: ', runtime, ' sec.\n', sep = '')
                RW_ExpDict(expdict, wtire = True, dir_name = dir_name, date = date[d], batch = setup+search)
            if curve == 'nNormal':
                break
            else:
                stk += 1