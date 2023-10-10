
import os
# os.chdir()
import time
import numpy as np
import pandas as pd
import TPMP_ModelTraining as MT
import TPMP_Class as TPMP

# In[實驗:模型定價與收益分配]:

if __name__ == "__main__":
    dir_name = 'var'
    date = ['1010']
    k_list = np.append(np.arange(50, 251, step = 50), 1)
    # RmMode定價模式strategy: 0原始定價Base, 1最高收益上限MaxBudget, 2最高收益上限(不等式參與加權)MaxBudget_W, 3最小覆蓋集MinCover
    #                        4最大可改進收益MaxImprove, 5最大可改進收益(聯集)MaxImprove_Uion
    RmMode = np.array([0, 3, 4])
    strategy = ['Base', 'MinCover', 'MaxImprove']
    # normal = np.array(['pNormal', 'nNormal'])
    # dispMap = np.array([0.0, 0.3])
    normal = np.array(['pNormal']) # 需求函數 = 標準常態分布(正向)
    dispMap = np.array([0.0]) # 需求函數偏移量 = 0
    SMT_search = 0 # 0為BFS, 1為DFS
    batchObj = 20
    p2 = False # 是否要進行收益分配
    total = len(date)*batchObj
    VioMap = np.zeros((total, 2, 3), int) # 非最佳解的定價結果紀錄
    stk = 0
    search = '(BFS)' if SMT_search == 0 else '(DFS)'
    for curve in normal:
        for disp in dispMap:
            starttime = time.time()
            setup = '[' + curve + ' disp=' + str(disp) + '] '
            records = np.zeros((9, len(RmMode)*len(k_list), total), float) # 定價與分配結果(CIP有無移除)
            basePricing = np.zeros((len(k_list), 8, total), float) # 原始定價結果(CIP與求解器)
            if stk == 0:
                Ineq_distribution = np.zeros((5, len(RmMode)*len(k_list), total), float) # 不等式分布(CIP有無移除)
                FilterRuntime = np.zeros((len(k_list), 4, total), float) # 過濾多餘不等式的效率改善(CIP與求解器)
                SupportedSV = np.zeros((11, len(RmMode)*len(k_list)+1, total), float) # 支持SV相關性
                AccDistribution = np.zeros((51, len(RmMode)*len(k_list)+1, total), float) # 模型準確度分布
                SizeDistribution = np.zeros((11, len(RmMode)*len(k_list)+1, total), float) # 模型大小分布
                Participation = np.zeros((11, len(RmMode)*len(k_list)+1, total), float) # 資料集參與分布
            for d in range(0, len(date)):
                # records = MT.RW_ClassObj(dirN = dir_name, name = 'records', date = date[d], batch = setup+search) # 接續紀錄
                # basePricing = MT.RW_ClassObj(dirN = dir_name, name = 'basePricing', date = date[d], batch = setup+search)
                # Ineq_distribution = MT.RW_ClassObj(dirN = dir_name, name = 'Ineq_distribution', date = date[d], batch = search+search)
                # FilterRuntime = MT.RW_ClassObj(dirN = dir_name, name = 'FilterRuntime', date = date[d], batch = search+search)
                # SupportedSV = MT.RW_ClassObj(dirN = dir_name, name = 'SupportedSV', date = date[d], batch = search+search)
                # AccDistribution = MT.RW_ClassObj(dirN = dir_name, name = 'AccDistribution', date = date[d], batch = search+search)
                # SizeDistribution = MT.RW_ClassObj(dirN = dir_name, name = 'SizeDistribution', date = date[d], batch = search+search)
                # Participation = MT.RW_ClassObj(dirN = dir_name, name = 'Participation', date = date[d], batch = search+search)
                for t in range(0, batchObj):
                    mj = 'm'+str(t)
                    Pre_Work = MT.RW_ClassObj(dirN = dir_name, name = 'ClassObj', date = date[d], batch = mj) # 載入模型訓練物件
                    for m_idx in range(0, len(k_list)):
                        pick = k_list[m_idx]
                        print(setup, 'K=', pick, sep = '')
                        cmb, cmb_acc, cmb_len, cmb_utility, cmb_marginal, training_time = MT.RunMS(ClassObj = Pre_Work, # SMT模型訓練
                                                                                                   search = SMT_search, # 0: 寬度優先走訪, 1: 深度優先走訪, 2: 完全隨機走訪
                                                                                                   k = pick, # 輸出模型數, 0~1為百分比(0~100%), 1以上為個數
                                                                                                   reboot = False, iterate = True)
                        Twophase = TPMP.Twophase_Predictive_Model_Pricing(cmb = cmb, acc = cmb_acc,lenGroup = cmb_len, # 建立TPMP物件
                                                                          utility = cmb_utility, marginal = cmb_marginal)
                        for rm in range(0, len(RmMode)):
                            if p2:
                                xOpt, r_total, r_ratio, r_time, ineqNum, itr, EC, AC, c_error, r_time_P2, R2Score, c_AvgError = TPMP.RunTPMP(
                                    ClassObj = Twophase,RunMode = [1, RmMode[rm], p2, True], ReadSurvey = True) # TPMP定價與分配
                            else:
                                xOpt, r_total, r_ratio, r_time, ineqNum, itr = TPMP.RunTPMP(
                                    ClassObj = Twophase,RunMode = [1, RmMode[rm], p2, True], ReadSurvey = True) # TPMP定價無分配
                            col_idx = m_idx*len(RmMode)+rm
                            records[0, col_idx, t] = r_total
                            records[1, col_idx, t] = r_ratio
                            records[2, col_idx, t] = r_time
                            records[3, col_idx, t] = len(xOpt)
                            records[4, col_idx, t] = itr
                            if p2:
                                records[5, col_idx, t] = c_error
                                records[6, col_idx, t] = r_time_P2
                                records[7, col_idx, t] = R2Score
                                records[8, col_idx, t] = c_AvgError
                            if rm == 0:
                                basePricing[m_idx, 0, t] = Pre_Work.trained_count
                                basePricing[m_idx, 1, t] = sum(Pre_Work.trainedTime)
                                basePricing[m_idx, 3, t] = r_time
                                for i in range(0, 2):
                                    if i == 1:
                                        xOpt, r_total, r_ratio, r_time = TPMP.RunTPMP(
                                            ClassObj = Twophase, RunMode = [0, 0, False, True], ReadSurvey = True)
                                        basePricing[m_idx, 6, t] = r_time
                                    sm_vioNum = Twophase.Ineq_Verify(sm = True, xOpt = xOpt, verify = True, show = False) # 驗證訓練集單調性
                                    if sm_vioNum > 0:
                                        basePricing[m_idx, 4+i*3, t] = 1
                                        VioMap[t, i, stk] = 1
                                    else:
                                        sa_vioNum = Twophase.Ineq_Verify(sm = False, xOpt = xOpt, verify = True, show = False) # 驗證訓練集次可加性
                                        if sa_vioNum > 0:
                                            basePricing[m_idx, 4+i*3, t] = 1
                                            VioMap[t, i, stk] = 1
                                        else:
                                            basePricing[m_idx, 2+i*3, t] = r_ratio
                                    if stk == 0:
                                        FilterRuntime[m_idx, 1+i*2, t] = r_time
                            if stk == 0:
                                if m_idx == 0 and rm == 0:
                                    SupportedSV[:10, -1, t] = np.sum(Pre_Work.cmb_utility, axis = 0)
                                    SupportedSV[10, -1, t] = 1
                                    AccDistribution[:50, -1, t] = Twophase.Get_AccDistribution(stack = True)
                                    AccDistribution[50, -1, t] = 1
                                    SizeDistribution[:10, -1, t] = Twophase.Get_SizeDistribution()
                                    SizeDistribution[10, -1, t] = 1
                                    Participation[:10, -1, t] = Twophase.Get_ParticipationCounts()
                                    Participation[10, -1, t] = 1
                                SupportedSV[:10, col_idx, t] = np.sum(Pre_Work.cmb_utility, axis = 0)
                                SupportedSV[10, col_idx, t] = np.corrcoef(SupportedSV[:10, col_idx, t], SupportedSV[:10, -1, t])[0][1]
                                AccDistribution[:50, col_idx, t] = Twophase.Get_AccDistribution(stack = True)
                                AccDistribution[50, col_idx, t] = np.corrcoef(AccDistribution[:50, col_idx, t], AccDistribution[:50, -1, t])[0][1]
                                SizeDistribution[:10, col_idx, t] = Twophase.Get_SizeDistribution()
                                SizeDistribution[10, col_idx, t] = np.corrcoef(SizeDistribution[:10, col_idx, t], SizeDistribution[:10, -1, t])[0][1]
                                Participation[:10, col_idx, t] = Twophase.Get_ParticipationCounts()
                                Participation[10, col_idx, t] = np.corrcoef(Participation[:10, col_idx, t], Participation[:10, -1, t])[0][1]
                                Ineq_distribution[:, col_idx, t] = ineqNum
                                if rm == 0:
                                    _, _, _, r_time, _, _ = TPMP.RunTPMP(ClassObj = Twophase, # 未過濾多餘不等式的CIP定價
                                                                         RunMode = [1, 0, False, True], ReadSurvey = True, IneqFilter = False)
                                    FilterRuntime[m_idx, 0, t] = r_time
                                    _, _, _, r_time = TPMP.RunTPMP(ClassObj = Twophase, # 未過濾多餘不等式的求解器定價
                                                                   RunMode = [0, 0, False, True], ReadSurvey = True, IneqFilter = False)
                                    FilterRuntime[m_idx, 2, t] = r_time
                            print('  ', setup, 'method[', rm, '] done!', sep = '')
                    print(setup, d*batchObj+t+1, '-th done!\n', sep = '')
            runtime = time.time() - starttime
    
            print('[Removing Effectiveness]\n', setup, 'Runtime: ', runtime, ' sec.\n', sep = '')
            title = list()
            for k in k_list:
                k_str = 'MAX' if k == 1 else str(k)
                for s in range(0, len(RmMode)):
                    title += [('K='+k_str+':'+strategy[s])]
    
            records_sub = np.sum(records, axis = 2)
            records_std = np.std(records, axis = 2)
            item = ['Revenue', 'Revenue Ratio (%)', 'Runtime (s)', 'Quantity', 'Iterations', 'Distribution Error (%)', 'Runtime_P2 (s)', 'R2 Score', 'Avg. Distribution Error (%)']
            RMdf = pd.DataFrame(records_sub/total, index = item, columns = title)
            fname = 'Optimizing Effectiveness Test ' + setup + search + '.csv'
            RMdf.to_csv(fname, index = True, header = True)
            RMdf_std = pd.DataFrame(records_std, index = item, columns = title)
            fname = 'Optimizing Effectiveness std ' + setup + search + '.csv'
            RMdf_std.to_csv(fname, index = True, header = True)
    
            if stk == 0:
                Ineq_distribution_sub = np.sum(Ineq_distribution, axis = 2)
                Ineq_distribution_std = np.std(Ineq_distribution, axis = 2)
                item = ['AccMT', 'SetMT', 'SubAdd', 'SetMT_Reserved', 'SubAdd_Reserved']
                IDdf = pd.DataFrame(Ineq_distribution_sub/total, index = item, columns = title)
                fname = 'Removed Inequalities Distribution Test ' + search + '.csv'
                IDdf.to_csv(fname, index = True, header = True)
                IDdf_std = pd.DataFrame(Ineq_distribution_std, index = item, columns = title)
                fname = 'Removed Inequalities Distribution std ' + search + '.csv'
                IDdf_std.to_csv(fname, index = True, header = True)
    
                item = list()
                for di in range(0, 10):
                    item += ['d'+str(di)]
                item += ['CorrCoef']
                SupportedSV_sub = np.sum(SupportedSV, axis = 2)
                SupportedSV_std = np.std(SupportedSV, axis = 2)
                title += ['MAX']
                SVdf = pd.DataFrame(SupportedSV_sub/total, index = item, columns = title)
                fname = 'Supported SV Test ' + search + '.csv'
                SVdf.to_csv(fname, index = True, header = True)
                SVdf_std = pd.DataFrame(SupportedSV_std, index = item, columns = title)
                fname = 'Supported SV std' + search + '.csv'
                SVdf_std.to_csv(fname, index = True, header = True)
        
                Participation_sub = np.sum(Participation, axis = 2)
                Participation_std = np.std(Participation, axis = 2)
                PPdf = pd.DataFrame(Participation_sub/total, index = item, columns = title)
                fname = 'Participation Test ' + search + '.csv'
                PPdf.to_csv(fname, index = True, header = True)
                PPdf_std = pd.DataFrame(Participation_std, index = item, columns = title)
                fname = 'Participation std ' + search + '.csv'
                PPdf_std.to_csv(fname, index = True, header = True)    
    
                item = list()
                for di in range(50, 100):
                    item += ['0.'+str(di)]
                item += ['CorrCoef']
                AccDistribution_sub = np.sum(AccDistribution, axis = 2)
                AccDistribution_std = np.std(AccDistribution, axis = 2)
                ADdf = pd.DataFrame(AccDistribution_sub/total, index = item, columns = title)
                fname = 'Accuracy Distribution Test ' + search + '.csv'
                ADdf.to_csv(fname, index = True, header = True)
                ADdf_std = pd.DataFrame(AccDistribution_std, index = item, columns = title)
                fname = 'Accuracy Distribution std ' + search + '.csv'
                ADdf_std.to_csv(fname, index = True, header = True)
    
                item = list(np.arange(1, 11))
                item += ['CorrCoef']
                SizeDistribution_sub = np.sum(SizeDistribution, axis = 2)
                SizeDistribution_std = np.std(SizeDistribution, axis = 2)
                SDdf = pd.DataFrame(SizeDistribution_sub/total, index = item, columns = title)
                fname = 'Size Distribution Test ' + search + '.csv'
                SDdf.to_csv(fname, index = True, header = True)
                SDdf_std = pd.DataFrame(SizeDistribution_std, index = item, columns = title)
                fname = 'Size Distribution std ' + search + '.csv'
                SDdf_std.to_csv(fname, index = True, header = True)
    
                FilterRuntime_sub = np.sum(FilterRuntime, axis = 2)
                FilterRuntime_std = np.std(FilterRuntime, axis = 2)
                title = ['CIP_Unfiltered', 'CIP_Filtered', 'SLSQP_Unfiltered', 'SLSQP_Filtered']
                FRdf = pd.DataFrame(FilterRuntime_sub/total, index = k_list, columns = title)
                fname = 'Filtered Efficiency Test ' + search + '.csv'
                FRdf.to_csv(fname, index = True, header = True)
                FRdf_std = pd.DataFrame(FilterRuntime_std, index = k_list, columns = title)
                fname = 'Filtered Efficiency std ' + search + '.csv'
                FRdf_std.to_csv(fname, index = True, header = True)
    
            basePricing_sub = np.sum(basePricing, axis = 2)
            basePricing_std = np.std(basePricing, axis = 2)
            title = ['Trained_Models', 'Training_Time','CIP_R_Ratio', 'CIP_Runtime', 'CIP_Vio', 'SLSQP_R_Ratio', 'SLSQP_Runtime', 'SLSQP_Vio']
            PCdf = pd.DataFrame(basePricing_sub/total, index = k_list, columns = title)
            fname = 'Pricing Efficiency Test ' + setup + search + '.csv'
            PCdf.to_csv(fname, index = True, header = True)
            PCdf_std = pd.DataFrame(basePricing_std, index = k_list, columns = title)
            fname = 'Pricing Efficiency std ' + setup + search + '.csv'
            PCdf_std.to_csv(fname, index = True, header = True)
    
            MT.RW_ClassObj(obj = records, wtire = True, dirN = dir_name, name = 'records', date = date[d], batch = setup+search)
            MT.RW_ClassObj(obj = records_std, wtire = True, dirN = dir_name, name = 'records_std', date = date[d], batch = setup+search)
            MT.RW_ClassObj(obj = basePricing, wtire = True, dirN = dir_name, name = 'basePricing', date = date[d], batch = setup+search)
            MT.RW_ClassObj(obj = basePricing_std, wtire = True, dirN = dir_name, name = 'basePricing_std', date = date[d], batch = setup+search)
            MT.RW_ClassObj(obj = VioMap[:,:, stk], wtire = True, dirN = dir_name, name = 'VioMap', date = date[d], batch = setup+search)
            if stk == 0:
                MT.RW_ClassObj(obj = Ineq_distribution, wtire = True, dirN = dir_name, name = 'Ineq_distribution', date = date[d], batch = search)
                MT.RW_ClassObj(obj = Ineq_distribution_std, wtire = True, dirN = dir_name, name = 'Ineq_distribution_std', date = date[d], batch = search)
                MT.RW_ClassObj(obj = FilterRuntime, wtire = True, dirN = dir_name, name = 'FilterRuntime', date = date[d], batch = search)
                MT.RW_ClassObj(obj = FilterRuntime_std, wtire = True, dirN = dir_name, name = 'FilterRuntime_std', date = date[d], batch = search)
                MT.RW_ClassObj(obj = SupportedSV, wtire = True, dirN = dir_name, name = 'SupportedSV', date = date[d], batch = search)
                MT.RW_ClassObj(obj = SupportedSV_std, wtire = True, dirN = dir_name, name = 'SupportedSV_std', date = date[d], batch = search)
                MT.RW_ClassObj(obj = AccDistribution, wtire = True, dirN = dir_name, name = 'AccDistribution', date = date[d], batch = search)
                MT.RW_ClassObj(obj = AccDistribution_std, wtire = True, dirN = dir_name, name = 'AccDistribution_std', date = date[d], batch = search)
                MT.RW_ClassObj(obj = SizeDistribution, wtire = True, dirN = dir_name, name = 'SizeDistribution', date = date[d], batch = search)
                MT.RW_ClassObj(obj = SizeDistribution_std, wtire = True, dirN = dir_name, name = 'SizeDistribution_std', date = date[d], batch = search)
                MT.RW_ClassObj(obj = Participation, wtire = True, dirN = dir_name, name = 'Participation', date = date[d], batch = search)
                MT.RW_ClassObj(obj = Participation_std, wtire = True, dirN = dir_name, name = 'Participation_std', date = date[d], batch = search)
            if curve == 'nNormal':
                break
            else:
                stk += 1