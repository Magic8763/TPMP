
import numpy as np
import pandas as pd
from copy import deepcopy as dcp
import math
import time
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import norm
from scipy.optimize import minimize

import TPMP_ModelTraining as MT

def ftoi(f, a = 1, isnpA = False): # 小數轉換為整數
    if isnpA:
        return (np.around(f*a)).astype(int)
    else:
#        return (round(f*a)).astype(int)
        return int(round(f*a))

def Monotone_Ineq(idxL, idxR): # 單調性不等式
    def MT_Ineq(x): # 不等式右方的模型價格 - 不等式左方的模型價格
        return (x[idxR] - x[idxL]) 
    return MT_Ineq

def SubAdditive_Ineq(idx1, idx2, idx3): # 次可加性不等式
    def SA_Ineq(x): # 不等式左方的m3模型價格 - 不等式右方的m1模型價格 - 不等式右方的m2模型價格
        return (x[idx1] + x[idx2] - x[idx3])
    return SA_Ineq

def Monotone_Ineq_MinDiff(wm1, wm2, minC): # 單調性不等式(單調差距至少為minC以上)
    def MT_Ineq_MinDiff(x):
        return (x[wm1] - x[wm2]- minC) 
    return MT_Ineq_MinDiff

def wSum(Di_list, num = 1): # 貢獻正規化等式
    def Sum1(x):
        return (x[Di_list[0]] - num) 
    def Sum2(x):
        return (x[Di_list[0]] + x[Di_list[1]] - num)
    def Sum3(x):
        return (x[Di_list[0]] + x[Di_list[1]] + x[Di_list[2]] - num) 
    def Sum4(x):
        return (x[Di_list[0]] + x[Di_list[1]] + x[Di_list[2]] + x[Di_list[3]] - num)
    def Sum5(x):
        return (x[Di_list[0]] + x[Di_list[1]] + x[Di_list[2]] + x[Di_list[3]] + x[Di_list[4]] - num)
    def Sum6(x):
        return (x[Di_list[0]] + x[Di_list[1]] + x[Di_list[2]] + x[Di_list[3]] + x[Di_list[4]] + 
                x[Di_list[5]] - num)
    def Sum7(x):
        return (x[Di_list[0]] + x[Di_list[1]] + x[Di_list[2]] + x[Di_list[3]] + x[Di_list[4]] + 
                x[Di_list[5]] + x[Di_list[6]] - num) 
    def Sum8(x):
        return (x[Di_list[0]] + x[Di_list[1]] + x[Di_list[2]] + x[Di_list[3]] + x[Di_list[4]] + 
                x[Di_list[5]] + x[Di_list[6]] + x[Di_list[7]] - num)
    def Sum9(x):
        return (x[Di_list[0]] + x[Di_list[1]] + x[Di_list[2]] + x[Di_list[3]] + x[Di_list[4]] + 
                x[Di_list[5]] + x[Di_list[6]] + x[Di_list[7]] + x[Di_list[8]] - num) 
    def Sum10(x):
        return (x[Di_list[0]] + x[Di_list[1]] + x[Di_list[2]] + x[Di_list[3]] + x[Di_list[4]] + 
                x[Di_list[5]] + x[Di_list[6]] + x[Di_list[7]] + x[Di_list[8]] + x[Di_list[9]] - num)
    if len(Di_list) == 1:
        return Sum1
    elif len(Di_list) == 2:
        return Sum2
    elif len(Di_list) == 3:
        return Sum3
    elif len(Di_list) == 4:
        return Sum4
    elif len(Di_list) == 5:
        return Sum5
    elif len(Di_list) == 6:
        return Sum6
    elif len(Di_list) == 7:
        return Sum7
    elif len(Di_list) == 8:
        return Sum8
    elif len(Di_list) == 9:
        return Sum9
    elif len(Di_list) == 10:
        return Sum10

class Twophase_Predictive_Model_Pricing(object): # TPMP Class
    def __init__(self, cmb, acc, lenGroup, utility, marginal): # 載入版本模型
        self.ML_cmb_og = cmb
        self.ML_acc_og = acc
        self.ML_lenGroup_og = lenGroup
        self.ML_utility_og = utility
        self.ML_marginal_og = marginal

    def Initialize(self, irregular = False, ReadSurvey = False, plot = False, **kwargs): # 載入定價參數
        self.ML_size_og = len(self.ML_cmb_og)
        self.X_size_og = np.size(self.ML_utility_og, axis = 1)
        self.irregular = irregular # 考慮不規則預算曲線
        self.RemovedCap = kwargs.get('RemovedCap', -1) # 模型移除數上限
        self.acc_base = kwargs.get('acc_base', 0.5) # 定價起始點
        self.TB = kwargs.get('TB', 0.002) # 等價約束的容許邊界(準確度差值邊界)
        self.IneqFilter = kwargs.get('IneqFilter', True) # 是否需要過濾多餘不等式
        if ReadSurvey:
            dir_name = kwargs.get('dir_name', 'cifar-10-batches-py/Classification')
            date = kwargs.get('date', 'Market Adjustment function')
            self.V_acc = MT.RW_ClassObj(dirN = dir_name, name = 'V_acc', date = date)
            self.V_acc_og = dcp(self.V_acc)
            self.D_acc = MT.RW_ClassObj(dirN = dir_name, name = 'D_acc', date = date)
        else:
            Vacc_general = kwargs.get('Vacc_general', True)
            Vacc_growthRate = kwargs.get('Vacc_growthRate', 0.1)
            Vacc_p0 = kwargs.get('Vacc_p0', 0) # 底價
            Vacc_acc0 = kwargs.get('Vacc_acc0', 0.5) # 預算起始點
            Vacc_reverse = kwargs.get('Vacc_reverse', False)
            self.CreateBudgetCurve(general = Vacc_general, growth_rate = Vacc_growthRate,
                                   p0 = Vacc_p0, acc0 = Vacc_acc0, reverse = Vacc_reverse)
            Dacc_interval = kwargs.get('Dacc_interval', 1)
            Dacc_disp = kwargs.get('Dacc_disp', 0)
            Dacc_reverse = kwargs.get('Vacc_reverse', False)
            Dacc_plot = kwargs.get('Dacc_plot', False)
            Dacc_fontsize = kwargs.get('Dacc_fontsize', 14)
            self.CreateDemandCurve(interval = Dacc_interval, disp = Dacc_disp,
                                   reverse = Dacc_reverse, histplot = Dacc_plot, fontsize = Dacc_fontsize)
        if plot:
            Vacc_general = kwargs.get('Vacc_general', True)
            Vacc_acc0 = kwargs.get('Vacc_acc0', 0.5)
            Plot_points = kwargs.get('Plot_points', False)
            Dacc_fontsize = kwargs.get('Dacc_fontsize', 14)
            self.SurveyPlot(general = Vacc_general, acc0 = Vacc_acc0, point = Plot_points, fontsize = Dacc_fontsize)

    def Reload_Models(self, more_weight = False): # 初始化定價參數
        self.ML_cmb = dcp(self.ML_cmb_og)
        self.ML_acc = dcp(self.ML_acc_og)
        self.ML_lenGroup = dcp(self.ML_lenGroup_og)
        self.ML_utility = dcp(self.ML_utility_og)
        self.ML_marginal = dcp(self.ML_marginal_og)
        self.ML_pmt_idx = [np.array([i]) for i in range(0, len(self.ML_cmb))]
        self.ML_idxloc = np.arange(0, self.ML_size_og)
        self.ML_price = np.zeros(0, float) # 當前價格函數
        self.BasePrice = np.zeros(0, float) # 原始價格函數
        self.more_weight = more_weight
        self.CaptureSurvey()
        self.Set_accGroup()
        self.ML_accloc_og = dcp(self.ML_accloc)
        self.Set_rMax()

    def CurveReverse(self, curve): #垂直翻轉
        return curve*-1 + max(curve)
    
    def CreateBudgetCurve(self, general = True, growth_rate = 0.1, p0 = 0, acc0 = 0.5, reverse = False): # 價格上限(準確度預算)函數
        if general:
            price_start = 0
            price_end = 500
        else:
            price_base = ftoi(acc0, 1000)
            price_start = ftoi(min(self.ML_acc), 1000) - price_base
            price_end = ftoi(max(self.ML_acc), 1000) - price_base + 1
        V_acc = np.zeros(0, float)
        alp = 1
        dwctr = 0
        for i in range(0, price_end):
            if self.irregular:
                dwctr += 1
                if dwctr % 10 == 0:
                    alp *= 0.8
            if i >= price_start:
                V_acc = np.append(V_acc, alp*(1 + i*growth_rate)**2)
        if reverse:
            V_acc = self.CurveReverse(V_acc)[::-1] # 垂直反轉 + 水平反轉 = 上下凸反轉
        self.V_acc = ftoi(V_acc, 100, isnpA = True)/100 + p0
        # pd.DataFrame([self.V_acc]).transpose().plot()
        self.V_acc_og = dcp(self.V_acc)
        if self.irregular: # 不規則預算曲線的單調化修正
            pCap = self.V_acc[-1]
            for i in reversed(range(0, len(self.V_acc)-1)):
                if self.V_acc[i] > pCap:
                    self.V_acc[i] = pCap
                elif self.V_acc[i] < pCap:
                    pCap = self.V_acc[i]

    def CreateDemandCurve(self, interval = 1, disp = 0, reverse = False, histplot = False, fontsize = 14): # 需求分布函數
        size = 500
        cut = 5000
        batch = math.ceil(size/interval)
        db = np.random.normal(0, 1, cut) # 平均=0, 標準差=1, 抽樣數=5000
        bins = cut//(10*interval)
        if histplot:
            num, dbr, _ = plt.hist(db, bins-1, density = True)
            # D_dbr = scipy.stats.norm.pdf(dbr, 0, 1)
            D_dbr = norm.pdf(dbr, 0, 1)
            df = pd.DataFrame(dbr, columns = ['Scale'])
            df['PDF'] = D_dbr
            plt.plot(df['Scale'], df['PDF'], linewidth = 4)
            plt.title('The Standard Normal Distribution', fontsize = fontsize)
            plt.xlabel('Z-score', fontsize = fontsize)
            plt.ylabel('Probability Density', fontsize = fontsize)
            plt.xticks(fontsize = fontsize-2)
            plt.yticks(fontsize = fontsize-2)
            lb = int(len(df['Scale'])*0.25-1)
            rb = int(len(df['Scale'])*0.75-1)
            # plt.axvline(x = df['Scale'].iloc[lb], color = 'r', linestyle = '--', lw = 2)
            # plt.axvline(x = df['Scale'].iloc[rb], color = 'r', linestyle = '--', lw = 2)
            plt.tight_layout()
            plt.savefig('Histograms.jpg', bbox_inches = 'tight', dpi = 300)
            plt.show()
        else:
            _, dbr = np.histogram(db, bins = bins-1)
            # D_dbr = scipy.stats.norm.pdf(dbr, 0, 1)
            D_dbr = norm.pdf(dbr, 0, 1)
        if reverse:
            D_dbr = self.CurveReverse(D_dbr) # 垂直翻轉
        move = ftoi(len(D_dbr)*disp, 1)
        if reverse:
            mid = np.argmin(D_dbr)
        else:
            mid = np.argmax(D_dbr)
        lb = mid - batch//2 - move
        rb = lb + batch
        if lb < 0:
            fillup = -lb
            self.D_acc = np.append(np.repeat(D_dbr[0], fillup), D_dbr[0:rb])
        elif rb > bins:
            fillup = rb-bins
            self.D_acc = np.append(D_dbr[lb:], np.repeat(D_dbr[-1], fillup))
        else:
            self.D_acc = D_dbr[lb:rb]
        print('lb:', lb, ', rb:', rb)
        D_acc_uf = np.zeros(size, float)
        for i in range(0, batch):
            D_acc_uf[interval*i:interval*(i+1)] = self.D_acc[i]
        # self.D_acc = dcp(D_acc_uf)
        # self.D_acc /= sum(self.D_acc)
        self.D_acc = D_acc_uf/sum(D_acc_uf) # 測試改變

    def SurveyPlot(self, general = True, acc0 = 0.5, point = False, fontsize = 14): # 繪製市場調查函數
        D_start = (10-(ftoi(min(self.ML_acc_og), 1000)%10))%10
        point_idx = np.arange(D_start, len(self.D_acc), step = 10)
        D = self.D_acc/sum(self.D_acc[point_idx])*100
        # D = self.D_acc/sum(self.D_acc)*100
        nan_idx = np.setdiff1d(np.arange(len(self.D_acc)), point_idx, False)
        D_point = dcp(D)
        D_point[nan_idx] = np.nan
        if general:
            acc = np.arange(acc0, 1, step = 0.001)
        else:
            acc = np.arange(min(self.ML_acc_og), max(self.ML_acc_og)+0.001, step = 0.001)
        if len(acc) == len(self.D_acc)+1:
            acc = acc[:-1]
        df = pd.DataFrame({'Accuracy':acc, 'Budget':self.V_acc, 'Demand':D, 'Point':D_point})
        img = df.plot(kind = 'line', x = 'Accuracy', y = 'Budget', color = 'Blue', legend = False, fontsize = fontsize-2)
        img1 = df.plot(kind = 'line', x = 'Accuracy', y = 'Demand', secondary_y = True, color = 'Red', ax = img, legend = False, fontsize = fontsize-2)
        if point:
            img2 = df.plot(kind = 'scatter', x = 'Accuracy', y = 'Point', color = 'Red', ax = img1, legend = False, fontsize = fontsize-2)
        img.set_title('Accuracy Budget and Demand Distribution', fontsize = fontsize)
        img.set_xlabel('Prediction Accuracy', fontsize = fontsize)
        img.set_ylabel('Buyer Budget', fontsize = fontsize)
        img.yaxis.label.set_color('Blue')
        img.tick_params(axis = 'y', colors='Blue')
        img1.set_ylabel('Buyer Distribution', fontsize = fontsize)
        img1.yaxis.label.set_color('Red')
        img1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 1))
        img1.tick_params(axis = 'y', colors='Red')
        plt.tight_layout()
        plt.show()
        img.figure.savefig('Hypothetical_Curve.jpg', bbox_inches = 'tight')
        img.remove()
        img1.remove()
        if point:
            img2.remove()

    def CaptureSurvey(self, rm = False): # 精簡市場調查函數
        self.ML_accRange = np.sort(np.unique(self.ML_acc), axis = 0) # 由小到大的準確度序列
        acc_set = ftoi((self.ML_accRange - self.acc_base), 1000, isnpA = True)
        if not rm:
            self.acc_budget_og = self.V_acc_og[acc_set]
        self.acc_budget = self.V_acc[acc_set]
        self.acc_demand = self.D_acc[acc_set]

    def Set_accGroup(self): # 模型點依照其準確度分組
        self.ML_accGroup = list()
        self.ML_accloc = np.zeros(len(self.ML_cmb), int)
        acc_range = np.argsort(self.ML_acc, axis = 0) # 準確度由小到大的模型點序列
        last_acc = -1 # 前一個刻度的準確度
        idx_scale = np.zeros(0, int)
        for i in acc_range:
            if self.ML_acc[i] == last_acc:
                idx_scale = np.append(idx_scale, i)
            else:
                if i != acc_range[0]:
                    self.ML_accloc[idx_scale] = len(self.ML_accGroup)
                    self.ML_accGroup.append(idx_scale)
                idx_scale = np.array([i])
                last_acc = self.ML_acc[i]
        self.ML_accloc[idx_scale] = len(self.ML_accGroup)
        self.ML_accGroup.append(idx_scale)

    def Set_rMax(self): # 計算價格上限總和
        rMax = 0
        for loc in range(0, len(self.ML_accGroup)):
            rMax += self.acc_budget_og[loc]*self.acc_demand[loc]
        self.rMax = np.around(rMax, decimals = 3)

    def Expected_Revenue(self, x, total = True, weighted = True): # 計算預期收益
        if not weighted:
            if total:
                return sum(x)
            else:
                return x
        elif total:
            r_total = 0
            for loc in range(0, len(self.ML_accGroup)):
            # r_total += sum(x[self.ML_accGroup[loc]])*self.acc_demand[loc]
                r_total += sum(x[self.ML_accGroup[loc]])*self.acc_demand[loc]/len(self.ML_accGroup[loc])
            return np.around(r_total, decimals = 3)
        else:
            r = np.zeros(len(x), float)
            for loc in range(0, len(self.ML_accGroup)):
                for mj in self.ML_accGroup[loc]:
                    r[mj] = x[mj]*self.acc_demand[loc]/len(self.ML_accGroup[loc])
            return np.around(r, decimals = 3)

    def Normalize_SV(self, sv): # 正規化SV: 平移負值且加總為1
        if min(sv) < 0:
            sv -= min(sv)
        return sv/sum(sv)

    def Update_Exp(self, rm_idx = [], updateOnly = True, mpW = False): # 更新邊際貢獻矩陣
        sv = np.zeros(self.X_size_og, float)
        for idx in rm_idx:
            self.ML_utility[idx,] = self.ML_utility[-1,]
            self.ML_utility = np.delete(self.ML_utility, -1, axis = 0)
            self.ML_marginal[idx,] = self.ML_marginal[-1,]
            self.ML_marginal = np.delete(self.ML_marginal, -1, axis = 0)
        if not updateOnly:
            if mpW:
                r = self.Expected_Revenue(self.ML_price, total = False)
                for Di in range(0, self.X_size_og):
                    sv[Di] = sum(self.ML_utility[:, Di]*r)
            else:
                sv = np.sum(self.ML_utility, axis = 0)
            diff = np.sum(self.ML_marginal, axis = 0)
            return self.Normalize_SV(sv), self.Normalize_SV(diff)

    def DelML(self, idxs): # 移除模型
        if len(idxs) > 0:
            removed_idxs = np.sort(np.unique(idxs), axis = 0)[::-1] # 從idx高的模型開始刪除
            for i in range(0, len(removed_idxs)):
                idx = removed_idxs[i]
                final_idx = len(self.ML_cmb)-1
                final_len = len(self.ML_cmb[final_idx])-1
        
                if idx != final_idx:
                    self.ML_acc[idx], self.ML_acc[-1] = self.ML_acc[-1], self.ML_acc[idx]
                    self.ML_cmb[idx], self.ML_cmb[-1] = self.ML_cmb[-1], self.ML_cmb[idx]
                    self.ML_idxloc[idx] = self.ML_idxloc[final_idx]
                self.ML_idxloc[final_idx] = -1
                self.ML_acc.pop()
                acc_idx = len(self.ML_cmb.pop())-1
        
                tf = np.in1d(self.ML_lenGroup[acc_idx], idx)
                self.ML_lenGroup[acc_idx] = self.ML_lenGroup[acc_idx][~tf]
                final_len_idx = np.where(self.ML_lenGroup[final_len] == final_idx)
                self.ML_lenGroup[final_len][final_len_idx] = idx
        
                for j in range(0, len(self.ML_pmt_idx)):
                    if idx in self.ML_pmt_idx[j]:
                        tf = np.in1d(self.ML_pmt_idx[j], idx)
                        self.ML_pmt_idx[j] = self.ML_pmt_idx[j][~tf]
                    if final_idx in self.ML_pmt_idx[j]:
                        final_pmt_idx = np.where(self.ML_pmt_idx[j] == final_idx)
                        self.ML_pmt_idx[j][final_pmt_idx] = idx
                # print('Removed idx:', idx)
            self.CaptureSurvey(rm = True)
            self.Set_accGroup() # 模型點刪除完畢後必須重建預算曲線和self.ML_accGroup準確度分組
            self.Update_Exp(rm_idx = removed_idxs, updateOnly = True)

    def ML_idx_Recovery(self, mp): # 還原移除後的模型idx位置
        price = np.zeros(self.ML_size_og, float)
        for i in range(0, len(self.ML_idxloc)):
            if self.ML_idxloc[i] >= 0:
                price[self.ML_idxloc[i]] = mp[i]
        return price

    def MaxBudget(self, ineqlist, DemOnly = False, more_weight = True, cover = True): # 最高[需求量/收益上限/價格上限]優先
        ans = np.zeros(0, int)
        idxlist = np.zeros(0, int)
        for vioM in ineqlist:
            idxlist = np.append(idxlist, vioM) # 串聯所有不等式上的模型
        idxlist = np.unique(idxlist) # 去除重複模型
        paring = np.ones((len(idxlist), len(ineqlist)), float) # 比對矩陣(模型點, 不等式)
        acc_budget_max = max(self.acc_budget)
        grouplen = np.zeros(len(self.ML_accGroup), int)
        prob = np.zeros(0, float)
        wrn = np.zeros(0, float)
        pMaxlist = np.zeros(0, float)
        for g in range(0, len(self.ML_accGroup)):
            grouplen[g] = len(self.ML_accGroup[g])
            prob = np.append(prob, self.acc_demand[g]/grouplen[g])
            wrn = np.append(wrn, self.acc_budget[g]*self.acc_demand[g])
            pMaxlist = np.append(pMaxlist, self.acc_budget[g]*prob[g])
        prob_max = max(prob)
        wrn_max = max(wrn)
        pMax_max = max(pMaxlist)
        if cover:
            for j in range(0, len(ineqlist)):
                for idx in ineqlist[j]:
                    for i in range(0, len(idxlist)):
                        if idxlist[i] == idx: # 模型i有無參與不等式j
                            if DemOnly:
                                paring[i, j] = prob[self.ML_accloc[idx]] - prob_max # 紀錄相對需求量
                            elif more_weight:
                                paring[i, j] = pMaxlist[self.ML_accloc[idx]] - pMax_max # 紀錄相對加權價格
                            else:
                                paring[i, j] = self.acc_budget[self.ML_accloc[idx]] - acc_budget_max # 紀錄相對價格
        while len(paring[0,]) > 0: # 沒有不等式未處理便會跳出
            if cover:
                values = np.min(paring, axis=1)
            else:
                values = np.repeat(wrn_max, len(idxlist))
                for i in range(0, len(idxlist)):
                    if idxlist[i] not in ans:
                        if DemOnly:
                            values[i] = self.acc_demand[self.ML_accloc[idxlist[i]]]/grouplen[self.ML_accloc[idxlist[i]]]
                        else:
                            values[i] = wrn[self.ML_accloc[idxlist[i]]]/grouplen[self.ML_accloc[idxlist[i]]]
                if min(values) == max(values):
                    break
            Minidx = np.argmin(values) # 挑出需求量最低的模型
            ans = np.append(ans, idxlist[Minidx]) # 放進待移除清單
            if self.RemovedCap >= 0 and self.RemovedCap == len(ans):
                break
            paring = paring[:, np.where(paring[Minidx,] == 1)[0]] # 移除比對矩陣內該模型所參與過的不等式的欄位
            grouplen[self.ML_accloc[idxlist[Minidx]]] -= 1
            # print('Ans:', ans[-1])
        return ans

    def MinCover(self, ineqlist, more_weight = False): # 最小覆蓋集 = 最多參與優先
        ans = np.zeros(0, int)
        idxlist = np.zeros(0, int)
        for vioM in ineqlist:
            idxlist = np.append(idxlist, vioM[:-2]) # 串聯所有不等式上的模型
        unq_idx = np.unique(idxlist) # 去除重複模型
        sorted_idx = np.argsort(np.array(self.ML_acc)[unq_idx])
        idxlist = unq_idx[sorted_idx]
        grouplen = np.zeros(len(self.ML_accGroup), int)
        wrn = np.zeros(0, float)
        for g in range(0, len(self.ML_accGroup)):
            grouplen[g] = len(self.ML_accGroup[g])
            wrn = np.append(wrn, self.acc_budget[g]*self.acc_demand[g])
        paring = np.zeros((len(idxlist), len(ineqlist)), float) # 比對矩陣(模型點, 不等式)
        for j in range(0, len(ineqlist)):
            for idx in ineqlist[j][:-2]:
                for i in range(0, len(idxlist)):
                    if idxlist[i] == idx: # 模型i有無參與不等式j: 有=1, 無=0
                        paring[i, j] = 1
        while len(paring[0,]) > 0: # 沒有不等式未處理便會跳出
            coverlist = np.sum(paring, axis=1) # 重新計算剩餘模型各自參與的不等式總數
            if more_weight:
                for i in range(0, len(idxlist)):
                    if idxlist[i] not in ans:
                        coverlist[i] *= grouplen[self.ML_accloc[idxlist[i]]]
                        coverlist[i] /= wrn[self.ML_accloc[idxlist[i]]]
            Maxidx = np.argmax(coverlist) # 挑出參與不等式數量最多的模型
            ans = np.append(ans, idxlist[Maxidx]) # 放進待移除清單
            if self.RemovedCap >= 0 and self.RemovedCap == len(ans):
                break
            paring = paring[:, np.where(paring[Maxidx,] == 0)[0]] # 移除比對矩陣內該模型所參與過的不等式的欄位
            grouplen[self.ML_accloc[idxlist[Maxidx]]] -= 1
    #        print('Ans:', ans[-1])
        return ans

    def MaxImprove(self, ineqlist): # 最大可改進收益
        ans = np.zeros(0, int)
        idxlist = np.zeros(0, int)
        for vioM in ineqlist:
            idxlist = np.append(idxlist, ftoi(vioM[1:], 1, isnpA = True)) # 串聯所有不等式上的模型
        unq_idx = np.unique(idxlist) # 去除重複模型
        sorted_idx = np.argsort(np.array(self.ML_acc)[unq_idx])
        idxlist = unq_idx[sorted_idx]
        grouplen = np.zeros(len(self.ML_accGroup), int)
        for g in range(0, len(self.ML_accGroup)):
            grouplen[g] = len(self.ML_accGroup[g])
        RCost = (self.acc_budget*self.acc_demand)[self.ML_accloc[idxlist]]
        paring = np.zeros((len(idxlist), len(ineqlist)), float) # 比對矩陣(模型點, 不等式)
        for j in range(0, len(ineqlist)):
            for idx in range(1, len(ineqlist[j])):
                for i in range(0, len(idxlist)):
                    if idxlist[i] == ineqlist[j][idx]: # 模型i有無參與不等式j: 有=該不等式的潛在收益, 無=0
                        paring[i, j] = ineqlist[j][0]
        while len(paring[0,]) > 0: # 沒有不等式未處理便會跳出
            improvement = np.sum(paring, axis=1) # 重新計算剩餘模型各自的潛在收益總和
            for j in range(0, len(idxlist)):
                if improvement[j] > 0: # 收益改進 = 潛在收益總和 - 該模型目前的價值(成本)
                    if grouplen[self.ML_accloc[idxlist[j]]] == 1:
                        improvement[j] -= RCost[j] # 可改進收益 = 收益損失總合 - 候選模型的移除成本(若有)
            if max(improvement) <= 0 or (self.RemovedCap  >= 0 and self.RemovedCap == len(ans)): # 提前結束(剩餘的收益改進都<=0)
                break
            Maxidx = np.argmax(improvement) # 挑出可增長收益最大的模型
            ans = np.append(ans, idxlist[Maxidx]) # 放進待移除清單
            paring = paring[:, np.where(paring[Maxidx,] == 0)[0]] # 移除比對矩陣內該模型所參與過的不等式的欄位
            grouplen[self.ML_accloc[idxlist[Maxidx]]] -= 1
    #        print('Ans:', ans[-1])
        return ans

    def Revenue_Imp(self, ineqlist, more_weight = False): # 可改進收益
        if more_weight:
            cover = np.zeros(len(self.acc_budget), int)
            for ineq in ineqlist:
                LBM_accloc = int(ineq[-2])+1
                RBM_accloc = int(ineq[-1])+1
                cover[LBM_accloc:RBM_accloc] += 1
        unzip_ineq = list()
        for ineq in ineqlist:
            pGap = 0
            pCap = ineq[0]
            LBM_accloc = int(ineq[-2])+1
            RBM_accloc = int(ineq[-1])+1
            for accloc in range(LBM_accloc, RBM_accloc):
                if more_weight:
                    pGap += (self.acc_budget[accloc] - pCap)*self.acc_demand[accloc]/cover[accloc]
                else:
                    pGap += (self.acc_budget[accloc] - pCap)*self.acc_demand[accloc]
            unzip_ineq.append(np.append(pGap, ineq[1:-2]))
        return unzip_ineq

    def MaxImprove_Union(self, ineqlist): # 最大可改進收益(聯集)
        ans = np.zeros(0, int)
        idxlist = np.zeros(0, int)
        for vioM in ineqlist:
            idxlist = np.append(idxlist, ftoi(vioM[0:-2], 1, True)) # 串聯所有不等式上的模型
        idxlist = np.unique(idxlist) # 去除重複模型
        prob = np.zeros(0, float)
        for g in range(0, len(self.ML_accGroup)):
            prob = np.append(prob, self.acc_demand[g]/len(self.ML_accGroup[g]))
        pMaxlist = self.acc_budget[self.ML_accloc[idxlist]]*prob[self.ML_accloc[idxlist]] # 有需求加權
        PR = np.zeros(len(self.ML_accGroup), float)
        for i in range(0, len(self.ML_accGroup)):
            PR[i] = (self.acc_budget[i] - self.BasePrice[self.ML_accGroup[i][0]])*prob[i]
        paring = np.zeros((len(idxlist), len(ineqlist)), float) # 比對矩陣(模型點, 不等式)
        for j in range(0, len(ineqlist)):
            for idx in range(0, len(ineqlist[j])-2):
                for i in range(0, len(idxlist)):
                    if idxlist[i] == ineqlist[j][idx]: # 模型i有無參與不等式j: 有=該不等式的潛在收益, 無=0
                        paring[i, j] = 1
        accGroup_len = np.zeros(len(self.ML_accGroup), int)
        for i in range(0, len(self.ML_accGroup)):
            accGroup_len[i] += len(self.ML_accGroup[i])
        cover = np.zeros((len(ineqlist), len(self.ML_accGroup)), int) # 準確度區間的不等式覆蓋數
        for i in range(0, len(ineqlist)):
            ineq = ineqlist[i]
            lb = ftoi(ineq[-2], 1, True)+1
            rb = ftoi(ineq[-1], 1, True)
            cover[i, lb:rb] = 1
        while np.sum(np.sum(paring, axis = 1)) > 0: # 沒有不等式未處理便會跳出
            improvement = np.zeros(len(idxlist), float)
            for vio_idx in range(0, len(idxlist)):
                ineq_idx = np.where(paring[vio_idx,] == 1)[0]
                improvement[vio_idx] = self.Revenue_Imp_Union(PR, ineq_idx, cover, accGroup_len) - pMaxlist[vio_idx]
            if max(improvement) <= 0 or (self.RemovedCap >= 0 and self.RemovedCap == len(ans)): # 提前結束(剩餘的收益改進都<=0)
                break
            Maxidx = np.argmax(improvement) # 挑出可增長收益最大的模型
            ans = np.append(ans, idxlist[Maxidx]) # 放進待移除清單
            rm_ineq_idx = np.where(paring[Maxidx,] == 1)[0]
            paring[:, rm_ineq_idx] = 0 # 移除比對矩陣內該模型所參與過的不等式的欄位
            cover[rm_ineq_idx, :] = 0
            accGroup_len[self.ML_accloc[Maxidx]] -= 1
    #        print('Ans:', ans[-1])
        return ans

    def Revenue_Imp_Union(self, ploss, ineq_idx, cover, accGroup_len): # 可改進收益(聯集)
        cover_sum = np.sum(cover, axis = 0)
        cover_ineq = np.sum(cover[ineq_idx,:], axis = 0)
        cover_real = np.where(cover_ineq > 0)[0]
        pGap = 0
        for loc in cover_real:
            pGap += ploss[loc]*accGroup_len[loc]/cover_sum[loc]
        for ineq in cover[ineq_idx,:]:
            for i in reversed(range(0, len(ineq))):
                if ineq[i] == 1 and cover_ineq[i+1] == 0:
                    cover_ineq[i+1] = 1
                    if cover_sum[i+1] > 0:
                        pGap += ploss[i]/cover_sum[i+1]
                    else:
                        pGap += ploss[i]
        return pGap

    def GreedyPick(self, mode, ineqlist): # 選擇收益優化方法
        if mode == 1:
            # return self.MaxBudget(ineqlist, True, True, False) # 最高需求量
            return self.MaxBudget(ineqlist, False, True, False) # 最高收益上限
        elif mode == 2:
            return self.MaxBudget(ineqlist, False, True, True) # 最高收益上限(不等式參與加權)
        elif mode == 3:
            return self.MinCover(ineqlist, self.more_weight) # 最小覆蓋集
        elif mode == 4:
            unzip_ineq = self.Revenue_Imp(ineqlist, False) # 可改進收益
            return self.MaxImprove(unzip_ineq) # 最大可改進收益
        elif mode == 5:
            return self.MaxImprove_Union(ineqlist) # 最大可改進收益(聯集)
        else:
            return np.zeros(0, float)

    def RevenueRatio(self, x): # 計算平均收益百分比
        r_total = self.Expected_Revenue(x)
        r_ratio = np.around(100*r_total/self.rMax, decimals = 3)
        return r_total, r_ratio

    def CompensationRatio(self, cError, cMax): # 計算總收益百分比
        if cError > cMax:
            return np.around(cMax, decimals = 3), 100.0
        else:
            c_ratio = np.around(100*cError/cMax, decimals = 3)
            return np.around(cError, decimals = 3), c_ratio

    def RLoss(self, x): # 預期收益損失
        return self.rMax - self.Expected_Revenue(x)

    def MPLoss(self, x): # 目標函數: 模型總價損失
        fx = 0
        for j in range(0, len(self.ML_accGroup)):
            for idx in self.ML_accGroup[j]:
                if self.acc_budget[j] > x[idx]:
                    fx += self.acc_budget[j] - x[idx] # 模型的加權價格與價格上限之間的損失總合
        return fx

    def CError(self, x, mode = 0): # 目標函數: 回饋分布誤差
        EC_mt = np.zeros((len(self.ML_cmb), self.X_size_og), float)
        AC_mt = np.zeros((len(self.ML_cmb), self.X_size_og), float)
        wi = 0
        for g in range(0, len(self.ML_accGroup)):
            for mj in self.ML_accGroup[g]:
                wr = self.ML_price[mj]*self.acc_demand[g]/len(self.ML_accGroup[g])
                EC_mt[mj,:] = wr*self.Exp_SV
                for Di in range(0, len(self.ML_marginal[mj,:])):
                    if self.ML_marginal[mj, Di] > 0:
                        AC_mt[mj, Di] = wr*x[wi]
                        wi += 1
        EC = np.sum(EC_mt, axis = 0)
        AC = np.sum(AC_mt, axis = 0)
        if mode == 0:
            return sum(abs(EC - AC))
        elif mode == 1:
            return np.around(abs(EC - AC), decimals = 3), np.around(EC, decimals = 3), np.around(AC, decimals = 3)
        elif mode == 2:
            return np.around(sum(EC) - sum(AC), decimals = 3)
        elif mode == 3:
            for i in range(0, self.X_size_og):
                AC[i] = abs(EC[i] - AC[i])
                if EC[i] != 0:
                    AC[i] /= EC[i]
            return np.around(sum(AC)/self.X_size_og*100, decimals = 3)

    def CError_R2Score(self, x, perc = True): # 回饋分布誤差的決定係數
        rs = np.zeros(self.X_size_og, float)
        r_total = 0
        wi = 0
        for g in range(0, len(self.ML_accGroup)):
            for mj in self.ML_accGroup[g]:
                wi_end = wi + len(np.where(self.ML_marginal[mj,:] > 0)[0])
                dbSum = sum(x[wi:wi_end])
                wr = self.ML_price[mj]*self.acc_demand[g]/len(self.ML_accGroup[g])
                r_total += wr
                for Di in range(0, len(self.ML_marginal[mj,:])):
                    if self.ML_marginal[mj, Di] > 0:
                        if perc:
                            rs[Di] += wr*x[wi]
                        else:
                            rs[Di] += wr*x[wi]/dbSum
                        wi += 1
        rs_total = r_total*self.Exp_SV
        R2 = r2_score(rs_total, rs)
        return -np.around(R2, decimals = 3)

    def Set_vio(self, removed_ineq): # 不等式參與模型的集合
        removed_idx = np.zeros(0, int)
        for idx in removed_ineq:
            removed_idx = np.append(removed_idx, idx)
        return removed_idx

    def Add_SM_Ineq(self, RmMode = 0, count = False): # 產生訓練集單調性不等式
        const = list()
        vioM = list()
        total = 0
        remaining = 0
        checked = np.zeros(len(self.ML_cmb), dtype = int) # 紀錄已檢查的模型點idx的矩陣
        for m2_j in range(0, len(self.ML_pmt_idx)): # 整個搜索建立在ML_pmt_idx結構之上
            for m2_i in range(0, len(self.ML_pmt_idx[m2_j])):
                m2_idx = self.ML_pmt_idx[m2_j][m2_i]
                if checked[m2_idx] == 0:
                    checked[m2_idx] = 1
                    ign_idx = self.ML_pmt_idx[m2_j][m2_i+1:] # 紀錄可忽略的模型點idx
                    m2_set = set(self.ML_cmb[m2_idx])
                    m2_len = len(self.ML_cmb[m2_idx])
                    for m1_j in range(m2_len, len(self.ML_lenGroup)):
                        for m1_idx in self.ML_lenGroup[m1_j]:
                            m1_set = set(self.ML_cmb[m1_idx])
                            if (m1_idx not in ign_idx) and (m2_set.issubset(m1_set)):
                                total += 1
                                if not self.IneqFilter or self.ML_acc[m1_idx] < self.ML_acc[m2_idx]:
                                    remaining += 1 
                                    if count:
                                        continue
                                    removed_ineq = np.append(m1_idx, m2_idx)
                                    m1_accloc = self.ML_accloc[m1_idx] # 超集模型(左邊界)的準確度位置
                                    m2_accloc = self.ML_accloc[m2_idx] # 子集模型(右邊界)的準確度位置
                                    if RmMode >= 0 and RmMode <= 5:
                                        const.append({'type': 'ineq', # 訓練集組合單調性不等式
                                                      'fun' : Monotone_Ineq(m2_idx, m1_idx)})
                                    else:
                                        m2Cap = self.acc_budget[m1_accloc] # 水平線的價格限制 = 超集的價格上限
                                        record = np.append(m2Cap, removed_ineq)
                                        record = np.append(record, m2_accloc)
                                        const.append(record)
                                    if RmMode > 0 and m2_accloc - m1_accloc > 1:
                                        removed_idx = self.Set_vio(removed_ineq)
                                        # if len(removed_idx) > 0:
                                        if len(removed_idx) > 0 and self.ML_acc[m2_idx] - self.ML_acc[m1_idx] > self.TB:
                                            if RmMode <= 2:
                                                vioM.append(removed_idx)
                                            elif RmMode >= 3 and self.ML_acc[m1_idx] < self.ML_acc[m2_idx]: # 子集準確度 > 超集準確度
                                                if RmMode == 4:
                                                    m2Cap = self.acc_budget[m1_accloc] # 水平線的價格限制 = 超集的價格上限
                                                    LRB_loc = np.append(m1_accloc, m2_accloc)
                                                    temp = np.append(removed_idx, LRB_loc)
                                                    vioM.append(np.append(m2Cap, temp))
                                                elif RmMode == 3 or RmMode == 5:
                                                    LRB_loc = np.append(m1_accloc, m2_accloc)
                                                    vioM.append(np.append(removed_idx, LRB_loc))
        if count:
            return total, remaining
        return const, vioM, total

    def Add_SA_Ineq(self, RmMode = 0, count = False): # 產生訓練集次可加性不等式
        const = list()
        vioM = list()
        total = 0
        remaining = 0
        paring = np.diag(np.diag(np.ones((len(self.ML_cmb), len(self.ML_cmb)), dtype=int))) # 紀錄已比對的模型點idx的矩陣
        for m1_j in range(0, len(self.ML_pmt_idx)): # 整個搜索建立在ML_pmt_idx結構之上
            for m1_i in range(0, len(self.ML_pmt_idx[m1_j])):
                m1_idx = self.ML_pmt_idx[m1_j][m1_i]
                for i in range(m1_i+1, len(self.ML_pmt_idx[m1_j])): # 屬於相同排列曲線上的模型點可以跳過比對(直接紀為已比對)
                    paring[m1_idx, self.ML_pmt_idx[m1_j][i]] = 1 # 紀錄跳過的模型點idx
                    paring[self.ML_pmt_idx[m1_j][i], m1_idx] = 1
                for m2_j in range(m1_j+1, len(self.ML_pmt_idx)):
                    for m2_i in range(0, len(self.ML_pmt_idx[m2_j])):
                        m2_idx = self.ML_pmt_idx[m2_j][m2_i]
                        if paring[m1_idx, m2_idx] == 0:
                            paring[m1_idx, m2_idx] = 1 # 紀錄已比對的模型點idx
                            paring[m2_idx, m1_idx] = 1
                            m3_set = set(self.ML_cmb[m1_idx]).union(set(self.ML_cmb[m2_idx])) # 聯集m1與m2的訓練集組合
                            if m3_set != set(self.ML_cmb[m1_idx]) and m3_set != set(self.ML_cmb[m2_idx]): # 集合有改變才需要做次可加性檢測
                                m3_exist = False
                                for m3_idx in self.ML_lenGroup[len(m3_set)-1]: # 從ML_lenGroup的長度分類陣列中找出訓練集組合完全相同的模型idx
                                    if set(self.ML_cmb[m3_idx]) == m3_set:
                                        m3_exist = True
                                        break
                                if m3_exist:
                                    total += 1
                                    if not self.IneqFilter or (self.ML_acc[m1_idx] < self.ML_acc[m3_idx] and self.ML_acc[m2_idx] < self.ML_acc[m3_idx]):
                                        remaining += 1
                                        if count:
                                            continue
                                        if self.ML_acc[m1_idx] <= self.ML_acc[m2_idx]:
                                            removed_ineq = np.append(m1_idx, m2_idx)
                                        else:
                                            removed_ineq = np.append(m2_idx, m1_idx)
                                        removed_ineq = np.append(removed_ineq, m3_idx)
                                        m3Cap = self.acc_budget[self.ML_accloc[m1_idx]] + self.acc_budget[self.ML_accloc[m2_idx]] # m3Cap = m1_pMax+m2_pMax
                                        if RmMode >= 0 and RmMode <= 5:
                                            const.append({'type': 'ineq', # 次可加性不等式
                                                          'fun' : SubAdditive_Ineq(m1_idx, m2_idx, m3_idx)})
                                        else:
                                            m3_accloc = self.ML_accloc[m3_idx] # 水平線右邊界的準確度位置
                                            record = np.append(m3Cap, removed_ineq)
                                            record = np.append(record, m3_accloc)
                                            const.append(record)
                                        if RmMode > 0:
                                            m4_accidx = np.where(self.ML_accRange < self.ML_acc[m3_idx])[0] # 選擇準確度低於m3的點
                                            m4_pMaxidx = np.where(m3Cap < self.acc_budget)[0] # 選擇預算高於m3Cap的點
                                            m4_list = np.intersect1d(m4_accidx, m4_pMaxidx) # 取交集找出m4候選者
                                            if len(m4_list) > 0: # 左邊界存在
                                                m4_accloc = m4_list[0]-1 # 取第一項(準確度最低者)作為水平線左邊界(m4)的準確度位置
                                                if m4_accloc >= 0:
                                                    removed_idx = self.Set_vio(removed_ineq)
                                                    # if len(removed_idx) > 0:
                                                    if len(removed_idx) > 0 and self.ML_acc[m3_idx] - self.ML_accRange[m4_accloc] > self.TB:
                                                        if RmMode <= 2:
                                                            vioM.append(removed_idx)
                                                        elif RmMode >= 3:
                                                            m3_accloc = self.ML_accloc[m3_idx] # 水平線右邊界的準確度位置
                                                            if RmMode == 4:
                                                                LRB_loc = np.append(m4_accloc, m3_accloc)
                                                                temp = np.append(removed_idx, LRB_loc)
                                                                vioM.append(np.append(m3Cap, temp))
                                                            elif RmMode == 3 or RmMode == 5:
                                                                LRB_loc = np.append(m4_accloc, m3_accloc)
                                                                vioM.append(np.append(removed_idx, LRB_loc))
        if count:
            return total, remaining
        return const, vioM, total

    def Add_AM_Ineq(self, count = False): # 產生準確度單調性不等式
        const = list()
        total = 0
        preMax = 0
        for j in range(0, len(self.ML_accGroup)):
            for i in range(0, len(self.ML_accGroup[j])):
                idxR = self.ML_accGroup[j][i]
                if j > 0 and i == 0: # 跨準確度不等式
                    const.append({'type': 'ineq', # pre[-1] <= now[0]
                                  'fun' : Monotone_Ineq(preMax, idxR)})
                    total += 1
                elif i > 0: # 同準確度內不等式
                    idxL = self.ML_accGroup[j][i-1] # pre[i-1] <= now[i]
                    const.append({'type': 'ineq', # 以兩個相反的不等式表示等式約束
                                  'fun' : Monotone_Ineq(idxL, idxR)})
                    const.append({'type': 'ineq',
                                  'fun' : Monotone_Ineq(idxR, idxL)})
                    total += 1
            preMax = idxR
        if count:
            return total
        return const

    def Pricing_Ineq(self, LP = True, RmMode = 0, Get_RM_idxs = False): # 無套利不等式
        const = list()
        ML_size = len(self.ML_cmb)
        while True:
    #    if True:
            sm_ineq, sm_vio, _ = self.Add_SM_Ineq(RmMode) # 添加訓練集單調性不等式
            sa_ineq, sa_vio, _ = self.Add_SA_Ineq(RmMode) # 添加訓練集次可加性不等式
    #        print('\nDelete idxs[sa]:', np.unique(sa_vio))
    #        print('Delete idxs[sm]:', np.unique(sm_vio))
            sa_vio += sm_vio
            if RmMode > 0 and len(sa_vio) > 0:
    #            removed_idxs = sa_vio
    #            removed_idxs = GreedyPick(RmMode, sm_vio)
                removed_idxs = self.GreedyPick(RmMode, sa_vio)
                # print('DelML:\n', removed_idxs)
                if Get_RM_idxs:
                    return removed_idxs
                self.DelML(removed_idxs)
            if ML_size == len(self.ML_cmb) or len(removed_idxs) >= self.RemovedCap:
                break
            else:
                ML_size = len(self.ML_cmb)
        if LP:
            const += sm_ineq + sa_ineq
            amt_ineq = self.Add_AM_Ineq() # 添加準確度單調性不等式
            const += amt_ineq
            return tuple(const)

    def Define_Bounds(self, PhMode = 1): # 定義邊界約束(LP決策變數的上下限), PhMode = 1: 兩階段LP-1, 2:兩階段LP-2
        tp = list()
        x = np.zeros(0, float)
        if PhMode == 1:
            for i in range(0, len(self.ML_cmb)): # 模型價格mp的邊界
                loc = self.ML_accloc[i]
                tp.append((0, self.acc_budget[loc]))
                x = np.append(x, self.acc_budget[loc])
        elif PhMode == 2:
            for i in range(0, self.X_size_og): # 分配權重w的邊界
                tp.append((1, None))
                x = np.append(x, 1)
        return tuple(tp), x

    def LP_Pricing(self, RmMode = 0, p2 = False, iterate = False): # 無套利定價 by 線性規劃器
        solver = 'SLSQP'
        maxiter = 1000
        options = {'maxiter': maxiter, 'eps': 0.1}
        starttime = time.time()
        if RmMode == 5 and len(self.BasePrice) == 0:
            Total_Const = self.Pricing_Ineq(True, 0, False)
            bnds, xIni = self.Define_Bounds(PhMode = 1) # 邊界約束和變量初始值
            solution = minimize(fun = self.MPLoss, x0 = xIni, method = solver, bounds = bnds, constraints = Total_Const, options = options)
            self.BasePrice = solution.x
        Total_Const = self.Pricing_Ineq(True, RmMode, False)
        bnds, xIni = self.Define_Bounds(PhMode = 1) # 邊界約束和變量初始值
        solution = minimize(fun = self.MPLoss, x0 = xIni, method = solver, bounds = bnds, constraints = Total_Const, options = options)
        runtime = time.time() - starttime
        self.ML_price = np.around(solution.x, decimals = 3)
        if RmMode == 0:
            self.BasePrice = dcp(self.ML_price)
        r_total, r_ratio = self.RevenueRatio(self.ML_price)
        if iterate:
            if p2:
                _, c_error, _, c_loss = self.LP_Distribution(cMax = r_total, iterate = iterate)
                return self.ML_price, r_total, r_ratio, runtime, c_error, c_loss
            else:
                return self.ML_price, r_total, r_ratio, runtime
        else:
            print('\nArbitrage-free Pricing:', solver)
            print("Optimization problem: {}".format(solution.message)) # 優化是否成功
            print('Price Loss = {:.4f}'.format(self.MPLoss(self.ML_price)))
            print('Revenue Loss = {:.4f}'.format(self.RLoss(self.ML_price)))
            print('Total_Revenue: ', r_total, ' / ',  self.rMax, ' (', r_ratio, '%)', sep='')
            print('Pricing ML_', len(self.ML_cmb), ' P1: ', runtime, ' sec.', sep='')
            if p2:
                return self.LP_Distribution(cMax = r_total)
            return self.ML_price, self.ML_idx_Recovery(self.ML_price)

    def Distribution_Ineq(self): # 分配權重專用不等式
        const = list()
        tp = list()
        x = np.zeros(0, float)
        for g in range(0, len(self.ML_accGroup)):
            for mj in self.ML_accGroup[g]:
                idx0 = len(x)
                acc_diff_idxs = np.where(self.ML_marginal[mj,:] != 0)[0]
                dbSum = sum(self.ML_marginal[mj, acc_diff_idxs])
                acc_diff_sorted = np.argsort(self.ML_marginal[mj, acc_diff_idxs], axis = 0)[::-1]
                for j in range(0, len(acc_diff_sorted)):
                    tp.append((0, 1))
                    x = np.append(x, self.ML_marginal[mj, acc_diff_idxs[j]]/dbSum)
                    if j > 0:
                        diff = (self.ML_marginal[mj, acc_diff_idxs[acc_diff_sorted[j-1]]] - self.ML_marginal[mj, acc_diff_idxs[acc_diff_sorted[j]]])/dbSum
                        if diff > 0:
                            # const.append({'type': 'ineq', 'fun' : Monotone_Ineq_MinDiff(idx0 + acc_diff_sorted[j-1], idx0 + acc_diff_sorted[j], diff)})
                            const.append({'type': 'ineq', 'fun' : Monotone_Ineq_MinDiff(idx0 + acc_diff_sorted[j-1], idx0 + acc_diff_sorted[j], 0)})
                        elif diff == 0:
                            const.append({'type': 'eq', 'fun' : Monotone_Ineq_MinDiff(idx0 + acc_diff_sorted[j-1], idx0 + acc_diff_sorted[j], 0)})
                const.append({'type': 'eq', 'fun' : wSum(idx0 + acc_diff_sorted)})
        return tuple(const), tuple(tp), x

    def LP_Distribution(self, cMax, iterate = False): # 分配最佳化 by 線性規劃器
        solver = 'SLSQP'
        # solver = 'L-BFGS-B'
        maxiter = 5
        options = {'maxiter': maxiter, 'eps': 0.1}
        starttime = time.time()
        self.Exp_SV, self.Exp_AccDiff = self.Update_Exp(updateOnly = False)
        LossFun = self.CError
        Total_Const, bnds, xIni = self.Distribution_Ineq()
        solution = minimize(fun = LossFun, x0 = xIni, method = solver, bounds = bnds, constraints = Total_Const, options = options)
        runtime = time.time() - starttime
        xOpt = solution.x
        cDiff = LossFun(xOpt, 0)
        cLoss = LossFun(xOpt, 2)
        cAvgError = LossFun(xOpt, 3)
        cDiff, c_error = self.CompensationRatio(cDiff, cMax)
        R2Score = -self.CError_R2Score(xOpt, perc = True)
        if iterate:
            _, EC, AC = LossFun(xOpt, 1)
            return EC, AC, c_error, runtime, R2Score, cAvgError
        else:
            print('\nDistribution Optimization:', solver)
            # print('Optimization problem: {}'.format(solution.message)) # 優化是否成功
            print('Error(EC, AC) = {:.4f}'.format(cDiff))
            print('Loss(EC, AC) = {:.4f}'.format(cLoss))
            print('Avg. Error(EC, AC) = {:.4f}'.format(cAvgError))
            print('R2(EC, AC) = {:.4f}'.format(R2Score))
            print('Compensation Error: ', cDiff, ' / ',  cMax, ' (', c_error, '%)', sep='')
            print('Distribution ML', len(self.ML_cmb), ' P2: ', runtime, ' sec.', sep='')
            if len(self.ML_price) < self.ML_size_og:
                xOpt = np.append(self.ML_idx_Recovery(self.ML_price), xOpt)
            else:
                xOpt = np.append(self.ML_price, xOpt)
            return xOpt

    def Get_MaxP(self): # 價格函數初始化(模型價格設為價格上限)
        MaxP = np.zeros(len(self.ML_cmb), float)
        for j in reversed(range(0, len(self.ML_accGroup))):
            for idx in self.ML_accGroup[j]:
                MaxP[idx] = self.acc_budget[j]
        return MaxP

    def Constraint_based_Iterative_Pricing(self, ineqlist): # 基於約束的模型定價方法CIP
        NewPrice = self.Get_MaxP()
        itr_t = 0
        extra_check = True
        while True:
            stop = True
            for k in range(0, len(ineqlist)):
                ineq_mi = ftoi(ineqlist[k][1:-1], 1, isnpA = True)
                pCap = 0
                if len(ineq_mi) == 2:
                    pCap = NewPrice[ineq_mi[0]] # LBM_idx
                elif len(ineq_mi) == 3:
                    pCap = NewPrice[ineq_mi[0]] + NewPrice[ineq_mi[1]]
                if NewPrice[ineq_mi[-1]] > pCap: # RBM_idx
                    NewPrice[ineq_mi[-1]] = pCap
                    stop = False
                    RBM_loc = int(round(ineqlist[k][-1]))+1
                    if self.irregular and RBM_loc == len(self.ML_accGroup):
                        extra_check = False
                    for j in reversed(range(0, RBM_loc)):
                        minWB = min(NewPrice[self.ML_accGroup[j]])
                        if minWB < pCap:
                            pCap = minWB
                        NewPrice[self.ML_accGroup[j]] = pCap
            itr_t += 1
            if stop:
                break                    
        if self.irregular and extra_check:
            pCap = NewPrice[-1]
            for j in reversed(range(0, len(self.ML_accGroup))):
                pCap_next = pCap
                for idx in self.ML_accGroup[j]:
                    if NewPrice[idx] > pCap:
                        NewPrice[idx] = pCap
                    elif NewPrice[idx] != 0 and NewPrice[idx] < pCap_next:
                        pCap_next = NewPrice[idx]
                pCap = pCap_next
            itr_t += 1
        return NewPrice, itr_t

    def CIP_Pricing(self, RmMode = 0, sort_ineq = True, p2 = False, iterate = False): # CIP切換模型優化前處理的介面
        starttime = time.time()
        IneqNum = np.zeros(5, int)
        if RmMode == 0 or RmMode == 5:
    #        sm_ineq = list()
    #        sa_ineq = list()
            sm_ineq, _, IneqNum[1] = self.Add_SM_Ineq(-1)
            sa_ineq, _, IneqNum[2] = self.Add_SA_Ineq(-1)
            if sort_ineq:
                ineqlist = list(sorted(sm_ineq + sa_ineq, key = lambda x: (x[0], x[-2])))
            else:
                ineqlist = sm_ineq + sa_ineq
            if RmMode == 0:
                xOpt, itr_t = self.Constraint_based_Iterative_Pricing(ineqlist)
                self.BasePrice = dcp(xOpt)
            elif RmMode == 5 and len(self.BasePrice) == 0:
                self.BasePrice, itr_t2 = self.Constraint_based_Iterative_Pricing(ineqlist)
        if RmMode >= 1 and RmMode <= 5:
            self.Pricing_Ineq(False, RmMode, False)
            sm_ineq, _, IneqNum[1] = self.Add_SM_Ineq(-1)
            sa_ineq, _, IneqNum[2] = self.Add_SA_Ineq(-1)
            if sort_ineq:
                ineqlist = list(sorted(sm_ineq + sa_ineq, key = lambda x: x[0]))
            else:
                ineqlist = sm_ineq + sa_ineq
            xOpt, itr_t = self.Constraint_based_Iterative_Pricing(ineqlist)
            if RmMode == 5 and len(self.BasePrice) == 0:
                itr_t += itr_t2
        runtime = time.time() - starttime
        self.ML_price = np.around(xOpt, decimals = 3)
        r_total, r_ratio = self.RevenueRatio(self.ML_price)
        if iterate:
            IneqNum[0] = len(self.Add_AM_Ineq())
            IneqNum[3] = len(sm_ineq)
            IneqNum[4] = len(sa_ineq)
            if p2:
                EC, AC, c_error, runtime_P2, R2Score, c_AvgError = self.LP_Distribution(cMax = r_total, iterate = iterate)
                return self.ML_price, r_total, r_ratio, runtime, IneqNum, itr_t, EC, AC, c_error, runtime_P2, R2Score, c_AvgError
            else:
                return self.ML_price, r_total, r_ratio, runtime, IneqNum, itr_t
        else:
            print('\nArbitrage-free Pricing: CIP')
            print('Iterations =', itr_t)
            print('Price Loss = {:.4f}'.format(self.MPLoss(self.ML_price)))
            print('Revenue Loss = {:.4f}'.format(self.RLoss(self.ML_price)))
            print('Total_Revenue: ', r_total, ' / ',  self.rMax, ' (', r_ratio, '%)', sep = '')
            print('Pricing ML', len(self.ML_cmb), ' P1: ', runtime, ' sec.', sep = '')
            if p2:
                return self.LP_Distribution(cMax = r_total)
            return self.ML_price, self.ML_idx_Recovery(self.ML_price)

    def DFplot(self, df, fontsize = 14):
        name = list(df.columns)
        df = df.sort_values(by = name, kind = 'mergesort')
        df = df.replace(0, np.nan)
        img = df.plot(x = 'Acc', y = name[1:], style = '.-', fontsize = fontsize-2)
        img.set_xlabel('Prediction Accuracy', fontsize = fontsize)
        img.set_ylabel('Price', fontsize = fontsize)
        img.set_title('Price of Version Models', fontsize = fontsize)
        img.legend(loc = 'best', fontsize = fontsize-2)
        # plt.fill_between(df['Acc'], df['Budget'], df['Arbitrage-free Price'], facecolor = 'green', alpha = 0.3, interpolate = True)
        plt.show()
        img.figure.savefig('Model_Pricing_Curve.jpg', bbox_inches = 'tight')
        img.remove()
        return df

    def PlotCurve(self, xOpt = [[0]], RmMode = [0], fontsize = 14):        
        strategies = ['Base', 'MaxRevenue', 'MaxRevenueW', 'MinCover', 'MaxImprove', 'MaxImprove_Uion']
        xOptDF = pd.DataFrame(self.ML_acc_og, columns = ['Acc'])
        if self.irregular:
            xOptDF['v(.)'] = self.acc_budget_og[self.ML_accloc_og]
            xOptDF['Budget_fixed'] = self.acc_budget[self.ML_accloc_og]
        else:
            xOptDF['Budget'] = self.acc_budget_og[self.ML_accloc_og]
        for i in range(0, len(RmMode)):
            xOptDF[strategies[RmMode[i]]] = xOpt[i][:self.ML_size_og]
        return self.DFplot(df = xOptDF, fontsize = fontsize)

    def CIP_Pricing_RM1by1(self, RmMode = 4, p2 = False): # 逐項移除模型並定價
        xOpt = self.CIP_Pricing(RmMode = 0)
        if RmMode > 0:
            ans = self.Pricing_Ineq(LP = False, RmMode = RmMode, Get_RM_idxs = True)
            sorted_ans = np.sort(ans)[::-1]
            for i in range(1, len(sorted_ans)+1):
                self.Reload_Models(more_weight = False)
                rl = list(sorted_ans[0:i])
                self.DelML(rl)
                print('Latest Removed Model: m', rl[-1], ' = ',  self.ML_cmb_og[rl[-1]], sep = '')
                _, r_total, r_ratio, _, _, _ = self.CIP_Pricing(RmMode = 0, iterate = True)
                print('[P1]: Rr = ', r_total, ' / ',  self.rMax, ' (', r_ratio, '%)', sep = '')
                if p2:
                    EC, AC, c_error, runtime, R2Score, cAvgError = self.LP_Distribution(cMax = r_total, iterate = True)
                    print('SV:', self.Exp_SV)
                    print('AccDiff:', self.Exp_AccDiff)
                    print('[P2]: CError = ', c_error, '%, R2Score = ', R2Score, '\n', EC, '\n -', AC, '\n =', abs(EC - AC), sep = '')
            print('\n')
        else:
            print('RmMode = [1, ..., 5]!')

    def Ineq_Verify(self, sm = True, xOpt = np.zeros(0, int), verify = True, show = True): # 不等式驗證
        const = list()
        VioNum = 0 # 違反不等式的次數
        IneqNum = 0 # 不等式總數
        if len(xOpt) == 0:
            xOpt = self.ML_price
        pmt_idx = np.zeros(0, int)
        for group in self.ML_lenGroup:
            for idx in group:
                pmt_idx = np.append(pmt_idx, idx)
        for m1_i in range(0, len(pmt_idx)-1):
            m1_idx = pmt_idx[m1_i]
            if sm:
                m1_set = set(self.ML_cmb[m1_idx])
                m1_len = len(self.ML_cmb[m1_idx])
                for m2_j in range(0, m1_len-1):
                    for m2_idx in self.ML_lenGroup[m2_j]:
                        m2_set = set(self.ML_cmb[m2_idx])
                        if m2_set.issubset(m1_set) and self.ML_acc[m1_idx] <= self.ML_acc[m2_idx]:
                            removed_ineq = np.append(m1_idx, m2_idx)
                            if verify:
                                m1_price = xOpt[m1_idx]
                                m2_price = xOpt[m2_idx]
                                IneqNum += 1
                                if m1_price < m2_price-1:
                                    VioNum += 1
                                    if show:
                                        print('Set Monotone Violation!')
                                        print('m1(', m1_idx, '): ', m1_price, sep = '') # 違反不等式的m1
                                        print('m2(', m2_idx, '): ', m2_price, '\n', sep = '') # 違反不等式的m2
                            else:
                                print('SM\n  m1(', m1_idx, ') >= m2(', m2_idx, ')', sep = '') # 違反不等式的m1
                                const.append({'type': 'ineq', # 訓練集單調性不等式
                                              'fun' : Monotone_Ineq(m2_idx, m1_idx)})
            else:
                for m2_i in range(m1_i+1, len(pmt_idx)):
                    m2_idx = pmt_idx[m2_i]
                    m3_set = set(self.ML_cmb[m1_idx]).union(set(self.ML_cmb[m2_idx]))
                    if m3_set != set(self.ML_cmb[m1_idx]) and m3_set != set(self.ML_cmb[m2_idx]):
                        m3_exist = False
                        for m3_idx in self.ML_lenGroup[len(m3_set)-1]: # 從ML_lenGroup的長度分類陣列中找出訓練集組合完全相同的模型idx
                            if m3_set == set(self.ML_cmb[m3_idx]):
                                m3_exist = True
                                break
                        if m3_exist and self.ML_acc[m1_idx] <= self.ML_acc[m3_idx] and self.ML_acc[m2_idx] <= self.ML_acc[m3_idx]:
                            if verify:
                                m1_price = xOpt[m1_idx] # 為找到的m3和m1, m2計算加權價格
                                m2_price = xOpt[m2_idx]
                                m3_price = xOpt[m3_idx]
                                IneqNum += 1
                                if m1_price + m2_price < m3_price-1: # 比較m3和m1, m2的價格是否違規
                                    VioNum += 1
                                    if show:
                                        print('Set SubAdditive Violation!')
                                        print('m3(', m3_idx, '): ', m3_price, sep = '') # 違反不等式的m3
                                        print('m1(', m1_idx, '): ', m1_price, sep = '') # 違反不等式的m1
                                        print('m2(', m2_idx, '): ', m2_price, '\n', sep = '') # 違反不等式的m2
                            else:
                                if self.ML_acc[m1_idx] < self.ML_acc[m2_idx]:
                                    removed_ineq = np.append(m1_idx, m2_idx)
                                else:
                                    removed_ineq = np.append(m2_idx, m1_idx)
                                removed_ineq = np.append(removed_ineq, m3_idx)
                                print('SA\n  m1(', m1_idx, ') + m2(', m2_idx, ') >= m3(', m3_idx,')', sep='') # 違反不等式的m1
                                const.append({'type': 'ineq', # 訓練集次可加性不等式
                                              'fun' : SubAdditive_Ineq(m1_idx, m2_idx, m3_idx)})
        if verify:
            if show:
                if sm:
                    print('\nSM')
                else:
                    print('\nSA')
                print('IneqNum: ', IneqNum, ', VioNum: ', VioNum, sep = '')
                print('Violation rate: ', ftoi(VioNum/IneqNum, 100), '%', sep='') # 違規比例
            return VioNum
        else:
            return const

    def Get_AccDistribution(self, stack = True): # 模型準確度分布
        AccDistribution = np.zeros(50, int)
        for g in range(0, len(self.ML_accRange)):
            acc_loc = ftoi(self.ML_accRange[g]*1000)//10 - 50
            if stack:
                AccDistribution[acc_loc] += len(self.ML_accGroup[g])
            else:
                AccDistribution[acc_loc] += 1
        return AccDistribution
    
    def Get_SizeDistribution(self): # 模型大小分布
        SizeDistribution = np.zeros(len(self.ML_lenGroup), int)
        for g in range(0, len(self.ML_lenGroup)):
            SizeDistribution[g] = len(self.ML_lenGroup[g])
        return SizeDistribution
    
    def Get_ParticipationCounts(self): # 資料集參與分布
        counts = np.zeros(self.X_size_og, int)
        for c in self.ML_cmb:
            counts[c] += 1
        return counts

## In[模型定價與分配]:

def RunTPMP(ClassObj, RunMode = [1, 0, False, False], irregular = False, ReadSurvey = False, plot = False, **kwargs):
    # RunMode[定價方法, 定價模式, 有無收益分配, 是否為迭代進行(實驗時隱藏多餘的文字輸出)]
    # irregular考慮不規則價格上限函數, ReadSurvey載入外部市調函數, plot繪圖
    ClassObj.Initialize(irregular = irregular, ReadSurvey = ReadSurvey, plot = plot, **kwargs) # 載入定價參數
    ClassObj.Reload_Models(more_weight = False)
    if RunMode[0] == 0: # 以求解器定價
        return ClassObj.LP_Pricing(RmMode = RunMode[1], p2 = RunMode[2], iterate = RunMode[3])
    elif RunMode[0] == 1: # 以CIP定價
        return ClassObj.CIP_Pricing(RmMode = RunMode[1], p2 = RunMode[2], iterate = RunMode[3])
    elif RunMode[0] == 2: # 逐項移除並以CIP定價
        ClassObj.CIP_Pricing_RM1by1(RmMode = RunMode[1], p2 = RunMode[2], iterate = RunMode[3])

