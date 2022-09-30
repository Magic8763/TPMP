
import os
import time
import numpy as np
import pandas as pd
import warnings
from sklearn.base import clone
from itertools import combinations # permutations
import scipy.special
import random
import pickle

## In[shap_utils.py提供的函數]:

from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def ftoi(f, a = 1, isnp = False): # 小數轉換為整數
    if isnp:
        return (np.around(f*a)).astype(int)
    else:
#        return (round(f*a)).astype(int)
        return int(round(f*a))

def binomial_coefficients(n,k): # Cn取k個數
    product = 1
    for i in range(k):
        product = (product*(n-i))//(i+1)
    return product 

def generate_features(latent, dependency): # 特徵產生器(特徵向量, 所需特徵數)
    features = []
    n = latent.shape[0]
    exp = latent
    holder = latent
    for order in range(1,dependency+1):
        features.append(np.reshape(holder, [n,-1]))
        exp = np.expand_dims(exp,-1)
        holder = exp * np.expand_dims(holder, 1)
    return np.concatenate(features, axis=-1)  

def label_generator(X, problem, param, difficulty = 1, beta = None, important = None):
# 資料產生器(問題=分類, 訓練集X, 迭代係數=1.0*1.1^(0~n), difficulty=1, beta=None, important=5)
    if important is None or important > X.shape[-1]: # 若important為None或大於訓練集維度
        important = X.shape[-1] # 將important設為訓練集維度(特徵數)
    dim_latent = sum([important**i for i in range(1, difficulty+1)]) # sum(important^(1~(difficulty+1)))=sum(5^1)=5
    if beta is None:
        beta = np.random.normal(size=[1, dim_latent]) # 長度為1*dim_latent的常態分布矩陣
    important_dims = np.random.choice(X.shape[-1], important, replace=False) # 從訓練集中隨機選擇important個不重複的index作為特徵向量
    funct_init = lambda inp: np.sum(beta * generate_features(inp[:,important_dims], difficulty), -1) # 先將訓練資料X[:,特徵向量]轉為5100*5矩陣, 再乘上beta (1*5的矩陣), 最後加總5個維度的值
    batch_size = max(100, min(len(X), 10000000//dim_latent)) # 批次大小=5100
    y_true = np.zeros(len(X)) # 初始化答案向量
    while True:
        try:
            for itr in range(int(np.ceil(len(X)/batch_size))): # 迭代至(訓練集/批次大小)向上取整=range(0,1)
                y_true[itr * batch_size: (itr+1) * batch_size] = funct_init(
                    X[itr * batch_size: (itr+1) * batch_size]) # y_true[0:5100]=funct_init(X[0:5100])
                # 從訓練集中隨機選擇important個不重複的特徵作為特徵向量 (5個5100*1矩陣)
                # 將這些特徵向量分別乘上常態分佈矩陣beta (1個1*5矩陣)
                # 最後將5個特徵向量的值加總作為答案向量 (1個5100*1矩陣)
            break
        except MemoryError:
            batch_size = batch_size//2 # 將批次大小除2向下取整
    mean, std = np.mean(y_true), np.std(y_true)
    funct = lambda x: (np.sum(beta * generate_features(x[:, important_dims], difficulty), -1) - mean) / std
    y_true = (y_true - mean)/std # 將答案向量y_true標準化
    if problem == 'classification':
        y_true = logistic.cdf(param * y_true) # 將答案向量y_true以累積分佈函數做轉換 (變成介於0~1之間), param越大, 轉換後的值越接近邊緣 (趨近0或1)
        y = (np.random.random(X.shape[0]) < y_true).astype(int) # 將答案向量y_true與隨機陣列(介於0~1之間)做比較, 藉此將答案向量轉為{0,1}二元標籤向量y
    elif problem == 'regression':
        y = y_true + param * np.random.normal(size=len(y_true))
    else:
        raise ValueError('Invalid problem specified!')
    return beta, y, y_true, funct

def return_model(mode = 'logistic' , **kwargs): # 初始化模型
    if mode == 'logistic': # 羅吉斯回歸分類器
        solver = kwargs.get('solver', 'liblinear') # 優化演算法=liblinear座標軸下降法
        tol = kwargs.get('tol', 0.0001) # 優化演算法=liblinear座標軸下降法
        n_jobs = kwargs.get('n_jobs', 1) # 平行處理CPU核心數=None
        max_iter = kwargs.get('max_iter', 5000) # 收斂的最大迭代數=5000
        model = LogisticRegression(solver = solver, tol = tol, n_jobs = n_jobs, max_iter = max_iter, random_state = 666) # 隨機種子=666
    elif mode == 'SVC':
        class_weight = kwargs.get('class_weight', 'balanced') # 優化演算法=liblinear座標軸下降法
        model = SVC(kernel = 'rbf', class_weight = class_weight, random_state = 666)
#        model = SVC(kernel = 'linear', random_state = 666)
#        model = SVC(kernel = 'poly', random_state = 666)
    elif mode == 'LinearSVC':
        dual = kwargs.get('dual', False)
        max_iter = kwargs.get('max_iter', 1000) # 收斂的最大迭代數
        model = LinearSVC(dual = dual, max_iter = max_iter, random_state = 666)
    elif mode == 'KNN': # KNN分類器(最近鄰居數K=1)
        n_neighbors = kwargs.get('n_neighbors', 5)
        model = KNeighborsClassifier(n_neighbors = n_neighbors)
    return model

## In[產生模擬資料]:

def Data_generator(model, train_size = 100, test_size = 1000, target_accuracy = 0.8, randomSort = False, show = True, d = 10, difficulty = 1, important_dims = 1):
    _param = 1.0
    for _ in range(500): # 最多迭代100次
        x_raw = np.random.multivariate_normal(mean=np.zeros(d), cov = np.eye(d), size = train_size + test_size)
        # multivariate_normal 多元常態分佈, 產生一個常態分佈的5100*50的矩陣

        _, y_raw, _, _ = label_generator(x_raw, problem = 'classification', param = _param, difficulty = difficulty, important = important_dims)
        # 產生正確答案 (分類的標籤)

        if np.sum(y_raw[:train_size]) in [len(y_raw[:train_size]), 0]:
            print("all one class")
        else:
            model.fit(x_raw[:train_size], y_raw[:train_size]) # 訓練分類模型
            test_acc = model.score(x_raw[train_size:], y_raw[train_size:]) # 模型準確度評估
            if show:
                print("Model Accuracy: ", test_acc)
            if test_acc > target_accuracy: # 迭代至模型準確度>target_accuracy後跳出
                break
        _param *= 1.1 # 答案縮放係數, _param越大, 轉換後的值越接近邊緣 (趨近0或1), 使得誤差會逐漸下降?
    if show:
        print('Performance using the whole training set = {0:.2f}'.format(test_acc))
    if randomSort:
        rd = random.sample(range(0, train_size), train_size)
        x_raw[:train_size] = x_raw[rd]
        y_raw[:train_size] = y_raw[rd]
    return x_raw, y_raw

## In[主要方法的類別]:

class SupportedSV(object): # 模型訓練Class
    def __init__(self, x_raw, y_raw, size, model, cut, K = 1):
        self.size = size # 訓練資料量
        self.x_train, self.y_train = x_raw[:size], y_raw[:size]
        self.x_test, self.y_test = x_raw[size:], y_raw[size:]
        self.model = model
        self.K = K # 採用KNN模型時才需要設定數值
        self.combine(cut) # 原始資料分組分聚合

    def initialize(self, threshold = 0, getSV = False, getAC = False, reboot = True): # 輸出變數初始化
        self.trained_count = 0
        self.getSV = getSV # 同步計算SV
        self.getAC = getAC # 同步計算準確度分布
        self.cmb = list() # 輸出模型的訓練集
        self.cmb_acc = list() # 輸出模型的準確度
        self.cmb_len = list() # 輸出模型的大小分布
        self.trainedTime = np.zeros(0, float) # 輸出模型的實際訓練時間
        self.selected_trainedTime = np.zeros(0, float) # 輸出模型的實際訓練時間
        if reboot: # 重新走訪與訓練模型實例
            self.full_score = -1 # 完整模型的準確度
            self.all_cmb = list() # 所有模型實例的訓練集
            self.all_cmb_acc = list() # 所有模型實例的準確度
            self.all_cmb_len = list() #  所有模型實例的大小分布
            self.all_trainedTime = np.zeros(0, float) # 所有模型實例的實際訓練時間
        for i in range(0, self.train_size):
            self.cmb_len.append(np.zeros(0, int))
            if reboot:
                self.all_cmb_len.append(np.zeros(0, int))
        self.vt0 = np.max(np.bincount(self.y_test).astype(float)/len(self.y_test)) # 訓練集為空集合的模型準確度
        if getSV:
            self.cmb_utility = np.zeros((0, self.train_size), float) # 輸出模型的邊際貢獻矩陣(準確度差值/組合係數)
            self.cmb_marginal = np.zeros((0, self.train_size), float) # 輸出模型的邊際貢獻矩陣(準確度差值)
        self.acc_count = np.zeros(50, int) # 輸出模型的準確度分布
        self.vt_base = threshold # 輸出模型的準確度門檻(舊功能)

    def get_RawData(self, split = False): # 輸出原始資料
        if split:
            return self.x_train, self.y_train, self.x_test, self.y_test
        else:
            return np.concatenate([self.x_train, self.x_test]), np.concatenate([self.y_train, self.y_test])

    def combine(self, cut): # 原始資料分組分聚合
        alist = list()
        sta = 0
        for set_size in cut:
            end = int(set_size)
            alist.append(np.arange(start = sta, stop = end, step = 1))
            sta = end
        self.source = alist
        self.train_size = len(self.source)

    def utility_adder(self, cmb_loc, utility): # 紀錄模型的效用與邊際貢獻
        if self.getSV and cmb_loc == len(self.cmb)-1:
            freq = binomial_coefficients(self.train_size-1, len(self.cmb[cmb_loc])-1)*self.train_size
            self.cmb_utility = np.concatenate([self.cmb_utility, [utility/freq]])
            self.cmb_marginal = np.concatenate([self.cmb_marginal, [utility]])

    def all_cmb_adder(self, score, idx, len_loc, tt = 0): # 紀錄模型實例的所有資訊
        cmb_loc = len(self.all_cmb)
        self.all_cmb.append(idx) # 紀錄此模型的訓練集
        self.all_cmb_acc.append(score) # 紀錄此模型的準確度
        self.all_cmb_len[len_loc] = np.append(self.all_cmb_len[len_loc], cmb_loc) # 將此模型的idx加至其長度所屬的紀錄陣列中
        self.all_trainedTime = np.append(self.all_trainedTime, tt) # 記錄此模型的訓練時間

    def cmb_adder(self, score, idx, cmb_loc, len_loc, tt = 0): # 紀錄版本模型的所有資訊
        if cmb_loc == len(self.cmb):
            if self.getAC:
                acc_idx = ftoi(score, 100)-50
                self.acc_count[acc_idx] += 1 # 該準確度的出現次數+1
            self.cmb.append(idx) # 紀錄此模型的訓練集
            self.cmb_acc.append(score) # 紀錄此模型的準確度
            self.cmb_len[len_loc] = np.append(self.cmb_len[len_loc], cmb_loc) # 將此模型的idx加至其長度所屬的紀錄陣列中
            self.selected_trainedTime = np.append(self.selected_trainedTime, tt) # 記錄此模型的訓練時間

    def unfold(self, sv_avg, remain_idx): # 展開SV(將資料集的SV均分給每一筆資料)
        vals = np.zeros(self.size, float)
        for i in range(0, len(remain_idx)):
            vals[self.source[remain_idx[i]]] = sv_avg[i]
        return vals

    def Get_ShapleyValue(self, NDU = False, getAC = False): # 計算SV精確值
        if NDU: # Non-disutility SV 非負效用集SV
            self.cmb_utility = np.zeros((0, self.train_size), float)
            self.cmb_marginal = np.zeros((0, self.train_size), float)
        else: # TrueSV 原始SV
            self.initialize(getAC = getAC)
            self.full_score, _ = self.retrain_model(self.x_train, self.y_train) # 原始模型的準確度
            self.full_idx = np.array(range(0, self.train_size)) # 完整的訓練集組idx
        self.sv = np.zeros(self.train_size, float) # 總累積SV
        self.all_cmb, self.all_cmb_acc, self.all_cmb_len = self.RandomTraining()
        for len_idx in reversed(range(0, len(self.all_cmb_len))):
            coef = binomial_coefficients(self.train_size-1, len_idx)*self.train_size # 分母係數
            for S_xi in self.all_cmb_len[len_idx]:
                utility = np.zeros(self.train_size, float) # 單個模型提供的SV
                S_xi_set = set(self.all_cmb[S_xi])
                S_xi_val = self.all_cmb_acc[S_xi]
                if len_idx == 0:
                    xi = list(S_xi_set)[0]
                    utility[xi] = S_xi_val - self.vt0
                else:
                    count = 0
                    for S in self.all_cmb_len[len_idx-1]:
                        S_set = set(self.all_cmb[S])
                        xi_set = S_xi_set - S_set
                        if len(xi_set) == 1:
                            count += 1
                            xi = list(xi_set)[0]
                            utility[xi] = S_xi_val - self.all_cmb_acc[S]
                            if count == len_idx+1:
                                break
                if NDU and min(utility) >= 0:
                    self.cmb_utility = np.concatenate([self.cmb_utility, [utility/coef]])
                    self.cmb_marginal = np.concatenate([self.cmb_marginal, [utility]])
                    self.cmb_len[len_idx] = np.append(self.cmb_len[len_idx], len(self.cmb))
                    self.cmb.append(self.all_cmb[S_xi]) # 紀錄此模型點的訓練集組合
                    self.cmb_acc.append(self.all_cmb_acc[S_xi]) # 紀錄此模型點的準確度
                self.sv += (utility/coef)
        if not NDU:
            sv_uf = self.unfold(self.sv, remain_idx = np.arange(0, self.train_size))
            return self.sv, sv_uf, self.acc_count

    def RandomTraining(self, k = 1, pref = 0, getAC = False):
    # 模型實例隨機抽樣(k輸出模型數, pref隨機抽樣範圍: 0=[0%:100%], 1=[50%:100%], -1=[0%:50%])
        cmb = list([self.full_idx])
        cmb_acc = list([self.full_score])
        cmb_len = list()
        for i in range(0, self.train_size-1):
            cmb_len.append(np.zeros(0, int))
        cmb_len.append(np.zeros(1, int))
        allcmb = list()
        sample_idx = np.zeros(0, int)
#        for S_size in range(1, self.train_size+1):
#            allcmb += list(combinations(np.arange(self.train_size), S_size))
#        sample_idx = random.sample(range(len(allcmb)), len(allcmb))
        for S_size in range(1, self.train_size):
            SameSizeCMB = list(combinations(np.arange(self.train_size), S_size))
            if pref != 0:
                lb = len(allcmb)
                rb = lb + len(SameSizeCMB)
                sample_idx = np.append(sample_idx, random.sample(range(lb, rb), len(SameSizeCMB)))
            allcmb += SameSizeCMB
        if pref == 0:
            sample_idx = random.sample(range(len(allcmb)), len(allcmb))
        elif pref > 0:
            sample_idx = sample_idx[::-1]
        if k > 0 and k <= 1:
            n = int((len(allcmb)+1)*k)
        elif k > len(allcmb)+1:
            n = len(allcmb)+1
        else:
            n = int(k)
        for cmb_idx in sample_idx:
            remain_idx = np.array(allcmb[cmb_idx])
            raw_idx = np.zeros(0, int)
            for idx in remain_idx:
                raw_idx = np.append(raw_idx, self.source[idx])
            x_new = self.x_train[raw_idx] # 代入訓練集組的特徵
            y_new = self.y_train[raw_idx] # 代入訓練集組的標籤
            now_score = self.vt0
            if len(x_new) >= self.K:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if len(set(y_new)) == len(set(self.y_test)):
                        now_score, _ = self.retrain_model(x_new, y_new)
                        if now_score >= self.vt_base:
                            cmb.append(remain_idx) # 紀錄此模型點的訓練集組合
                            cmb_acc.append(now_score) # 紀錄此模型點的準確度
                            cmb_len[len(remain_idx)-1] = np.append(cmb_len[len(remain_idx)-1], len(cmb)-1) # 將此模型點的idx加至其長度所屬的紀錄陣列中
                            if getAC:
                                acc_idx = ftoi(now_score, 100)-50
                                self.acc_count[acc_idx] += 1 # 該準確度的出現次數+1
            if len(cmb) >= n:
                break
        return cmb, cmb_acc, cmb_len

    def Sequence_based_Model_Training(self, BFS = True, Sorted_Enqueue = False, RD = False, k = 1, reboot = True):
    # 基於序列的模型訓練方法(BFS是否採用深度優先走訪, Sorted_Enqueue子模型放入佇列前是否要根據準確度排序, RD從佇列中隨機選擇子模型,
    #                      k輸出模型數, reboot重啟模型訓練而不沿用先前訓練好的模型實例)
        queue = list([(self.full_idx, self.full_score, self.full_tt)])
        EnqueueTT = np.append(np.zeros(0, float), self.full_tt)
        cmb_len = list()
        for i in range(0, self.train_size):
            cmb_len.append(np.zeros(0, int))
        if reboot:
            self.all_cmb_adder(self.full_score, self.full_idx, self.train_size-1, self.full_tt)
            cmb_len[-1] = np.append(cmb_len[-1], len(self.all_cmb)-1)
        else:
            idx = self.all_cmb_len[-1][0]
            cmb_len[-1] = np.append(cmb_len[-1], idx)
        batch_end = len(queue)
        n = int(k)
        while len(queue) > 0:
            SubModel = list()
            if BFS:
                if RD:
                    S_xi_idx = random.sample(range(0, batch_end), 1)[0]
                    batch_end -= 1
                    queue_tuple = queue.pop(S_xi_idx)
                else:
                    queue_tuple = queue.pop(0)
            else:
                if RD:
                    sta = 0
                    for q in reversed(range(1, len(queue))):
                        if len(queue[q][0]) < len(queue[q-1][0]):
                            sta = q
                            break
                    S_xi_idx = random.sample(range(sta, len(queue)), 1)[0]
                    queue_tuple = queue.pop(S_xi_idx)
                else:
                    queue_tuple = queue.pop()
            S_xi = queue_tuple[0]
            S_xi_set = set(S_xi)
            S_xi_acc = queue_tuple[1]
            S_xi_loc = len(S_xi)-1
            S_xi_tt = queue_tuple[2]
            utility = np.zeros(self.train_size, float)
            if S_xi_loc == 0:
                xi = S_xi[0]
                utility[xi] = S_xi_acc - self.vt0
            else:
                xi_exist = np.zeros(self.train_size, int)
                xi_exist[S_xi] += 1
                for idx in self.all_cmb_len[S_xi_loc-1]: # 檢查S是否已訓練過
                    S_set = set(self.all_cmb[idx])
                    xi_set = S_xi_set - S_set
                    if len(xi_set) == 1:
                        xi = list(xi_set)[0]
                        utility[xi] = S_xi_acc - self.all_cmb_acc[idx]
                        xi_exist[xi] -= 1
                        if idx not in cmb_len[S_xi_loc-1]:
                            if not reboot:
                                EnqueueTT = np.append(EnqueueTT, self.all_trainedTime[idx])
                            if self.all_cmb_acc[idx] >= self.vt_base:
                                SubModel.append((self.all_cmb[idx], self.all_cmb_acc[idx], idx, self.all_trainedTime[idx]))
                        if sum(xi_exist) == 0:
                            break
                for xi in S_xi: # 訓練未訓練過的S
                    if xi_exist[xi] == 1:
                        S = np.setdiff1d(S_xi, xi, False)
                        raw_idx = np.zeros(0, int)
                        for idx in S: # 展開訓練集在分組前的原始資料
                            raw_idx = np.append(raw_idx, self.source[idx])
                        x_new = self.x_train[raw_idx] # 代入訓練集組的特徵
                        y_new = self.y_train[raw_idx] # 代入訓練集組的標籤
                        if len(x_new) >= self.K:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                if len(set(y_new)) == len(set(self.y_test)):
                                    S_acc, tt = self.retrain_model(x_new, y_new)
                        utility[xi] = S_xi_acc - S_acc
                        if not reboot:
                            EnqueueTT = np.append(EnqueueTT, tt)
                        if S_acc >= self.vt_base:
                            SubModel.append((S, S_acc, -1, tt))
            if Sorted_Enqueue:
                SubModel = sorted(SubModel, key = lambda x: x[1])
                if BFS:
                    SubModel = SubModel[::-1]
            for subM in SubModel:
                queue.append((subM[0], subM[1], subM[3]))
                if subM[2] != -1:
                    cmb_len[S_xi_loc-1] = np.append(cmb_len[S_xi_loc-1], subM[2])
                else:
                    self.all_cmb_adder(subM[1], subM[0], S_xi_loc-1, subM[3])
                    cmb_len[S_xi_loc-1] = np.append(cmb_len[S_xi_loc-1], len(self.all_cmb)-1)
            if RD and BFS and batch_end == 0:
                batch_end = len(queue)
            if min(utility) >= 0 and sum(utility) > 0: # 儲存符合篩選條件的版本模型
                cmb_idx = len(self.cmb)
                self.cmb_adder(S_xi_acc, S_xi, cmb_idx, S_xi_loc, S_xi_tt)
                if self.getSV:
                    self.utility_adder(cmb_idx, utility)
                if k != 1 and len(self.cmb) >= n:
                    break
        if not reboot:
            self.trainedTime = EnqueueTT
            self.trained_count = len(EnqueueTT)

    def RandomPick(self, k = 1, reboot = False): # 隨機版本模型走訪與輸出
        if reboot:
            self.Model_Selection()
        cmb = list()
        cmb_acc = list()
        cmb_len = list()
        for i in range(0, self.train_size):
            cmb_len.append(np.zeros(0, int))
        cmb_utility = np.zeros((0, self.train_size), float)
        cmb_marginal = np.zeros((0, self.train_size), float)
        runtime = 0
        if k > 0 and k <= 1:
            n = int(len(self.cmb)*k)
        elif k > len(self.cmb):
            n = len(self.cmb)
        else:
            n = int(k)
        sample_idx = random.sample(range(0, len(self.cmb)), len(self.cmb))
        for idx in sample_idx[:n]:
            cmb_len[len(self.cmb[idx])-1] = np.append(cmb_len[len(self.cmb[idx])-1], len(cmb))
            cmb.append(self.cmb[idx])
            cmb_acc.append(self.cmb_acc[idx])
            cmb_utility = np.concatenate([cmb_utility, [self.cmb_utility[idx,:]]])
            cmb_marginal = np.concatenate([cmb_marginal, [self.cmb_marginal[idx,:]]])
            runtime += self.selected_trainedTime[idx]
        # print('\n[Model Sequence Curve] ML_', len(cmb), ': ', runtime, ' sec.\n', sep='')
        return cmb, cmb_acc, cmb_len, cmb_utility, cmb_marginal, runtime

    def Model_Selection(self, BFS = True, SEQ = False, RD = False, threshold = 0, k = 1, reboot = True):
        self.initialize(getSV = True, getAC = True, reboot = reboot)
        self.vt_base = threshold # 準確度低標
        if reboot:
            self.full_score, self.full_tt = self.retrain_model(self.x_train, self.y_train) # 模型初始準確度
            self.full_idx = np.array(range(0, self.train_size)) # 完整的訓練集組idx
        self.Sequence_based_Model_Training(BFS = BFS, Sorted_Enqueue = SEQ, RD = RD, k = k, reboot = reboot)
        return self.cmb, self.cmb_acc, self.cmb_len, self.cmb_utility, self.cmb_marginal

    def reset_model(self, model): # 重置模型變數
        self.model = clone(model)

    def retrain_model(self, x_batch, y_batch): # 訓練模型與評估準確度
        starttime = time.time()
        self.trained_count += 1
        self.reset_model(self.model)
        self.model.fit(x_batch, y_batch)
        score = self.model.score(self.x_test, self.y_test)
        tt = time.time() - starttime
        self.trainedTime = np.append(self.trainedTime, tt)
        return score, tt

    def plot(self): # 繪製模型實例的準確度散布圖
        if len(self.cmb_acc) > 0:
            cmb_len = np.zeros(0, int)
            for m in self.cmb:
                cmb_len = np.append(cmb_len, len(m))
            imgdf = pd.DataFrame(self.cmb_acc, columns = ['acc'])
            imgdf['len'] = cmb_len
            img = imgdf.plot(x = 'len', y = 'acc', kind = 'scatter')
            img.set_xlabel('Training Set Size')
            img.set_ylabel('Prediction Accuracy')
            img.set_title('Training Set Size and Accuracy Distribution of Models')
            plt.show()
            img.figure.savefig('Models_Distribution_Scatter.jpg', bbox_inches = 'tight')
            img.remove()

# In[外部資料集讀寫]:

def Read_RawCSV(xfile = 'x_raw101.csv', yfile = 'y_raw101.csv'):
    x_raw = pd.read_csv(xfile).values
    y_raw = np.reshape(pd.read_csv(yfile).values, (1,-1))[0]
    return x_raw, y_raw

def Write_RawCSV(ClassObj, xfile = 'x_raw.csv', yfile = 'y_raw.csv'):
    x_raw, y_raw = ClassObj.get_RawData(False)
    pd.DataFrame(x_raw).to_csv(xfile, index = False, header = True)
    pd.DataFrame(y_raw).to_csv(yfile, index = False, header = True)

# In[模型訓練]:

def RunMS(ClassObj, search = 0, SEQ = True, RD = False, threshold = 0, k = 1, reboot = True, iterate = False):
    if search != 2:
        starttime = time.time()
        BFS = (search == 0)
        bf_cmb, bf_cmb_acc, bf_cmb_len, bf_utility, bf_marginal = ClassObj.Model_Selection(BFS, SEQ, RD, threshold, k, reboot)
        runtime = time.time() - starttime 
        if not reboot:
            runtime += sum(ClassObj.trainedTime)
    else:
        bf_cmb, bf_cmb_acc, bf_cmb_len, bf_utility, bf_marginal, runtime = ClassObj.RandomPick(k = k, reboot = reboot) # 隨機篩選模型
    ClassObj.plot()
    print('Trained Models:', ClassObj.trained_count)
    if not iterate:
        print('\n[Model Selection] ML_', len(bf_cmb), ': ', runtime, ' sec.\n', sep = '')
    return bf_cmb, bf_cmb_acc, bf_cmb_len, bf_utility, bf_marginal, runtime

# In[計算訓練集的真實SV]:

def RunSV(ClassObj, getAC = False):
    starttime = time.time()
    fm_vals, fm_vals_uf, fm_acc_count = ClassObj.Get_ShapleyValue(getAC = getAC)
    runtime = time.time() - starttime
    print('Trained Models:', ClassObj.trained_count)
    print('\n[True Shapley]: ', runtime, ' sec.\n', sep='')
    return fm_vals, fm_vals_uf, fm_acc_count

# In[儲存目前的類別物件]:

def RW_ClassObj(obj = None, wtire = False, dirN = 'cifar-10-batches-py/Classification', name = 'ClassObj', date = 'today', batch = ''):
    dir_name = dirN + '/' + date + '/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if len(batch) > 0:
        dir_name += batch + '/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name) 
    if wtire:
        with open(dir_name + name, 'wb') as file:
            pickle.dump(obj, file)
    else:
        with open(dir_name + name, 'rb') as file:
            obj = pickle.load(file)
        return obj

# In[假設曲線範例]:

def BG(reverse = False): # 準確度預算函數
    V_acc = np.ones(50, float)
    for i in range(0, 500):
        temp = (1 + i*0.1)**2
        if i%10 == 0:
            V_acc[i//10] = temp
    if reverse:
        V_acc = (V_acc*-1 + max(V_acc))[::-1] # 上下凸反轉
    V_acc = ftoi(V_acc, 100, True)/100 + 0
    # pd.DataFrame([V_acc]).transpose().plot()
    return V_acc

def DM(reverse = False): # 準確度需求分布函數
    db = np.random.normal(0, 1, 5000)
    # _, dbr = np.histogram(db, bins = 74)
    dbr = plt.hist(db, 74)[1]
    y0 = scipy.stats.norm.pdf(dbr, 0, 1)
    if reverse:
        D_acc = y0*-1 + min(y0) + max(y0) #垂直翻轉
        mid = np.argmin(D_acc)
    else:
        D_acc = y0
        mid = np.argmax(D_acc)
    if mid < 25:
        lb = 0
        rb = lb + 50
    elif mid > 75:
        rb = 100
        lb = rb - 25
    else:
        lb = mid-25
        rb = lb + 50
    D_acc = D_acc[lb:rb]
    # pd.DataFrame([D_acc]).transpose().plot()
    return D_acc

def PlotExample(V_acc, D_acc, disp = 0, style = 0, fontsize = 12):
    move = ftoi(disp, 100)
    if move > 0:
        D = np.append(np.repeat(D_acc[0], move, axis=0), D_acc[:-move])
    elif move < 0:
        D = np.append(D_acc[-move:], np.repeat(D_acc[-1], -move, axis=0))
    else:
        D = D_acc
    D = 100*D/sum(D)
    V = V_acc
    R = sum(V*D/100)
    acc = np.arange(0.5, 1, step = 0.01)
    df = pd.DataFrame({'Accuracy':acc, 'Budget':V, 'Demand':D})
    if style == 0:
        img = df.plot(kind = 'line', x = 'Accuracy', y = 'Budget', color = 'Blue', legend = False, fontsize = fontsize-2)
        img2 = df.plot(kind = 'line', x = 'Accuracy', y = 'Demand', secondary_y = True, color = 'Red', ax = img, legend = False, fontsize = fontsize-2)
        img.set_title('Accuracy Budget and Demand Distribution', fontsize = fontsize)
        img.set_xlabel('Prediction Accuracy', fontsize = fontsize)
        img.set_ylabel('Buyer Budget', fontsize = fontsize)
        img.yaxis.label.set_color('Blue')
        img.tick_params(axis = 'y', colors='Blue')
        img2.set_ylabel('Buyer Distribution')
        img2.yaxis.label.set_color('Red')
        img2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 1))
        img2.tick_params(axis = 'y', colors='Red')
        img.axhline(y = R, color = 'purple', linestyle = '--', lw = 2)
        plt.tight_layout()
        img.figure.savefig('Accuracy_Budget_&_Demand_Distribution.jpg', bbox_inches = 'tight', dpi = 300)
    elif style == 1:
        df['Weighted_Revenue'] = V*D/100
        img = df.plot(kind = 'line', x = 'Accuracy', y = 'Weighted_Revenue', color = 'Purple', legend = False, fontsize = fontsize-2)
        # img2 = df.plot(kind = 'line', x = 'Accuracy', y = 'Demand', secondary_y = True, color = 'Red', ax = img, legend = False, fontsize = fontsize-2)
        img.set_title('Upper Bound of Revenue', fontsize = fontsize)
        img.set_xlabel('Prediction Accuracy', fontsize = fontsize)
        img.set_ylabel('Revenue', fontsize = fontsize)
        img.yaxis.label.set_color('Purple')
        img.tick_params(axis = 'y', colors='Purple')
        # img2.set_ylabel('Buyer Distribution')
        # img2.yaxis.label.set_color('Red')
        # img2.tick_params(axis = 'y', colors='Red')
        plt.tight_layout()
        img.figure.savefig('Upper_Bound_of_Revenue.jpg', bbox_inches = 'tight', dpi = 300)
    plt.show()
    img.remove()

def SurveyExample(budget_reverse = False, demand_reverse = False, disp = 0, style = 0, fontsize = 14):
    V_acc = BG(budget_reverse)
    D_acc = DM(demand_reverse)
    PlotExample(V_acc = V_acc, D_acc = D_acc, disp = disp, style = style, fontsize = fontsize)
