
import os
import time
import numpy as np
import random
import pickle
import TPMP_ModelTraining as MT

img_size = 32*32
clf = MT.return_model('logistic', tol = 0.1, solver = 'liblinear', n_jobs = 1) # 以logistic建立分類模型
dirN = 'cifar-10-batches-py/Classification'

def ReadImage(filename): # 讀原始檔
    rawdata = dict()
    x_raw = np.zeros((0, 3072), int)
    y_batch = list()
    imgName = list()
    for file in filename:
        with open(file, 'rb') as f:
            rawdata = pickle.load(f, encoding = 'bytes')
            x_raw = np.append(x_raw, rawdata[b'data'], axis = 0)
            y_batch += rawdata[b'labels']
            imgName += rawdata[b'filenames']
    y_raw = np.array(y_batch)
    return x_raw, y_raw, imgName

def RW_Segment(x_raw = np.zeros(0, int), pName = list(), wtire = False, date = 'today', batch = '0'): # 讀寫分類檔
    global dirN
    dir_name = dirN + '/' + date
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    dir_name += '/' + batch
    if not os.path.exists(dir_name):
        os.mkdir(dir_name) 
    if wtire:
        with open(dir_name+'/x_raw_'+batch, 'wb') as file:
            pickle.dump(x_raw, file)
        with open(dir_name+'/img_name_'+batch, 'wb') as file:
            pickle.dump(pName, file)
    else:
        with open(dir_name+'/x_raw_'+batch, 'rb') as file:
            xSegment = pickle.load(file)
            print('x_raw_'+batch,'Size =', len(xSegment))
        with open(dir_name+'/img_name_'+batch, 'rb') as file:
            pName = pickle.load(file)
            print('imgName_'+batch,'Size =', len(pName))
        return xSegment, pName

def SegmentSave(x_raw, y_raw, imgName, date = 'today'): # 依分類寫檔
    global img_size
    idxlist = list()
    for i in range(0, 10):
        idxlist += [np.zeros(0, int)]
    for img_idx in range(0, len(y_raw)):
        idxlist[y_raw[img_idx]] = np.append(idxlist[y_raw[img_idx]], img_idx)
    for img_loc in range(0, len(idxlist)):
        pixel = x_raw[idxlist[img_loc],:]
        pName = list()
        for img_idx in idxlist[img_loc]:
            pName += [imgName[img_idx]]
        RW_Segment(pixel, pName, wtire = True, date = date, batch = str(img_loc))

def GroupingRules(x_train, y_train): # 依標籤分組資料集
    global img_size
    x_raw = np.zeros((0, img_size*3), float)
    y_raw = np.zeros(0, int)
    train_size = np.size(x_train, axis = 0) # 訓練: 鳥2 x1500, 飛機0 x750, 船8 x750; 測試: 鳥500, 飛機250, 船250; 是鳥非鳥的二元分類
    group_size = 10
    x0_start = 0
    x1_start = train_size//2
    x2_start = x1_start + train_size//4
    batch = np.arange(1, 10, step = 2)
    for i in reversed(batch):
        x0_end = x0_start + (train_size//group_size)*i//10
        x1_end = x1_start + (train_size//group_size) - (x0_end - x0_start)
        x_raw = np.append(x_raw, x_train[x0_start:x0_end], axis = 0)
        x_raw = np.append(x_raw, x_train[x1_start:x1_end], axis = 0)
        y_raw = np.append(y_raw, y_train[x0_start:x0_end])
        y_raw = np.append(y_raw, y_train[x1_start:x1_end])
        # print('x0_start:', x0_start, ' x0_end:', x0_end)
        # print('x1_start:', x1_start, ' x1_end:', x1_end)
        x0_start = x0_end
        x1_start = x1_end
        x0_end = x0_start + (train_size//group_size)*i//10
        x2_end = x2_start + (train_size//group_size) - (x0_end - x0_start)
        x_raw = np.append(x_raw, x_train[x0_start:x0_end], axis = 0)
        x_raw = np.append(x_raw, x_train[x2_start:x2_end], axis = 0)
        y_raw = np.append(y_raw, y_train[x0_start:x0_end])
        y_raw = np.append(y_raw, y_train[x2_start:x2_end])
        # print('x0_start:', x0_start, ' x0_end:', x0_end)
        # print('x2_start:', x2_start, ' x2_end:', x2_end)
        x0_start = x0_end
        x2_start = x2_end
    return x_raw, y_raw

def RW_RawData(x_raw = np.zeros(0, int), y_raw = np.zeros(0, int), wtire = False, date = 'today', batch = '0', acc = 0.0): # 讀寫分組資料集
    global dirN
    dir_name = dirN + '/' + date
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    dir_name += '/m' + batch
    if not os.path.exists(dir_name):
        os.mkdir(dir_name) 
    if wtire:
        with open(dir_name+'/x_raw', 'wb') as file:
            pickle.dump(x_raw, file)
        with open(dir_name+'/y_raw', 'wb') as file:
            pickle.dump(y_raw, file)
        with open(dir_name+'/acc', 'wb') as file:
            pickle.dump(acc, file)
    else:
        with open(dir_name+'/x_raw', 'rb') as file:
            xSegment = pickle.load(file)
            print('x_raw Size =', len(xSegment))
        with open(dir_name+'/y_raw', 'rb') as file:
            ySegment = pickle.load(file)
            print('y_raw Size =', len(ySegment))
        with open(dir_name+'/acc', 'rb') as file:
            acc = pickle.load(file)
            print('m' + batch, 'acc =', acc)
        return xSegment, ySegment, acc

def SegmentSplit(times = 1, date = 'today', labels = [], train_batch_size = [], test_batch_size = [], normalize = False): # 分組資料集分割
    test_idx0 = 5000
    raw = 'Raw'
    for i in range(0, len(labels)):
        if i == 0:
            x_raw, _ = RW_Segment(batch = str(labels[i]), date = raw)
        else:
            x_raw_batch, _ = RW_Segment(batch = str(labels[i]), date = raw)
            x_raw = np.dstack((x_raw, x_raw_batch))
    batch_size = test_idx0//train_batch_size[0]
    batch_times = int(np.ceil(times/batch_size))
    x_test = np.zeros((0, img_size*3), float)
    y_test = np.zeros(0, int)
    rd_test = random.sample(range(test_idx0, x_raw.shape[0]), x_raw.shape[0] - test_idx0)
    for t in range(0, batch_times):
        rd = random.sample(range(0, test_idx0), test_idx0)
        for j in range(0, batch_size):
            x_train = np.zeros((0, img_size*3), float)
            y_train = np.zeros(0, int)
            for i in range(0, len(labels)):
                if j == 0:
                    x_raw[0:test_idx0,:, i] = x_raw[rd,:, i]
                    if t == 0:
                        x_raw[test_idx0:,:, i] = x_raw[rd_test,:, i]
                x_train = np.append(x_train, x_raw[train_batch_size[i]*j:train_batch_size[i]*(j+1),:, i], axis = 0)
                if i == 0:
                    y_train = np.append(y_train, np.ones(train_batch_size[i], int))
                else:
                    y_train = np.append(y_train, np.zeros(train_batch_size[i], int))
                if t == 0 and j == 0:
                    x_test = np.append(x_test, x_raw[test_idx0:test_idx0+test_batch_size[i],:, i], axis = 0)
                    if i == 0:
                        y_test  = np.append(y_test, np.ones(test_batch_size[i], int))
                    else:
                        y_test  = np.append(y_test, np.zeros(test_batch_size[i], int))
            if normalize:
                x_train /= 255
                if t == 0 and j == 0:
                    x_test /= 255
            starttime = time.time()
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            runtime = time.time() - starttime
            num = t*batch_size + j
            print('batch', num, ', acc: ', score, ', runtime: ', runtime, ' sec', sep = '')
            x_raw_m, y_raw_m = GroupingRules(x_train, y_train)
            x_raw_m = np.append(x_raw_m, x_test, axis = 0)
            y_raw_m = np.append(y_raw_m, y_test)
            RW_RawData(x_raw = x_raw_m, y_raw = y_raw_m, wtire = True, date = date, batch = str(num), acc = score)
