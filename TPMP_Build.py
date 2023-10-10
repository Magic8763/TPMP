
import os
# os.chdir()
import time
import numpy as np
import threading
import TPMP_ModelTraining as MT
import TPMP_Preprocessing as PP

# In[CIFAR-10資料集的前處理]:

def Build_Image_Segments(times, date, labels, train_batch_size, test_batch_size, normalize = False):
    PP.dirN = 'cifar-10-batches-py/Classification'
    filename = ['cifar-10-batches-py/data_batch_1',
                'cifar-10-batches-py/data_batch_2',
                'cifar-10-batches-py/data_batch_3',
                'cifar-10-batches-py/data_batch_4',
                'cifar-10-batches-py/data_batch_5',
                'cifar-10-batches-py/test_batch']
    x_raw, y_raw, imgName = PP.ReadImage(filename = filename) # 讀原始檔
    PP.SegmentSave(x_raw, y_raw, imgName, date = 'img') # 依分類寫檔
    PP.SegmentSplit(times = times, date = date,
                    labels = labels, train_batch_size = train_batch_size,
                    test_batch_size = test_batch_size, normalize = normalize) # 分組資料集分割

# In[實驗:模型訓練(有外部模型訓練物件存在時可以省略不做)]:

def Threading_Training(train_size, day, cut, search, dir_name, t):
    x_raw, y_raw, acc = PP.RW_RawData(date = day, batch = str(t)) # 讀分組資料集
    Pre_Work = MT.SupportedSV(x_raw = x_raw, y_raw = y_raw, size = train_size,
                              model = clf, K = 1, cut = cut)
    _, _, _, _, _, _ = MT.RunMS(ClassObj = Pre_Work, search = search,
                                k = 1, reboot = True) # 輸出模型數, 0~1為百分比(0~100%), 1以上為個數
    MT.RW_ClassObj(obj = Pre_Work, wtire = True, dir_name = dir_name, name = 'ClassObj',
                   date = day, batch = 'm'+str(t))

def Training_Model_Instance(total, date = [], thread_count = 5):
    train_size = 2000 # test_size = 3000-train_size = 1000
    group_size = 10
    cut = np.arange(start = 0, stop = train_size+1, step = train_size//group_size)[1:] # 資料集分割
    dir_name = 'var'
    search = 0 # 0: 寬度優先走訪, 1: 深度優先走訪, 2: 完全隨機走訪
    for day in date:
        starttime = time.time()
        for t in range(10, total, thread_count):
            threads = []
            for i in range(thread_count): # 建立多個子執行緒
                threads.append(threading.Thread(target=Threading_Training, args=(train_size, day, cut, search, dir_name, t+i)))
                threads[i].start()
            for j in range(thread_count): # 等待所有子執行緒結束
                threads[j].join()
        runtime = time.time()-starttime
        print('m0 ~ m', total, ' done! runtime: ', runtime, ' sec', sep = '')

# In[main]:

if __name__ == "__main__":
    mode = 'logistic' # 以logistic建立分類模型
    # mode = 'SVC' # 以SVC建立分類模型
    # mode = 'LinearSVC' # 以LinearSVC建立分類模型
    clf = MT.return_model(mode = mode, tol = 0.1, solver = 'liblinear', n_jobs = 1) # 以logistic建立分類模型物件clf
    build_date, batchObj = '1010', 20
    Build_Image_Segments(times = batchObj, date = build_date,
                         labels = [2, 0, 8], train_batch_size = [1000, 500, 500],
                         test_batch_size = [500, 250, 250])
    Training_Model_Instance(total = batchObj, date = [build_date], thread_count = 10)


