
import os
# os.chdir()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

strategies = ['Base', 'MaxBudget', 'MaxBudget_IneqWeighted', 'MinCover', 'MaxImprove', 'MaxImprove_Uion']
colors = ['tab:blue', 'purple', 'tab:green', 'tab:red', 'black', 'DarkOrange', 'tab:cyan', 'DarkBlue']
fontsize = 14

dir_name = 'var'
day = '1017'
batch = '[pNormal disp=0.0]'
# batch = '[nNormal disp=0.0]'
minK = 50
xlabel_inputs = [str(minK*i) for i in range(1, 6)]+['MAX']

# In[]:

def PCplot(search): # 定價方法效果效率繪圖
    color = [colors[i] for i in (0, 1)]
    fname, search = 'Pricing Efficiency Test', ' ('+search+')'
    PC_df = pd.read_csv(fname+search+'.csv', index_col = 0)
    PC_df.index = xlabel_inputs
    y_left = list(PC_df.columns[np.arange(3, 7, 3)])
    y_left2 = list(PC_df.columns[np.arange(2, 6, 3)])
    img = PC_df.plot(kind = 'line', y = y_left, color = color, style = 'o-', logy = True, fontsize = fontsize-2)
    img.set_title('Pricing Efficiency'+search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.set_ylabel('Runtime (s)', fontsize = fontsize)
    img.legend(labels = ['CIP', 'SLSQP'], fontsize = fontsize-2)

    img2 = PC_df.plot(kind = 'bar', y = y_left2, color = color, fontsize = fontsize-2)
    img2.set_title('Pricing Effectiveness'+search, fontsize = fontsize)
    img2.set_xlabel('Input Models', fontsize = fontsize)
    img2.set_ylabel('Revenue Ratio (%)', fontsize = fontsize)
    img2.legend(labels = ['CIP', 'SLSQP'], fontsize = fontsize-2)
    img2.set_ylim(37, 103)
    plt.setp(img2.get_xticklabels(), rotation = 0)

    plt.show()
    img.figure.savefig('Pricing_Efficiency'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img2.figure.savefig('Pricing_Effectiveness'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()
    img2.remove()

def RMplot(search): # 移除策略效果繪圖
    color = [colors[i] for i in (0, 2, 3)]
    strategy = [strategies[i] for i in (0, 3, 4)]
    fname, search = 'Optimizing Effectiveness Test', ' ('+search+')'
    df = pd.read_csv(fname+search+'.csv', index_col = 0)
    title, col_size, row_size = list(df.columns), len(strategy), len(xlabel_inputs)
    RM_mt = np.zeros((0, col_size), float)
    for j in range(3): # j = {0: 'Revenue', 1: 'Revenue Ratio (%)', 2: 'Runtime (s)'}
        for i in range(row_size):
            RM_mt = np.concatenate([RM_mt, [df[title[col_size*i:col_size*(i+1)]].iloc[j]]])
    for i in range(row_size):
        temp = df[title[col_size*i:col_size*(i+1)]].iloc[3] # 3: 'Quantity'
        removed = temp.iloc[0]-temp # Number of removed models
        RM_mt = np.concatenate([RM_mt, [removed]])
    UIperM = np.zeros(col_size, float) # Unit improvement per model
    for i in range(row_size):
        UIperM[1:] = (RM_mt[row_size+i, 1:]-RM_mt[row_size+i, 0])/RM_mt[row_size*3+i, 1:]
        RM_mt = np.concatenate([RM_mt, [UIperM]])
    for i in range(row_size):
        RM_mt = np.concatenate([RM_mt, [df[title[col_size*i:col_size*(i+1)]].iloc[4]]]) # 4: 'Iterations'

    RM_df = pd.DataFrame(RM_mt[:row_size,:], index = xlabel_inputs, columns = strategy)
    img = RM_df.plot(kind = 'bar', color = color, fontsize = fontsize-2)
    img.set_title('Optimizing Effectiveness'+search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.set_ylabel('Revenue', fontsize = fontsize)
    img.legend(loc = 'best', fontsize = fontsize-2)
    plt.setp(img.get_xticklabels(), rotation = 0)

    RM_df = pd.DataFrame(RM_mt[row_size:row_size*2,:], index = xlabel_inputs, columns = strategy)
    img1 = RM_df.plot(kind = 'bar', color = color, fontsize = fontsize-2)
    img1.set_title('Optimizing Effectiveness'+search, fontsize = fontsize)
    img1.set_xlabel('Input Models', fontsize = fontsize)
    img1.set_ylabel('Revenue Ratio (%)', fontsize = fontsize)
    img1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 1))
    img1.legend(loc = 'upper left', bbox_to_anchor = (0.6, 0.62), fontsize = fontsize-2)
    img1.set_ylim(37, 103)
    plt.setp(img1.get_xticklabels(), rotation = 0)

    RM_df = pd.DataFrame(RM_mt[row_size*2:row_size*3,:], index = xlabel_inputs, columns = strategy)
    img2 = RM_df.plot(kind = 'line', style = 'o-', color = color, fontsize = fontsize-2)
    img2.set_title('Optimizing Efficiency' + search, fontsize = fontsize)
    img2.set_xlabel('Input Models', fontsize = fontsize)
    img2.set_ylabel('Runtime (s)', fontsize = fontsize)
    img2.legend(loc = 'best', fontsize = fontsize-2)
    plt.setp(img2.get_xticklabels(), rotation = 0)

    RM_df = pd.DataFrame(RM_mt[row_size*3:row_size*4,:], index = xlabel_inputs, columns = strategy)
    img3 = RM_df.plot(kind = 'bar', y = strategy[1:], color = color[1:], fontsize = fontsize-2)
    img3.set_title('Removed Models' + search, fontsize = fontsize)
    img3.set_xlabel('Input Models', fontsize = fontsize)
    img3.set_ylabel('Number of Models', fontsize = fontsize)
    img3.legend(loc = 'best', fontsize = fontsize-2)
    plt.setp(img3.get_xticklabels(), rotation = 0)

    RM_df = pd.DataFrame(RM_mt[row_size*4:row_size*5,:], index = xlabel_inputs, columns = strategy)
    img4 = RM_df.plot(kind = 'bar', y = strategy[1:], color = color[1:], fontsize = fontsize-2)
    img4.set_title('Unit Improvement'+search, fontsize = fontsize)
    img4.set_xlabel('Input Models', fontsize = fontsize)
    img4.set_ylabel('Revenue Ratio (%)', fontsize = fontsize)
    img4.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 1))
    img4.legend(loc = 'best', fontsize = fontsize-2)
    plt.setp(img4.get_xticklabels(), rotation = 0)

    RM_df = pd.DataFrame(RM_mt[row_size*5:row_size*6,:], index = xlabel_inputs, columns = strategy)
    img5 = RM_df.plot(kind = 'bar', y = strategy, color = color, fontsize = fontsize-2)
    img5.set_title('Avg. Iterations of CIP'+search, fontsize = fontsize)
    img5.set_xlabel('Input Models', fontsize = fontsize)
    img5.set_ylabel('Iterations', fontsize = fontsize)
    img5.legend(loc = 'best', fontsize = fontsize-2)
    img5.set_ylim(1.45, 2.45)
    plt.setp(img5.get_xticklabels(), rotation = 0)

    plt.show()
    img.figure.savefig('Optimizing_Effectiveness'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img1.figure.savefig('Optimizing_Effectiveness_RevRatio'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img2.figure.savefig('Optimizing_Efficiency'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img3.figure.savefig('Optimizing_Removed'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img4.figure.savefig('Optimizing_UnitImprovement'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img5.figure.savefig('Optimizing_CIP_Iterations'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()
    img1.remove()
    img2.remove()
    img3.remove()
    img4.remove()
    img5.remove()

def PTplot(search): # 各階段耗時繪圖
    color = [colors[i] for i in (2, 0, 1, 3)]
    fname, search = ['Pricing Efficiency Test', 'Optimizing Effectiveness Test'], ' ('+search+')'
    labels = ['Training', 'Pricing (CIP)', 'Pricing (SLSQP)', 'Distribution']
    PT_mt = np.zeros((len(xlabel_inputs), len(labels)), float)

    df = pd.read_csv(fname[0]+search+'.csv', index_col = 0)
    PT_mt[:, 0:3] = df[['Training_Time', 'CIP_Runtime', 'SLSQP_Runtime']]

    df = pd.read_csv(fname[1]+search+'.csv', index_col = 0).T
    loc1 = np.arange(0, df.shape[0], 3)
    PT_mt[:, 3] = df['Runtime_P2 (s)'].iloc[loc1]

    PT_df = pd.DataFrame(PT_mt, index = xlabel_inputs, columns = labels)
    img = PT_df.plot(kind = 'line', style = 'o-', color = color, logy = True, fontsize = fontsize-2)
    img.set_title('The Runtime of Each Phase'+search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.set_ylabel('Runtime (s)', fontsize = fontsize)
    img.set_ylim(0.00005, 10000)
    img.legend(loc = 'best', labels = labels, fontsize = fontsize-2)
    plt.tight_layout()
    plt.show()
    img.figure.savefig('Phase_Runtime'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

def TNplot(): # 訓練效率繪圖
    color = [colors[i] for i in (5, 6)]
    searches, items = ['BFS', 'DFS'], ['Trained_', 'Time_']
    TN_df = pd.DataFrame()
    for search in searches:
        fname = 'Pricing Efficiency Test'+' ('+search+').csv'
        df = pd.read_csv(fname, index_col = 0)
        TN_df[['Trained_'+search, 'Time_'+search]] = df[['Trained_Models', 'Training_Time']]
    TN_df.index = xlabel_inputs
        
    for i, item in enumerate(items):
        columns = [item+searches[0], item+searches[1]]
        img = TN_df[columns].plot(kind = 'line', color = color, style = 'o-', fontsize = fontsize-2)
        if i == 0:
            img.set_title('Training Efficiency', fontsize = fontsize)
            img.set_ylabel('Number of Trained Models', fontsize = fontsize)
        else:
            img.set_title('The Training Time', fontsize = fontsize)
            img.set_ylabel('Runtime (s)', fontsize = fontsize)
        img.set_xlabel('Output Models', fontsize = fontsize)
        img.legend(labels = searches, fontsize = fontsize-2)
        plt.show()
        if i == 0:
            img.figure.savefig('ModelTraining_Efficiency.jpg', bbox_inches = 'tight', dpi = 300)
        else:
            img.figure.savefig('ModelTraining_Runtime.jpg', bbox_inches = 'tight', dpi = 300)
        img.remove()

def RDplot(search, avg = True): # 分配效果效率繪圖
    color = [colors[i] for i in (0, 2, 3)]
    strategy = [strategies[i] for i in (0, 3, 4)]
    labels = ['Avg. Distribution Error (%)'] if avg else ['Distribution Error (%)']
    labels.extend(['R2 Score', 'Runtime_P2 (s)'])
    fname, search = 'Optimizing Effectiveness Test', ' ('+search+')'
    df = pd.read_csv(fname+search+'.csv', index_col = 0).loc[labels,:]
    for i, item in enumerate(labels):
        vc = np.array(df.loc[item,:])
        mt = np.reshape(vc, (-1, len(vc)//len(xlabel_inputs)))
        RD_df = pd.DataFrame(mt, index = xlabel_inputs, columns = strategy)
        if i != 2:
            img = RD_df.plot(kind = 'bar', color = color, fontsize = fontsize-2)
            img.set_title('Distribution Effectiveness'+search, fontsize = fontsize)
        else:
            img = RD_df.plot(kind = 'line', color = color, style = 'o-', logy = True, fontsize = fontsize-2)
            img.set_title('Distribution Efficiency'+search, fontsize = fontsize)
        img.set_xlabel('Input Models', fontsize = fontsize)
        if i == 0:
            img.set_ylabel('Distribution Error (%)', fontsize = fontsize)
            img.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 1))
        elif i == 1:
            img.set_ylabel('R^2 score', fontsize = fontsize)
            img.set_ylim(0.8, 1)
            img.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        elif i == 2:
            img.set_ylabel('Runtime (s)', fontsize = fontsize)
            img.set_ylim(0.01, 150)
        img.legend(labels = strategy, fontsize = fontsize-2)
        plt.setp(img.get_xticklabels(), rotation = 0)
        plt.show()
        if i == 0:
            img.figure.savefig('Distribution_Effectiveness'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
            RD_df.to_csv('Distribution_Effectiveness'+search+'.csv', index = True, header = True)
        elif i == 1:
            img.figure.savefig('Distribution_Effectiveness_R2'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
            RD_df.to_csv('Distribution_Effectiveness_R2'+search+'.csv', index = True, header = True)
        elif i == 2:
            img.figure.savefig('Distribution_Efficiency'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

def FIplot(search): # 不等式過濾效果繪圖
    color = [colors[i] for i in (0, 1)]
    fname, search = 'Filtered Efficiency Test', ' ('+search+')'
    methods = ['CIP', 'SLSQP', 'CIP (Unfiltered)', 'SLSQP (Unfiltered)']
    df = pd.read_csv(fname+search+'.csv', index_col = 0)
    df.index = xlabel_inputs
    y_left = list(df.columns[np.arange(1, 4, 2)])
    y_left1 = list(df.columns[np.arange(0, 3, 2)])
    img = df.plot(kind = 'line', y = y_left, color = color, style = 'o-', logy = True, fontsize = fontsize-2)
    img1 = df.plot(kind = 'line', y = y_left1, color = color, style = 'o--', logy = True, legend = False, ax = img, fontsize = fontsize-2)
    img.set_title('Filtered Pricing Efficiency'+search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.set_ylabel('Runtime (s)', fontsize = fontsize)
    img.set_ylim(0.001, 150)
    img.legend(labels = methods, fontsize = fontsize-2)
    plt.setp(img.get_xticklabels(), rotation = 0)
    plt.tight_layout()
    plt.show()
    img.figure.savefig('Filtered_Pricing_Efficiency'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

def FIplot2(search): # 不等式過濾量繪圖
    color = [colors[i] for i in (0, 1)]
    fname, search = 'Removed Inequalities Distribution Test', ' ('+search+')'
    methods = ['CIP', 'SLSQP', 'CIP (Unfiltered)', 'SLSQP (Unfiltered)']
    df = pd.read_csv(fname+search+'.csv', index_col = 0)
    base = list(df.columns[np.arange(0, 16, 3)])
    mt = df[base].T.to_numpy()
    FI_mt = np.zeros((len(xlabel_inputs), 4), float)
    for k in range(0, len(xlabel_inputs)):
        FI_mt[k, 0] = sum(mt[k, 1:3])
        FI_mt[k, 1] = sum(mt[k, 3:5])
        FI_mt[k, 2] = sum(mt[k, 0:3])
        FI_mt[k, 3] = mt[k, 0]+sum(mt[k, 3:5])
    ID_df = pd.DataFrame(FI_mt, index = xlabel_inputs)
    y_left = list(ID_df.columns[np.arange(1, 4, step = 2)])
    y_left1 = list(ID_df.columns[np.arange(0, 3, step = 2)])
    img = ID_df.plot(kind = 'line', y = y_left, color = color, style = 'o-', fontsize = fontsize-2)
    img1 = ID_df.plot(kind = 'line', y = y_left1, color = color, style = 'o--', legend = False, ax = img, fontsize = fontsize-2)
    img.set_title('Filtered Inequalities'+search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.set_ylabel('Number of Remaining Inequalities', fontsize = fontsize)
    img.legend(labels = methods, fontsize = fontsize-2)
    plt.setp(img.get_xticklabels(), rotation = 0)
    plt.tight_layout()
    plt.show()
    img.figure.savefig('Filtered_Inequalities'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

def DBplot(search): # 資料集參與分布繪圖
    color = [colors[i] for i in (0, 2, 3)]
    fname, search = 'Participation Test', ' ('+search+')'
    df = pd.read_csv(fname+search+'.csv', index_col = 0)
    df = df.iloc[:-1,:]
    for i, k in enumerate(xlabel_inputs):
        strategy = [strategies[i]+' (K='+k+')' for i in (0, 3, 4)]
        y_left = list(df.columns[np.arange(i*3, (i+1)*3)])
        img = df.plot(kind = 'line', y = y_left, color = color, style = 'o-', fontsize = fontsize-2)
        img.set_title('Dataset Participation Distribution'+search, fontsize = fontsize)
        img.set_xlabel('Dataset Index', fontsize = fontsize)
        img.set_xticks(np.arange(len(df)))
        img.set_xticklabels(np.arange(10)+1)
        img.set_ylabel('Times of Participation', fontsize = fontsize)
        img.legend(labels = strategy, fontsize = fontsize-2)
        img.set_ylim(0, 140)
        plt.show()
        img.figure.savefig('Participation_K='+k+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

def CRplot(): # 交叉分布繪圖(準確度,資料集大小,資料集參與)
    color = [colors[i] for i in (5, 6)]
    fnames = ['Accuracy Distribution Test', 'Size Distribution Test', 'Participation Test']
    for f, fname in enumerate(fnames):
        BFS_df = pd.read_csv(fname+' (BFS).csv', index_col = 0)[:-1] # 排除'CorrCoef'索引
        DFS_df = pd.read_csv(fname+' (DFS).csv', index_col = 0)[:-1]
        n = len(BFS_df.columns)
        col = BFS_df.columns[np.arange(0, n, n//len(xlabel_inputs))]
        df = pd.concat([BFS_df[col], DFS_df[col]], axis = 1, join = 'inner')
        half = len(df.columns)//2
        for left, k in enumerate(xlabel_inputs):
            right = half+left
            CR_df = df.iloc[:, [left, right]]
            CR_df.columns = ['BFS (K='+k+')', 'DFS (K='+k+')']
            img = CR_df.plot(kind = 'line', color = color, style = 'o-', fontsize = fontsize-2)
            if f == 0:
                img.set_title('Accuracy Distribution', fontsize = fontsize)
                img.set_xlabel('Prediction Accuracy', fontsize = fontsize)
                img.set_ylabel('Number of Models', fontsize = fontsize)
            elif f == 1:
                img.set_title('Training Set Size Distribution', fontsize = fontsize)
                img.set_xlabel('Number of Datasets for Each Model', fontsize = fontsize)
                img.set_xticks(np.arange(len(CR_df)))
                img.set_xticklabels(np.arange(1, 11))
                img.set_ylabel('Number of Models', fontsize = fontsize)
            else:
                img.set_title('Dataset Participation Distribution', fontsize = fontsize)
                img.set_xlabel('Dataset Index', fontsize = fontsize-2)
                img.set_xticks(np.arange(len(CR_df)))
                img.set_xticklabels(np.arange(1, 11))
                img.set_ylabel('Number of Models Participated', fontsize = fontsize)
                # img.set_ylim(0, 150)
            plt.setp(img.get_xticklabels(), rotation = 0)
            plt.tight_layout()
            plt.show()
            if f == 0:
                img.figure.savefig('Accuracy_Distribution_K='+k+'.jpg', bbox_inches = 'tight', dpi = 300)
            elif f == 1:
                img.figure.savefig('Size_Distribution_K='+k+'.jpg', bbox_inches = 'tight', dpi = 300)
            else:
                img.figure.savefig('Participation_K='+k+'.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

def IsExisted():
    searches, path_dir = [], os.getcwd().replace('\\', '/')+'/'+dir_name+'/'+day
    if os.path.exists(path_dir+'/'+batch+' (BFS)'):
        searches.append('BFS')
    if os.path.exists(path_dir+'/'+batch+' (DFS)'):
        searches.append('DFS')
    if searches:
        os.chdir(path_dir+'/'+batch)
    return searches

# In[main]:

if __name__ == "__main__":
    searches = IsExisted()
    if searches:
        if 'BFS' in searches and 'DFS' in searches:
            TNplot() # 訓練效率繪圖
            CRplot() # 交叉分布繪圖(準確度,資料集大小,資料集參與)
        for search in searches:
            PCplot(search) # 定價方法效果效率繪圖
            RMplot(search) # 移除策略效果繪圖
            PTplot(search) # 各階段耗時繪圖
            RDplot(search) # 分配效果效率繪圖
            FIplot(search) # 不等式過濾效果繪圖
            FIplot2(search) # 不等式過濾量繪圖
            DBplot(search) # 資料集參與分布繪圖

