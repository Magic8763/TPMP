
import os
os.chdir('D:/ANDY/Documents/大學/研究所/論文/模型解釋/Shapley/Data Shapley/DataShapley-master')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# color = ['r', 'g', 'b', 'c', 'k', 'grey', 'DarkOrange', 'DarkGreen', 'DarkBlue', 'purple', 'pink']

# In[定價結果繪圖]:

def PCplot2(fname, batch = '', search = '', fontsize = 12):
    title = ['CIP', 'SLSQP']
    PC_df = pd.read_csv(fname+batch+search+'.csv', index_col = 0)
    idxlist = list(PC_df.index)
    idxlist[-1] = 'MAX'
    PC_df.index = idxlist
    y_left = list(PC_df.columns[np.arange(3, 7, step = 3)])
    y_left2 = list(PC_df.columns[np.arange(2, 6, step = 3)])
    img = PC_df.plot(kind = 'line', y = y_left, color = ['C0', 'purple'], style = '.-', logy = True, marker = 'o', fontsize = fontsize-2)
    img.set_title('Pricing Efficiency' + search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.set_ylabel('Runtime (s)', fontsize = fontsize)
    img.legend(labels = title, fontsize = fontsize-2)

    img2 = PC_df.plot(kind = 'bar', y = y_left2, color = ['C0', 'purple'], fontsize = fontsize-2)
    img2.set_title('Pricing Effectiveness' + search, fontsize = fontsize)
    img2.set_xlabel('Input Models', fontsize = fontsize)
    img2.set_ylabel('Revenue Ratio (%)', fontsize = fontsize)
    img2.legend(labels = title, fontsize = fontsize-2)
    img2.set_ylim(37, 103)
    plt.setp(img2.get_xticklabels(), rotation = 0)

    plt.show()
    img.figure.savefig('Pricing_Efficiency_Curve' + batch + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img2.figure.savefig('Pricing_Effectiveness_Curve' + batch + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()
    img2.remove()

batch = '[pNormal disp=0.0]'
# batch = '[nNormal disp=0.0]'
search = ' (BFS)'
# search = ' (DFS)'
fname = 'Pricing Efficiency Test '
PCplot2(fname, batch = batch, search = search, fontsize = 14)

# In[移除結果繪圖]:

def RMplot2(fname, batch = '', search = '', fontsize = 12):
    ColorMap = list()
    for i in range(0, 10):
        ColorMap += ['C'+str(i)] 
    ColorMap_sub = list()
    for i in [0, 2, 3]:
        ColorMap_sub += [ColorMap[i]]
    strategy = ['Base', 'MinCover', 'MaxImprove']
    k_list = np.arange(50, 251, step = 50)
    idxlist = list(k_list) + ['MAX']
    df = pd.read_csv(fname+batch+search+'.csv', index_col = 0)
    title = list(df.columns)
    col_size = len(strategy)
    row_size = len(idxlist)
    ICperM = np.zeros(col_size, float)
    RM_mt = np.zeros((0, col_size), float)
    for i in range(0, row_size):
        RM_mt = np.concatenate([RM_mt, [df[title[col_size*i:col_size*(i+1)]].iloc[0]]])
    for i in range(0, row_size):
        RM_mt = np.concatenate([RM_mt, [df[title[col_size*i:col_size*(i+1)]].iloc[1]]])
    for i in range(0, row_size):
        RM_mt = np.concatenate([RM_mt, [df[title[col_size*i:col_size*(i+1)]].iloc[2]]])
    for i in range(0, row_size):
        temp = df[title[col_size*i:col_size*(i+1)]].iloc[3]
        temp = temp[0] - temp
        RM_mt = np.concatenate([RM_mt, [temp]])
    for i in range(0, row_size):
        ICperM[1:] = (RM_mt[row_size+i, 1:] - RM_mt[row_size+i, 0])/RM_mt[row_size*3+i,1:]
        RM_mt = np.concatenate([RM_mt, [ICperM]])
    for i in range(0, row_size):
        RM_mt = np.concatenate([RM_mt, [df[title[col_size*i:col_size*(i+1)]].iloc[-1]]])
    RM_df = pd.DataFrame(RM_mt[:row_size,:], index = idxlist, columns = strategy)
    img = RM_df.plot(kind = 'bar', color = ColorMap_sub, fontsize = fontsize-2)
    img.set_title(batch + '\nOptimizing Effectiveness' + search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.set_ylabel('Revenue', fontsize = fontsize)
    img.legend(loc = 'best', fontsize = fontsize-2)
    plt.setp(img.get_xticklabels(), rotation = 0)

    RM_df1 = pd.DataFrame(RM_mt[row_size:row_size*2,:], index = idxlist, columns = strategy)
    img1 = RM_df1.plot(kind = 'bar', color = ColorMap_sub, fontsize = fontsize-2)
    img1.set_title('Optimizing Effectiveness' + search, fontsize = fontsize)
    img1.set_xlabel('Input Models', fontsize = fontsize)
    img1.set_ylabel('Revenue Ratio (%)', fontsize = fontsize)
    img1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 1))
    img1.legend(loc = 'upper left', bbox_to_anchor = (0.6, 0.62), fontsize = fontsize-2)
    img1.set_ylim(37, 103)
    plt.setp(img1.get_xticklabels(), rotation = 0)

    RM_df2 = pd.DataFrame(RM_mt[row_size*2:row_size*3,:], index = idxlist, columns = strategy)
    img2 = RM_df2.plot(kind = 'line', style = '.-', color = ColorMap_sub, marker = 'o', fontsize = fontsize-2)
    img2.set_title('Optimizing Efficiency' + search, fontsize = fontsize)
    img2.set_xlabel('Input Models', fontsize = fontsize)
    img2.set_ylabel('Runtime (s)', fontsize = fontsize)
    img2.legend(loc = 'best', fontsize = fontsize-2)
    plt.setp(img2.get_xticklabels(), rotation = 0)

    RM_df3 = pd.DataFrame(RM_mt[row_size*3:row_size*4,:], index = idxlist, columns = strategy)
    img3 = RM_df3.plot(kind = 'bar', y = strategy[1:], color = ColorMap_sub[1:], fontsize = fontsize-2)
    img3.set_title('Removed Models' + search, fontsize = fontsize)
    img3.set_xlabel('Input Models', fontsize = fontsize)
    img3.set_ylabel('Number of Models', fontsize = fontsize)
    img3.legend(loc = 'best', fontsize = fontsize-2)
    plt.setp(img3.get_xticklabels(), rotation = 0)

    RM_df4 = pd.DataFrame(RM_mt[row_size*4:row_size*5,:], index = idxlist, columns = strategy)
    img4 = RM_df4.plot(kind = 'bar', y = strategy[1:], color = ColorMap_sub[1:], fontsize = fontsize-2)
    img4.set_title(batch + '\nUnit Improvement' + search, fontsize = fontsize)
    img4.set_xlabel('Input Models', fontsize = fontsize)
    img4.set_ylabel('Revenue Ratio (%)', fontsize = fontsize)
    img4.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 1))
    img4.legend(loc = 'best', fontsize = fontsize-2)
    plt.setp(img4.get_xticklabels(), rotation = 0)

    plt.show()
    img.figure.savefig('Optimizing_Effectiveness_Plot' + batch + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img1.figure.savefig('Optimizing_Effectiveness_Rr_Plot' + batch + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img2.figure.savefig('Optimizing_Efficiency_Plot' + batch + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img3.figure.savefig('Optimizing_Removed_Plot' + batch + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img4.figure.savefig('Optimizing_UnitImprovement_Plot' + batch + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()
    img1.remove()
    img2.remove()
    img3.remove()
    img4.remove()

batch = '[pNormal disp=0.0]'
# batch = '[nNormal disp=0.0]'
search = ' (BFS)'
# search = ' (DFS)'
fname = 'Optimizing Effectiveness Test '
RMplot2(fname, batch = batch, search = search, fontsize = 14)

# In[階段時間繪圖]:

def PTplot(fname = [], search = [], fontsize = 12):
    k_list = np.arange(50, 251, step = 50)
    idxlist = list(k_list) + ['MAX']
    labels = ['Training', 'Pricing', 'Distribution', 'Total']
    methods = ['CIP', 'SLSQP']
    for s in range(0, len(search)):
        PT_mt = np.zeros((len(idxlist), len(labels)*2), float)
        for f in range(0, len(fname)):
            if f == 0:
                df = pd.read_csv(fname[f]+search[s]+'.csv', index_col = 0)
                title = list(df.columns)
                PT_mt[:, 0] = df[title[1]]
                PT_mt[:, 1] = df[title[3]]
                PT_mt[:, len(labels)] = df[title[1]]
                PT_mt[:, len(labels)+1] = df[title[6]]
            elif f == 1:
                df = pd.read_csv(fname[f]+search[s]+'.csv', index_col = 0).T
                title = list(df.columns)
                loc1 = np.arange(0, df.shape[0], step = 3)
                PT_mt[:, 2] = df[title[5]].iloc[loc1]
                PT_mt[:, len(labels)+2] = df[title[5]].iloc[loc1]
        PT_mt[:, len(labels)-1] = np.sum(PT_mt[:, 0:len(labels)], axis = 1)
        PT_mt[:, len(labels)*2-1] = np.sum(PT_mt[:, len(labels):len(labels)*2], axis = 1)
        for m in range(0, len(methods)):
            PT_df = pd.DataFrame(PT_mt[:, len(labels)*m:len(labels)*(m+1)], index = idxlist, columns = labels)
            img = PT_df.plot(kind = 'line', style = '.-', color = ['C0', 'C2', 'C3', 'black'], logy = True, marker = 'o', fontsize = fontsize-2)
            img.set_title('The Runtime of Each Phase: ' + methods[m] + search[s], fontsize = fontsize)
            img.set_xlabel('Input Models', fontsize = fontsize)
            img.set_ylabel('Runtime (s)', fontsize = fontsize)
            img.set_ylim(0.005, 10000)
            img.legend(loc = 'best', labels = labels, fontsize = fontsize-2)
            plt.tight_layout()
            plt.show()
            img.figure.savefig('Phase_Runtime_Curve_' + methods[m] + search[s] + '.jpg', bbox_inches = 'tight', dpi = 300)
            img.remove()

search = [' (BFS)',' (DFS)']
fname = ['Pricing Efficiency Test [pNormal disp=0.0]', 'Optimizing Effectiveness Test [pNormal disp=0.0]']
PTplot(fname, search = search, fontsize = 14)

# In[迭代次數繪圖]:

def ITplot(fname, search = '', method = [0], fontsize = 12):
    title = ['BFS', 'DFS']
    k_list = np.arange(50, 251, step = 50)
    idxlist = list(k_list) + ['MAX']
    IT_df = pd.DataFrame()
    for f in range(0, len(search)):
        df = pd.read_csv(fname+search[f]+'.csv', index_col = 0).T
        if f == 0:
            item = df.columns[4]
        for m_idx in method:
            m_list = np.arange(0 + m_idx, df.shape[0], step = 3)
            if len(method) > 1:
                IT_df[title[f]+str(m_idx)] = df[item].values[m_list]-1
            else:
                IT_df[title[f]] = df[item].values[m_list]-1
    IT_df.index = idxlist
    img = IT_df.plot(kind = 'line', color = ['DarkOrange', 'c'], style = '.-', marker = 'o', fontsize = fontsize-2)
    img.set_title('Avg. Iterations of CIP', fontsize = fontsize)
    img.set_ylabel('Iterations', fontsize = fontsize)
    img.set_xlabel('Output Models', fontsize = fontsize)
    img.legend(labels = title, fontsize = fontsize-2)
    plt.show()
    img.figure.savefig('CIP_Iterations_Curve.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

search = [' (BFS)', ' (DFS)']
fname = 'Optimizing Effectiveness Test [pNormal disp=0.0]'
a = ITplot(fname, search = search, method = [0], fontsize = 14)

# In[訓練結果繪圖]:

def TNplot(fname, search = '', fontsize = 12):
    methods = ['BFS', 'DFS']
    for col in range(0, 2):
        for f in range(0, len(search)):
            df = pd.read_csv(fname+search[f]+'.csv', index_col = 0)
            title = df.columns[col]
            if f == 0:
                k_list = np.arange(50, 251, step = 50)
                idxlist = list(k_list) + ['MAX']
                TN_df = pd.DataFrame()
            TN_df[methods[f]] = df[title]
        TN_df.index = idxlist
        if col == 0:
            img = TN_df.plot(kind = 'line', color = ['DarkOrange', 'c'], style = '.-', marker = 'o', fontsize = fontsize-2)
            img.set_title('Training Efficiency', fontsize = fontsize)
            img.set_ylabel('Number of Trained Models', fontsize = fontsize)
        else:
            img = TN_df.plot(kind = 'line', color = ['DarkOrange', 'c'], style = '.-', marker = 'o', fontsize = fontsize-2)
            img.set_title('The Training Time', fontsize = fontsize)
            img.set_ylabel('Runtime (s)', fontsize = fontsize)
        img.set_xlabel('Output Models', fontsize = fontsize)
        img.legend(labels = methods, fontsize = fontsize-2)
        plt.show()
        if col == 0:
            img.figure.savefig('Training_Efficiency_Curve.jpg', bbox_inches = 'tight', dpi = 300)
        else:
            img.figure.savefig('Training_Time_Curve.jpg', bbox_inches = 'tight', dpi = 300)
        img.remove()

search = [' (BFS)', ' (DFS)']
fname = 'Pricing Efficiency Test [pNormal disp=0.0]'
TNplot(fname, search = search, fontsize = 14)

# In[分配效果繪圖]:

def CRplot(fname, batch = '', search = '', item = 0, fontsize = 12):
    k_list = np.arange(50, 251, step = 50)
    idxlist = list(k_list) + ['MAX']
    score = np.append(5, [7, 8])
    ColorMap = list()
    for i in range(0, 10):
        ColorMap += ['C'+str(i)] 
    ColorMap_sub = list()
    for i in [0, 3]:
        ColorMap_sub += [ColorMap[i]]
    strategy = ['Base', 'MinCover', 'MaxImprove']
    df = pd.read_csv(fname+batch+search+'.csv', index_col = 0).iloc[score[item],:]
    vc = np.array(df)
    mt = np.reshape(vc, (-1, len(vc)//len(idxlist)))
    CR_df = pd.DataFrame(mt, index = idxlist, columns = strategy)
    methods = ['Base', 'MaxImprove']
    img = CR_df.plot(kind = 'line', y = methods, style = '.-', color = ColorMap_sub, marker = 'o', fontsize = fontsize-2)
    img.set_title(batch + '\nDistribution Effectiveness' + search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    if item == 0:
        img.set_ylabel('Distribution Error (%)', fontsize = fontsize)
        img.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = 1))
    elif item == 1:
        img.set_ylabel('R^2 score', fontsize = fontsize)
    img.legend(labels = methods, fontsize = fontsize-2)
    plt.setp(img.get_xticklabels(), rotation = 0)
    plt.show()
    img.figure.savefig('Distribution_Effectiveness_Curve' + batch + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

    fname = 'Distribution_Effectiveness_Table' + batch + search + '.csv'
    CR_df.to_csv(fname, index = True, header = True)
    return CR_df

batch = '[pNormal disp=0.0]'
# batch = '[nNormal disp=0.0]'
search = ' (BFS)'
# search = ' (DFS)'
fname = 'Optimizing Effectiveness Test '
CR_df = CRplot(fname, batch = batch, search = search, item = 1, fontsize = 14)

# In[分配效率繪圖]:

def CRplot2(fname, search = '', fontsize = 12):
    methods = ['Base', 'MaxImprove']
    k_list = np.arange(50, 251, step = 50)
    idxlist = list(k_list) + ['MAX']
    loc = [np.arange(0, len(idxlist)*3, step = 3)]
    loc += [np.arange(2, len(idxlist)*3, step = 3)]
    CR_df = pd.DataFrame(index = idxlist)
    for m in range(0, len(methods)):
        for f in range(0, len(search)):
            df = pd.read_csv(fname+search[f]+'.csv', index_col = 0).T
            CR_df[methods[m]+search[f]] = df.iloc[loc[m], 6].values
    img = CR_df.plot(kind = 'line', style = '.-', logy = True, marker = 'o', fontsize = fontsize-2)
    img.set_title('Distribution Efficiency', fontsize = fontsize)
    img.set_ylabel('Runtime (s)', fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.legend(fontsize = fontsize-2)
    plt.show()
    img.figure.savefig('Distribution_Efficiency_Curve.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

search = [' (BFS)', ' (DFS)']
fname = 'Optimizing Effectiveness Test [pNormal disp=0.0]'
CRplot2(fname, search = search, fontsize = 14)

# In[過濾結果繪圖]:

def FRplot(fname, search = '', fontsize = 12):
    methods = ['CIP', 'SLSQP', 'CIP (Filtered)', 'SLSQP (Filtered)']
    FR_df = pd.read_csv(fname, index_col = 0)
    idxlist = list(FR_df.index)
    idxlist[-1] = 'MAX'
    FR_df.index = idxlist
    y_left = list(FR_df.columns[np.arange(0, 3, step = 2)])
    y_left1 = list(FR_df.columns[np.arange(1, 4, step = 2)])
    img = FR_df.plot(kind = 'line', y = y_left, color = ['C0', 'purple'], style = '.-', logy = True, marker = 'o', fontsize = fontsize-2)
    img1 = FR_df.plot(kind = 'line', y = y_left1, color = ['C0', 'purple'], style = '.--', logy = True, legend = False, ax = img, marker = 'o', fontsize = fontsize-2)
    img.set_title('Filtered Efficiency' + search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.set_ylabel('Runtime (s)', fontsize = fontsize)
    img.legend(labels = methods, fontsize = fontsize-2)
    plt.setp(img.get_xticklabels(), rotation = 0)
    plt.tight_layout()
    plt.show()
    img.figure.savefig('Filtered_Efficiency_Curve' + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()
    img1.remove()

search = ' (BFS)'
# search = ' (DFS)'
fname = 'Filtered Efficiency Test' + search + '.csv'
FRplot(fname, search = search, fontsize = 14)

# In[不等式過濾繪圖]:

def IDplot2(fname, search = '', fontsize = 12):
    methods = ['CIP', 'SLSQP', 'CIP (Filtered)', 'SLSQP (Filtered)']
    df = pd.read_csv(fname, index_col = 0)
    base = list(df.columns[np.arange(0, 16, step = 3)])
    mt = df[base].T.to_numpy()
    k_list = np.arange(50, 251, step = 50)
    k_list = np.append(k_list, 1)
    ID_mt = np.zeros((len(k_list), 4), float)
    for k in range(0, len(k_list)):
        ID_mt[k, 0] = sum(mt[k, 1:3])
        ID_mt[k, 1] = sum(mt[k, 3:5])
        ID_mt[k, 2] = sum(mt[k, 0:3])
        ID_mt[k, 3] = mt[k, 0] + sum(mt[k, 3:5])
    title = ['CIP_Unfiltered', 'CIP_Filtered', 'SLSQP_Unfiltered', 'SLSQP_Filtered']
    idxlist = list(k_list[:-1]) + ['MAX']
    ID_df = pd.DataFrame(ID_mt, index = idxlist, columns = title)
    y_left = list(ID_df.columns[np.arange(0, 3, step = 2)])
    y_left1 = list(ID_df.columns[np.arange(1, 4, step = 2)])
    img = ID_df.plot(kind = 'line', y = y_left, color = ['C0', 'purple'], style = '.-', marker = 'o', fontsize = fontsize-2)
    img1 = ID_df.plot(kind = 'line', y = y_left1, color = ['C0', 'purple'], style = '.--', legend = False, ax = img, marker = 'o', fontsize = fontsize-2)
    img.set_title('Filtered Effectiveness' + search, fontsize = fontsize)
    img.set_xlabel('Input Models', fontsize = fontsize)
    img.set_ylabel('Number of Remaining Inequalities', fontsize = fontsize)
    img.legend(labels = methods, fontsize = fontsize-2)
    plt.setp(img.get_xticklabels(), rotation = 0)
    # img.set_xticks(list(ID_df.index))

    plt.tight_layout()
    plt.show()
    img.figure.savefig('Filtered_Effectiveness_Curve' + search + '.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()
    img1.remove()
    return ID_df

search = ' (BFS)'
# search = ' (DFS)'
fname = 'Removed Inequalities Distribution Test' + search + '.csv'
ID_df = IDplot2(fname, search = search, fontsize = 14)

# In[模型分布繪圖]:

def DBplot(fname = [], search = [], strategy = [0], fontsize = 12):
    k_list = np.arange(50, 251, step = 50)
    k_list = np.append(k_list, 1)
    labels = list()
    for f in range(0, len(fname)):
        for s in range(0, len(search)):
            if f == 0:
                for k in k_list:
                    if k == 1:
                        k_str = 'MAX'
                    else:
                        k_str = str(k)
                    if s == 0:
                        labels += ['BFS (K='+k_str+')']
                    else:
                        labels += ['DFS (K='+k_str+')']
            df = pd.read_csv(fname[f]+search[s]+'.csv', index_col = 0)
            if s == 0:
                item = list(df.index)[:-1]
                DB_df = pd.DataFrame(index = item)
            title = list(df.columns)
            name = list()
            for t in title:
                name += [t+search[s]]
                # name = list(df.columns)[:-1]
            DB_df[name] = df[title].iloc[:-1]
        idx_range = np.where(DB_df[name[-1]] > 0)
        DB_df = DB_df.iloc[idx_range]
        half = DB_df.shape[1]//2
        y_left = list(DB_df.columns[np.arange(strategy[0], DB_df.shape[1], step = half)])
        img = DB_df.plot(kind = 'line', y = y_left, color = ['DarkOrange', 'c', 'DarkBlue'], style = '.-', legend = False, marker = 'o', fontsize = fontsize-2)
        # img1 = DB_df.plot(kind = 'line', y = y_left1, color = ['DarkOrange', 'c'], style = '--', legend = False, ax = img, marker = 'o', fontsize = fontsize-2)
        if f == 0:
            img.set_title('Accuracy Distribution', fontsize = fontsize)
            img.set_xlabel('Prediction Accuracy', fontsize = fontsize)
            img.set_ylabel('Number of Models', fontsize = fontsize)
        elif f == 1:
            img.set_title('Training Set Size Distribution', fontsize = fontsize)
            img.set_xlabel('Number of Datasets for Each Model', fontsize = fontsize)
            img.set_xticks(np.arange(len(DB_df)))
            img.set_xticklabels(np.arange(10)+1)
            img.set_ylabel('Number of Models', fontsize = fontsize)
        else:
            img.set_title('Dataset Participation Distribution', fontsize = fontsize-2)
            img.set_xlabel('Dataset Index', fontsize = fontsize-2)
            img.set_xticks(np.arange(len(DB_df)))
            img.set_xticklabels(np.arange(10)+1)
            img.set_ylabel('Number of Models Participated', fontsize = fontsize-2)
        label = list()
        for i in np.arange(strategy[0]//3, len(labels), step = 6):
            label += [labels[i]]
        img.legend(labels = label, fontsize = fontsize-2)
        plt.setp(img.get_xticklabels(), rotation = 0)

        plt.tight_layout()
        plt.show()
        if f == 0:
            img.figure.savefig('Accuracy_Distribution_Curve_K='+str(k_list[strategy[0]//3])+'.jpg', bbox_inches = 'tight', dpi = 300)
        elif f == 1:
            img.figure.savefig('Size_Distribution_Curve_K='+str(k_list[strategy[0]//3])+'.jpg', bbox_inches = 'tight', dpi = 300)
        else:
            img.figure.savefig('Participation_Curve_K='+str(k_list[strategy[0]//3])+'.jpg', bbox_inches = 'tight', dpi = 300)
        img.remove()

search = [' (BFS)',' (DFS)']
fname = ['Accuracy Distribution Test', 'Size Distribution Test', 'Participation Test']
DBplot(fname, search = search, strategy = [9], fontsize = 14)

# In[模型分布繪圖]:

def DBplot2(fname, search = '', k = 0, fontsize = 12):
    k_list = np.arange(50, 251, step = 50)
    strategies = ['Base (K='+str(k_list[k])+')', 'MinCover (K='+str(k_list[k])+')', 'MaxImprove (K='+str(k_list[k])+')']
    DB_df = pd.read_csv(fname+search+'.csv', index_col = 0)
    title = list(DB_df.columns)
    idx_range = np.where(DB_df[title[-1]] > 0)
    DB_df = DB_df.iloc[idx_range][:-1]
    y_left = list(DB_df.columns[np.arange(k*3, (k+1)*3)])
    img = DB_df.plot(kind = 'line', y = y_left, color = ['C0', 'C2', 'C3'], style = '.-')
    img.set_title('Dataset Participation Distribution'+search, fontsize = fontsize-2)
    img.set_xlabel('Dataset Index')
    img.set_xticks(np.arange(len(DB_df)))
    img.set_xticklabels(np.arange(10)+1)
    img.set_ylabel('Times of Participation')
    img.legend(labels = strategies, fontsize = fontsize-2)

    plt.show()
    img.figure.savefig('Participation_Curve'+search+'.jpg', bbox_inches = 'tight', dpi = 300)
    img.remove()

# search = ' (BFS)'
search = ' (DFS)'
fname = 'Participation Test'
# fname = 'Accuracy Distribution Test'
DBplot2(fname, search = search, k = 0, fontsize = 14)

# In[定價舉例]:

df = pd.DataFrame()
df['v(.)'] = [10, 20, 30, 40, 50]
df.index = [0.6, 0.64, 0.68, 0.72, 0.76]
df['mp(.)'] = [10, 15, 15, 20, 30]
ineq = False
itr = False

# In[定價舉例]:

df = pd.DataFrame(index = [0.6, 0.62, 0.64, 0.68, 0.72, 0.76, 0.8])
df['v(.)'] = [10, 15, 20, 30, 40, 50, 60]
df['q1'] = [10, 10, 10, 0, 0, 0, 0]
df['q2'] = [0, 15, 20, 0, 35, 35, 0]
df['q3'] = [0, 0, 0, 0, 40, 40, 40]
ineq = True
itr = False

# In[定價舉例]:

df = pd.DataFrame(index = [0.6, 0.62, 0.64, 0.72, 0.76, 0.8])
df['v(.)'] = [10, 15, 20, 40, 50, 60]
df['Round1'] = [10, 10, 10, 40, 50, 60]
df['Round2'] = [10, 10, 10, 20, 20, 60]
df['Round3'] = [10, 10, 10, 20, 20, 20]
ineq = True
itr = True

# In[定價舉例]:

df = pd.DataFrame(index = [0.6, 0.62, 0.64, 0.68, 0.72, 0.76, 0.8])
df['v(.)'] = [10, 15, 20, 30, 40, 50, 60]
df['q1'] = [10, 10, 10, 0, 0, 0, 0]
df['q3'] = [0, 0, 0, 30, 30, 30, 30]
df['q2'] = [0, 15, 20, 0, 35, 35, 0]
ineq = True
itr = False

# In[定價舉例]:

def PSplot(df, ineq = False, itr = False, fontsize = 12):
    title = list(df.columns)
    acc = list(df.index)
    C = ['C0', 'C1', 'DarkGreen', 'C3']
    for i in range(0, df.shape[1]):
        if itr:
            if ineq:
                name = 'CIP'
                if i == 0:
                    img = df.plot(kind = 'line', y = title[0], color = C[0], style = '.-', marker = 'o', fontsize = fontsize-2)
                    img.set_title('The Initial Price of the Models', fontsize = fontsize)
                    img.legend(fontsize = fontsize-2)
                else:
                    img = df.plot(kind = 'line', y = title[0], color = C[0], style = '--', fontsize = fontsize-2)
                    img1 = df.plot(kind = 'line', y = title[i], color = C[i], ax = img, style = '.-', marker = 'o', fontsize = fontsize-2)
                    img.set_title('Iterative Process of CIP', fontsize = fontsize)
                    img.legend(labels = ['v(.)', 'mp(.)'], fontsize = fontsize-2)
        else:
            if ineq:
                name = 'ICS'
                if i == 0:
                    img = df.plot(kind = 'line', y = title[0], color = C[0], style = '--', fontsize = fontsize-2)
                    img1 = df.replace(0, np.nan).plot(kind = 'line', y = title[1:], color = C[1:], ax = img, style = '-', linewidth = 3.0,  fontsize = fontsize-2)
                    img.set_title('The Initial State of CIP', fontsize = fontsize)
                elif i == 1:
                    img = df.plot(kind = 'line', y = title[0], color = C[0], style = '-', fontsize = fontsize-2)
                    img1 = df.replace(0, np.nan).plot(kind = 'line', y = title[1:], color = C[1:], ax = img, style = '.-', marker = 'o', fontsize = fontsize-2)
                    img.set_title('Inequality Constraints', fontsize = fontsize)
                    for q in range(1, len(title)):
                        for idx in range(0, len(acc)):
                            if df[title[q]].iloc[idx] > 0: 
                                img.vlines(x = acc[idx], ymin = df[title[q]].iloc[idx], ymax = df[title[0]].iloc[idx], color = C[q], linestyle = ':', lw = 2)
                else:
                    break
            else:
                name = 'MPLoss'
                if i == 0:
                    img = df.plot(kind = 'line', y = title, color = C[0:len(title)], style = '.-', marker = 'o', fontsize = fontsize-2)
                    img.set_title('Price Loss', fontsize = fontsize)
                    for idx in range(0, len(acc)):
                        img.vlines(x = acc[idx], ymin = df[title[-1]].iloc[idx], ymax = df[title[0]].iloc[idx], color = 'r', linestyle = ':', lw = 2)
                else:
                    break
            img.legend(fontsize = fontsize-2)
        img.set_xticks(list(df.index))
        img.set_xlabel('Accuracy', fontsize = fontsize)
        img.set_xticks(df.index)
        img.set_ylabel('Price', fontsize = fontsize)
        plt.setp(img.get_xticklabels(), rotation = 45)
        plt.show()
        if df.shape[1] > 1:
            img.figure.savefig(name + '_' + str(i) + '.jpg', bbox_inches = 'tight', dpi = 300)
        else:
            img.figure.savefig(name + '.jpg', bbox_inches = 'tight', dpi = 300)
        img.remove()

PSplot(df, ineq = ineq, itr = itr, fontsize = 14)

