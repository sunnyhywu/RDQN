import numpy as np 
from pandas.core.frame import DataFrame
#import cloudDecision

np.random.seed(3)           # 若要每次結果都不一樣的話，把這行註解掉


# 1. 讀取訂單生產資料(存在orderInfo[]、num_job、num_machine) # 注意機台數一定要一樣

num_orderProductionData = 10 # 共有幾筆訂單生產資料(讀幾個檔案, e.g., dmu01.txt, dumu02.txt, ...) 


class CJob:                 # 一個 job 的「訂單生產資料」
    permut_m = []           # 此list是 machine ID 的排序
    permut_p = []           # 此list是上述的對應 machine ID 的處理時間的排序
    
class CMachine:             # 一個 Machine 的「加工資料」
    permut_j = []           # 此list是 Job製程 的順序
    permut_p = []           # 此list是上述的對應 job 在此 machine 的處理時間

class COrder:               # 訂單的生產資料
    num_job = 0             # num_job[i] = 第i筆資料的job數目
    
    # 訂單生產資料
    allJob = []             # 此list中的每一個元素是CJob
    
    # Machine 加工資料
    allMachine = []         # 此list中的每一個元素是CMachine

    # 訂單特徵資料
    sum_p = 0               # 加工時間總和
    max_p = 0               # 最大加工時間
    min_p = 0               # 最小加工時間

# 系統特徵資料
F_bar = 0               # 平均流程時間
C_max = 0               # 總處理時間
Q_bar = 0               # job平均等候處理時間
mu = []                 # 各機器使用率(m個神經元)
O_bar = []              # 各機器在製品數量(m個神經元) 


# 所有的訂單生產資料，此list中的每一個元素是 COrder
allOrder = [COrder() for i in range(num_orderProductionData)]

# 讀取機器數目
#f = open('testing/01' + str(orderProductionDataInitialID) + '.txt', 'r')

f = open('testing/dmu01.txt', 'r')
row = f.readline().rstrip().split('\t')
num_machine = int(row[1])  # 機器數目不管讀哪個檔都一樣，所以設為全域變數
f.close()

# 用於建立 blue table (表1) 的一個橫列
class blue_row:
    job = []
    beg_time = []
    end_time = []

# ========== 計算特徵資訊的函數 ==========
def computeFeaturedInfo(m_ID, m_solutionInMachine):
    num_job = allOrder[m_ID].num_job
  
#    for i in range(num_orderProductionData):
#        print(allOrder[i].num_job)
#    
    # 填入表一    
    blue = [blue_row() for j in range(num_machine)]
    
    for j in range(num_machine):
        blue[j].job = [int for i in range(num_job)]
#        print(len(blue[j].job))
        for i in range(num_job):
#            print("j",j)
#            print("i",i)
#            print("blue",len(blue[j].job))
##       
#            print(m_solutionInMachine[0])
##          
#            print(m_solutionInMachine)
            blue[j].job[m_solutionInMachine[j][i]] = i

    for j in range(num_machine):
        t = 0
        blue[j].beg_time = [int for i in range(num_job)]
        blue[j].end_time = [int for i in range(num_job)]
        for i in range(num_job):
            blue[j].beg_time[i] = t           
            t = blue[j].end_time[i] = t + allOrder[m_ID].allMachine[j].permut_p[blue[j].job[i]]
    
    # 排程
    for i in range(num_job):
        for j in range(num_machine-1):
            m1 = allOrder[m_ID].allJob[i].permut_m[j]        # m1 = job i 的第j個製程的機台
            m2 = allOrder[m_ID].allJob[i].permut_m[j+1]      # m2 = job i 的第j+1個製程的機台
            j1 = m_solutionInMachine[m1][i]                 # j1 = job i 在 machine m1 的順序
            j2 = m_solutionInMachine[m2][i]                 # j2 = job i 在 machine m2 的順序
            t1 = blue[m1].end_time[j1]                          # t1 = job i 在 machine m1 的結束時間
            t2 = blue[m2].beg_time[j2]                          # t2 = job i 在 machine m2 的開始時間
            gap = t1 - t2
            
            if gap > 0:
                blue[m2].beg_time[j2] += gap
                blue[m2].end_time[j2] += gap
                
                for k in range(j2, num_job-1):
                    gap = blue[m2].end_time[k] - blue[m2].beg_time[k+1]
                    
                    if gap > 0:
                        blue[m2].beg_time[k+1] += gap
                        blue[m2].end_time[k+1] += gap
    
    # 計算各 job 的完工時間
    C = []
    for i in range(num_job):
        m1 = allOrder[m_ID].allJob[i].permut_m[num_machine-1]    # m1 = job i 的最後一個製程的機台
        j1 = m_solutionInMachine[m1][i]                         # j1 = job i 在 machine m1 的順序
        C.append(blue[m1].end_time[j1])

    return sum(C)/num_job, \
    max(C), \
    sum([C[i] - sum_p_of_job[i] for i in range(num_job)])/num_job, \
    [1 - (blue[j].beg_time[0] + sum([blue[j].beg_time[i+1] - blue[j].end_time[i] for i in range(num_job-1)]))/blue[j].end_time[num_job-1] for j in range(num_machine)], \
    [sum(blue[j].end_time)/blue[j].end_time[num_job-1] for j in range(num_machine)]


# ========== 計算edge排程的函數 ==========
def computeSolutionInMachine(jobGroup, target_of_job, isReverse):
    row_solutionInMachine = np.negative(np.ones(num_job))
    isOverlap = False
    
    k = 0
    for group_ID in range(3):
        if len(jobGroup[group_ID]) == 0:
            continue
        
        # 對應到此 group 的所有預測到達時間
        myTarget = [target_of_job[i] for i in jobGroup[group_ID]]

        # 排序
        myTarget, jobGroup[group_ID] = zip(*[(x, y) for x, y in sorted(zip(myTarget, jobGroup[group_ID]), reverse=isReverse)])
        
        #初始化
        row_solutionInMachine[jobGroup[group_ID][0]] = k
        num_overlap = 0
        k += 1
        
        for i in range(1, len(jobGroup[group_ID])):
            if myTarget[i] == myTarget[i-1] :
                num_overlap += 1
                isOverlap = True
            else:
                num_overlap = 0
            
            row_solutionInMachine[jobGroup[group_ID][i]] = k - num_overlap
            k += 1
        
    return row_solutionInMachine, isOverlap



# ========== 主程式 ==========
num_jobList = []
# Step 1: 讀取所有「訂單生產資料」 ==> allOrder[]
for data_ID in range(num_orderProductionData):
    # 檔名是 dmu01.txt ~ dmu{num_orderProductionData}.txt
    f = open('testing/dmu' + str(data_ID+1).zfill(2) + '.txt', 'r')


    row = f.readline().rstrip().split('\t')
    
    num_jobList.append(int(row[0]))
    
    num_job = int(row[0])

    num_machine = int(row[1])
  
    allOrder[data_ID].num_job = num_job

    allOrder[data_ID].allJob = [CJob() for j in range(num_job)]
  
    for i in range(num_job):
        row = f.readline().strip().split('\t')
    
        allOrder[data_ID].allJob[i].permut_m = [int(row[2*j]) for j in range(num_machine)]
        allOrder[data_ID].allJob[i].permut_p = [int(row[2*j+1]) for j in range(num_machine)]
  
    f.close()

    # 計算訂單特徵資料
    sum_p_of_job = []
    for i in range(num_job):
        sum_p_of_job.append(sum([allOrder[data_ID].allJob[i].permut_p[j] for j in range(num_machine)]))
        
    allOrder[data_ID].sum_p = sum(sum_p_of_job)
    allOrder[data_ID].max_p = max([max([allOrder[data_ID].allJob[i].permut_p[j] for j in range(num_machine)]) for i in range(num_job)])
    allOrder[data_ID].min_p = min([min([allOrder[data_ID].allJob[i].permut_p[j] for j in range(num_machine)]) for i in range(num_job)])

    ## 一筆資料的六個訂單特徵資訊
    # print(allOrder[data_ID].num_job, num_machine, allOrder[data_ID].sum_p, allOrder[data_ID].max_p, allOrder[data_ID].min_p)

    # 計算「Machine加工資訊(D_m)」
    allOrder[data_ID].allMachine = [CMachine() for j in range(num_machine)]
    
    for j in range(num_machine):
        allOrder[data_ID].allMachine[j].permut_j = [int for i in range(num_job)]
        allOrder[data_ID].allMachine[j].permut_p = [int for i in range(num_job)]

    for i in range(num_job):
        for j in range(num_machine):
            allOrder[data_ID].allMachine[allOrder[data_ID].allJob[i].permut_m[j]].permut_j[i] = j
            allOrder[data_ID].allMachine[allOrder[data_ID].allJob[i].permut_m[j]].permut_p[i] = allOrder[data_ID].allJob[i].permut_p[j]


# 印到檔案
f = open('random.txt', 'w')

for data_ID in range(num_orderProductionData):
    num_job =  allOrder[data_ID].num_job
    
    for i in range(num_job):
        sum_p_of_job.append(sum([allOrder[data_ID].allJob[i].permut_p[j] for j in range(num_machine)]))
        
    solutionInMachine = [np.negative(np.ones(num_job)) for j in range(num_machine)]

    # Step 2: 用 NN 來產生 output layer 的解(probs)


    
    
    # random 產生排程順序
    for j in range(num_machine):
        solutionInMachine[j] = np.arange(0, num_job)
        np.random.shuffle(solutionInMachine[j])

    
    # Step 4 & 7: 計算特徵資訊
    solutionInMachine = [list(map(int, i)) for i in solutionInMachine]  # 全轉成int
    F_bar, C_max, Q_bar, mu, O_bar = computeFeaturedInfo(data_ID, solutionInMachine)

    f.write('%d: F_bar = %.3f, C_max = %.3f, Q_bar = %.3f\n' %(data_ID, F_bar, C_max, Q_bar))
    f.write('   mu = ')
    f.writelines(['%f\t' % j for j in mu])
    f.write('\n   O_bar = ')
    f.writelines(['%f\t' % j for j in O_bar])
    f.write('\n')
    




f.close()
print('=========================Finish random testing=========================================')