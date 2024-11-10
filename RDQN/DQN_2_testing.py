import numpy as np 
import random
#import cloudDecision

np.random.seed(3)           # 若要每次結果都不一樣的話，把這行註解掉

# 1. 讀取訂單生產資料(存在orderInfo[]、num_job、num_machine) # 注意機台數一定要一樣

num_orderProductionData = 10 # 共有幾筆訂單生產資料(讀幾個檔案, e.g., dmu01.txt, dumu02.txt, ...) 
num_disptching_rule = 3     # 共有幾個派遣規則(0.FIFO; 2.MOPNR; 3.LOPT)

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
f = open('testing/dmu01' + '.txt', 'r')
row = f.readline().rstrip().split('\t')
num_machine = int(row[1])  # 機器數目不管讀哪個檔都一樣，所以設為全域變數
f.close()

# 用於建立 blue table (表1) 的一個橫列
class blue_row:
    job = []
    beg_time = []
    end_time = []

# NN的參數
nn_input_dim = 8 + 2 * num_machine                    # NN 的 input layer 的維度
nn_hdim = 160                                           # hidden layer 內神經元的個數
nn_output_dim = num_disptching_rule * num_machine     # output layer 的維度 

# NN用到的Gradient descent parameters (I picked these by hand) 
nn_epsilon = 0.01          # learning rate for gradient descent 
nn_reg_lambda = 0.01       # regularization strength 

# 初始化NN會用到的權重與誤差
W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim) 
b1 = np.zeros((1, nn_hdim)) 
W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim) 
b2 = np.zeros((1, nn_output_dim)) 

# solutionInMachine[j] = 紀錄 machine j 的解 (i.e., 機台j內job的處理順序)

# 讀取由 training 所訓練出來的模型的參數
f = open('2_testing_input.txt', 'r')
row = f.readline().rstrip().split('\t')
F_bar = float(row[0])
C_max = float(row[1])
Q_bar = float(row[2])

row = f.readline().rstrip().split('\t')
mu = [float(row[j]) for j in range(0, num_machine)]

row = f.readline().rstrip().split('\t')
O_bar = [float(row[j]) for j in range(0, num_machine)]

for i in range(0, nn_input_dim):
    row = f.readline().rstrip().split('\t')
    W1[i] = [float(row[j]) for j in range(0, nn_hdim)]


for i in range(0, nn_hdim):
    row = f.readline().rstrip().split('\t')
    W2[i] = [float(row[j]) for j in range(0, nn_output_dim)]

f.close()


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

# C_max對應到哪一類
def class_C_max(c, sum_p):
    class_ID = 3
    
    if c < 0.5 * sum_p :
        class_ID = 0
    elif c < 0.7 * sum_p :
        class_ID = 1
    elif c < 0.9 * sum_p :
        class_ID = 2
        
    return class_ID


# ========== 主程式 ==========
num_jobList = []
# Step 1: 讀取所有「訂單生產資料」 ==> allOrder[]
for data_ID in range(num_orderProductionData):
    # 檔名是 dmu01.txt ~ dmu{num_orderProductionData}.txt
    f = open('testing/dmu' + str(data_ID+1).zfill(2) + '.txt', 'r')
#    f = open('dmu' + str(data_ID+1).zfill(2) + '.txt', 'r')

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
f = open('2_testing_output.txt', 'w')

for data_ID in range(num_orderProductionData):
    num_job =  allOrder[data_ID].num_job
    
    for i in range(num_job):
        sum_p_of_job.append(sum([allOrder[data_ID].allJob[i].permut_p[j] for j in range(num_machine)]))
        
    solutionInMachine = [np.negative(np.ones(num_job)) for j in range(num_machine)]

    # Step 2: 用 NN 來產生 output layer 的解(probs)
    # 設定 NN 要讀的 input
    X = np.array([allOrder[data_ID].num_job, num_machine,
         allOrder[data_ID].sum_p, allOrder[data_ID].max_p, allOrder[data_ID].min_p,
         F_bar, C_max, Q_bar] \
        + [mu[j] for j in range(num_machine)] \
        + [O_bar[j] for j in range(num_machine)])

    # Forward propagation 
    z1 = np.dot(X, W1) + b1 
    a1 = np.tanh(z1) 
    z2 = np.dot(a1, W2) + b2 
    exp_scores = np.exp(z2) 
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
    
    # 從 NN 的 output layer 來計算 rule_ID[j] = machine j 要採用哪個rule
    rule_ID = [np.argmax([probs[0][num_disptching_rule*j + k] for k in range(num_disptching_rule)]) for j in range(num_machine)]
    
    # Step 3: 根據 rule_ID 來排各機台的排程
    for j in range(num_machine):
        # 初始解設定 --> 分三群(0的一群、-1的一群、num_machine-1的一群)
        jobGroup = [[], [], []]
        for i in range(num_job):
            if allOrder[data_ID].allMachine[j].permut_j[i] == 0:
                # solutionInMachine[j][i] = 0
                jobGroup[0].append(i)
            elif allOrder[data_ID].allMachine[j].permut_j[i] == num_machine-1:
                # solutionInMachine[j][i] = num_machine-1
                jobGroup[2].append(i)
            else:
                # solutionInMachine[j][i] = -1
                jobGroup[1].append(i)
        
        isReverse = False
        
        if rule_ID[j] == 0: # FIFO
            target_of_job = np.zeros(num_job)
            
            for i in range(num_job):
                for k in range(num_machine):
                    if allOrder[data_ID].allJob[i].permut_m[k] == j:
                        break
                    target_of_job[i] += allOrder[data_ID].allJob[i].permut_p[k]
                    
        elif rule_ID[j] == 1: # SPT
            target_of_job = allOrder[data_ID].allMachine[j].permut_p
        
        elif rule_ID[j] == 2: # LPT
            target_of_job = allOrder[data_ID].allMachine[j].permut_p
            isReverse = True
                    
        elif rule_ID[j] == 3: # MWKR
            target_of_job = np.zeros(num_job)
            
            for i in range(num_job):
                isStart = False
                for k in range(num_machine):
                    if allOrder[data_ID].allJob[i].permut_m[k] == j:
                        isStart = True
                    
                    if isStart :
                        target_of_job[i] += allOrder[data_ID].allJob[i].permut_p[k]
            
            isReverse = True
                    
        elif rule_ID[j] == 4: # MOPNR
            target_of_job = allOrder[data_ID].allMachine[j].permut_j
                
        elif rule_ID[j] == 5 or rule_ID[j] == 6: # SNQ or LNQ
            target_of_job = np.zeros(num_job)
            
            for i in range(num_job):
                isStart = False
                for k in range(num_machine):
                    if isStart :
                        target_of_job[i] = allOrder[data_ID].allJob[i].permut_p[k]
                        break
                    
                    if allOrder[data_ID].allJob[i].permut_m[k] == j:
                        isStart = True
            
            if rule_ID[j] == 6: # LNQ
                isReverse = True
        temp1 = computeSolutionInMachine(jobGroup, target_of_job, isReverse)
#            print("temp1",temp1[0])
        solutionInMachine[j], isOverlap = computeSolutionInMachine(jobGroup, target_of_job, isReverse)

       #處理卡關
        #1.check solutionInMachine[i]是否有重複元素
        if len(solutionInMachine[j])==len(set(solutionInMachine[j])):#沒重複
            solutionInMachine[j]
#                print(solutionInMachine[j])#test
            
        else:#重複
#                print('改',solutionInMachine[j])#test
            #2.找出重複的
            solution_same=list(set(solutionInMachine[j]))
            solution_same.sort()#solution_same=[0.0, 1.0, 3.0, 8.0]
            Final_solution=np.zeros(num_job)
            solution_count=0
            for p in range(num_job):
                same=[]
                for k in range(len(solutionInMachine[j])):
                    if(solutionInMachine[j][k]==p):
                        same.append(k)

                if len(same)==1:
#                        print(p,':',same)
#                        print('>>>>>',solution_count)
                    Final_solution[same]=solution_count
                    solution_count+=1
        
                same_MOPNR=[]
                MOPNR_order=np.zeros(len(same))
                if len(same)>1:
#                        print(p,':',same)
                    for s in range(len(same)):
#                            print('>>>>>',solution_count)
                        Final_solution[same[s]]=solution_count
                    solution_count+=len(same)
                    
                    #3.重複的用MOPNR修正
                    #3-1找MOPNR
                    for s in range(len(same)):
                            same_MOPNR.append(allOrder[data_ID].allMachine[j].permut_j[same[s]])
                            
#                        print('same MOPNR',same_MOPNR)
                    #3-2排MOPNR(數值大到小>>0~N)
                    count=0
                    for s in range(max(same_MOPNR),-1,-1):
                        check=0
                        for m in range(len(same_MOPNR)):
                            if(same_MOPNR[m]==s):      
                                MOPNR_order[m]=count
                                check=1
                        if check==1:
                            count+=1
                    
                    #MOPNR_order
                    a=np.zeros(len(MOPNR_order))
                    a_count=0
                    for s in range(int(max(MOPNR_order)+1)):
                        b=[]
                        for m in range(len(MOPNR_order)):
                            if(MOPNR_order[m]==s):
                                a[m]=a_count
                                b.append(k)
                        a_count+=len(b)
                    for s in range(len(MOPNR_order)):
                        MOPNR_order[s]=a[s]
 
                        
                    #check MOPNR_order 是否有重複
                    if len(MOPNR_order)==len(set(MOPNR_order)):#沒重複
                        for u in range(len(MOPNR_order)):
                            Final_solution[same[u]]+=MOPNR_order[u]
                    else:#有重複利用SPT排
                        same_MOPNR_order=[]
                        same_SPT=[]
                        SPT_order=[]                
                        for i in range(int(max(MOPNR_order))+1):
                            same_M=[]
                            check_same_M=0
                            for m in range(len(MOPNR_order)):
                                    if(MOPNR_order[m]==i):
                                        same_M.append(m)
#                                print(same_M)
                            if(len(same_M)>1):
                                    for t in range(len(same_M)):
                                       same_SPT.append(allOrder[data_ID].allMachine[j].permut_p[same[same_M[t]]])   
                                    SPT_order=sorted(same_SPT)
                                    list_SPT = list(set(SPT_order))
                                    for o in range(len(list_SPT)):
                                        for r in range(len(same_SPT)):
                                            if(list_SPT[o]==same_SPT[r]):
                                                SPT_order[r]=o
 
#                                        print(same_M)
#                                        print('SPT_order:',SPT_order)                
                                    #check SPT_order 是否有重複
                                    s_count=0
                                    if len(SPT_order)==len(set(SPT_order)):#沒重複                                             
                                        for t in range(len(same_M)):
                                            MOPNR_order[same_M[t]]+=SPT_order[t]
                                    else:#有重複
                                   
                                        SPT_order_final=np.zeros(len(SPT_order))
                                        random_count=0
                                        for o in range(max(SPT_order)+1):
                                            random_same=[]
                                            random_1=[]
                                            for r in range(len(SPT_order)):
                                                if(SPT_order[r]==o):
                                                    random_same.append(r)
                                            if(len(random_same)>1):
                                                for k in range(len(random_same)):
                                                    random_1.append(k)
                                                random.shuffle(random_1)
                                                for t in range(len(random_same)):
                                                    SPT_order_final[random_same[t]]+=random_count
                                                    SPT_order_final[random_same[t]]+=+random_1[t]
                                                random_count+=(len(random_same))
                                            if(len(random_same)==1):
                                                SPT_order_final[random_same[0]]+=random_count
                                                random_count+=1
#                                            print('=======SPT final::',SPT_order_final) 
                                        for t in range(len(same_M)):
                                            MOPNR_order[same_M[t]]+=SPT_order_final[t]
#                            print('====same===:',same)
#                            print('>>>>>++++++>>>>',MOPNR_order)
                        for u in range(len(MOPNR_order)):
                            Final_solution[same[u]]+=MOPNR_order[u]
                                
#                print('改之後>>',Final_solution)
            for k in range(len(solutionInMachine[j])):
                solutionInMachine[j][k]=Final_solution[k]
#            print('solutionInMachine[j]:',solutionInMachine[j])

    
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

print('=======================finish testning================')