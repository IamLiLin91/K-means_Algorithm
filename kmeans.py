import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

#Qns 3a
def loadData(filename):
    # Load data from file into X
    X = []
    count = 0
    
    text_file = open(filename, "r")
    lines = text_file.readlines()
        
    for line in lines:
        X.append([])
        words = line.split()  
        # Convert values of the first attribute into float
        for word in words:
            X[count].append(float(word))
        count += 1
    
    return np.asarray(X)


#checking whether loadData function is working
testing = loadData('test.txt')
print(testing)
print(' ')

#Load the  lightening data
load_data=loadData('2010825.txt')
raw_data= pd.DataFrame({'latitude': load_data[:, 0], 'longitude': load_data[:, 1]})
raw_data.insert(2,"clusterID", 0)#insert 'clusterID' column
X=raw_data.to_numpy()
print(X)
print(' ')

#check the data's size
print(X.shape)
print(' ')

#Qns 3b
plt.figure(figsize=(15, 10))  
plt.scatter(X[:,0], X[:,1])
plt.title("Initial data plot", fontsize=20)
plt.show();


#Qns 3d
def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))

def errCompute(data,Mean_pt_lst):
    lst=[]
    for i in range(Mean_pt_lst.shape[0]):
        grp=data[data[:,-1]==i,:2]
        dis=sum(euclidean(Mean_pt_lst[i], grp))
        lst.append(dis)
    return sum(lst)/data.shape[0]

#checking whether errCompute function is working
M=np.array([[0,0]])
print(round(errCompute(X,M),2))
print(' ')


#Qns 3f
def Group(X,M):
    lst = []
    data=X[:,:-1]
    for pts in M:
        distances = euclidean(pts, data) #Step 2
        lst.append(distances)
    zipped = zip(*lst)
    index_lst=[]
    for row in list(zipped):
        index_lst.append(row.index(min(row)))
    new_lst=np.reshape(index_lst,(-1,1))
    new_X = np.concatenate((data,new_lst),axis=1)
    return new_X


#checking whether Group function is working
x=Group(X,np.loadtxt("..\Tee_Li_Lin_4\iniMeans\iniMeans_5.txt")) 
print(x)
print('')

#checking the errCompute value
err_5=errCompute(x,np.loadtxt("..\Tee_Li_Lin_4\iniMeans\iniMeans_5.txt"))
print(round(err_5,2))
print('')

#Qns 3h
def calcMean(data,M_lst):
    lst=[]
    for m in range(M_lst.shape[0]):
        cluster_grp=data[data[:,-1]==m,:2]
        lst.append(np.mean(cluster_grp[:,:2], axis=0))
        result=np.vstack(lst)
    return result

#checking the calcMean function
new_m=calcMean(x,np.loadtxt("..\Tee_Li_Lin_4\iniMeans\iniMeans_5.txt"))
print(new_m)
print('')


def final_cal(data, M):
    M_new=np.zeros_like(M)
    clust=np.zeros_like(data)
    grp=Group(data,M)#dataset with initial labellings
    err=errCompute(grp,M)
    new_err=0
    count=0
    K_val=M.shape[0]
    while (M==M_new).all()==False and err!=new_err:
        M_new=M
        clust=Group(data,M_new)
        new_err=err
        M=calcMean(clust,M)
        err=errCompute(clust,M)
        count+=1
    return err,M_new,clust,count,K_val

print('Results for K=5 using top 5 objects from X')
print('----------------------------------------------------------')
t5 = time.process_time()
outcome5=final_cal(X, np.loadtxt("..\Tee_Li_Lin_4\iniMeans\iniMeans_5.txt"))
elapsed_time5 = time.process_time() - t5
print(f'\nTotal number of iterations for K={outcome5[4]}: {outcome5[3]}')
print(f'\nFinal errCompute for K={outcome5[4]}: {round(outcome5[0],2)}')
print(f'\nFinal M for K={outcome5[4]}:\n{outcome5[1]}')
print(f'\nFinal Group for K={outcome5[4]}:\n{outcome5[2]}')
print(f'\nTime taken for K={outcome5[4]}:{round(elapsed_time5,3)}s')
print('')

plt.figure(figsize=(15, 10))
graphK5=outcome5[2]
pointK5=outcome5[1]
plt.scatter(graphK5[:,0], graphK5[:,1], c=graphK5[:,2])
plt.scatter(pointK5[:,0], pointK5[:,1], marker='*', s=200, c='red', label='centroids')
plt.title("Cluster data plot for K=5", fontsize=20)
plt.legend()
plt.show();


#Qns 3i
"""
For K=50, K=100 as well as finding the optimal k, I used the random function to select my initial means. Below is the random function code:
def random_M(data,k):
    idx = np.random.choice(len(data), k, replace=False)
    centroids = data[idx, :2]
    return centroids
"""

print('Results for K=50 using random 50 objects from X')
print('----------------------------------------------------------')
t50 = time.process_time()
outcome50=final_cal(X, np.loadtxt("..\Tee_Li_Lin_4\iniMeans\iniMeans_50.txt"))
elapsed_time50 = time.process_time() - t50
print(f'\nTotal number of iterations for K={outcome50[4]}: {outcome50[3]}')
print(f'\nFinal errCompute for K={outcome50[4]}: {round(outcome50[0],2)}')
print(f'\nFinal M for K={outcome50[4]}:\n{outcome50[1]}')
print(f'\nFinal Group for K={outcome50[4]}:\n{outcome50[2]}')
print(f'\nTime taken for K={outcome50[4]}:{round(elapsed_time50,3)}s')
print('')

plt.figure(figsize=(20, 10))
graphK50=outcome50[2]
pointK50=outcome50[1]
plt.scatter(graphK50[:,0], graphK50[:,1], c=graphK50[:,2])
plt.scatter(pointK50[:,0], pointK50[:,1], marker='*', s=100, c='red', label='centroids')
plt.title("Cluster data plot for K=50", fontsize=20)
plt.legend()
plt.show();


print('Results for K=100 using random 100 objects from X')
print('----------------------------------------------------------')
t100 = time.process_time()
outcome100=final_cal(X, np.loadtxt("..\Tee_Li_Lin_4\iniMeans\iniMeans_100.txt"))
elapsed_time100 = time.process_time() - t100
print(f'\nTotal number of iterations for K={outcome100[4]}: {outcome100[3]}')
print(f'\nFinal errCompute for K={outcome100[4]}: {round(outcome100[0],2)}')
print(f'\nFinal M for K={outcome100[4]}:\n{outcome100[1]}')
print(f'\nFinal Group for K={outcome100[4]}:\n{outcome100[2]}')
print(f'\nTime taken for K={outcome100[4]}:{round(elapsed_time100,3)}s')

plt.figure(figsize=(20, 10))
graphK100=outcome100[2]
pointK100=outcome100[1]
plt.scatter(graphK100[:,0], graphK100[:,1], c=graphK100[:,2])
plt.scatter(pointK100[:,0], pointK100[:,1], marker='*', s=100, c='red', label='centroids')
plt.title("Cluster data plot for K=100", fontsize=20)
plt.legend()
plt.show();

#Choosing optimal K value
k_lst=[]
for i in range(1,21):
    tick=i*5
    k_lst.append(tick)
lst= np.load('..\Tee_Li_Lin_4\iniMeans\iniMeans_k.npy', allow_pickle=True)
list_cost=[]
for i in range(len(lst)):
    cost=final_cal(X, lst[i])[0]
    list_cost.append(cost)
plt.figure(figsize=(20, 10))
sns.lineplot(x=k_lst, y=list_cost, marker='o')
plt.xlabel('K values')
plt.ylabel('Mean Values')
plt.title("Mean values for various K", fontsize=20)
plt.show()