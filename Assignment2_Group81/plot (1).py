import matplotlib
import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
import re

repetitions = 10
pop_size = 100
n_generations = 50
enemies = [2,5,6]

algorithm = 'EA1'
experiments_dir_1 = 'generalist_'+str(algorithm)
experiment_name_1 = experiments_dir_1+'/metrics_'+str(algorithm)+'.txt'
algorithm = 'EA2'
experiments_dir_2 = 'generalist_'+str(algorithm)
experiment_name_2 = experiments_dir_2+'/metrics_'+str(algorithm)+'.txt'
file_aux  = open(experiment_name_1,'r')
stopword=open(experiment_name_1,"r")
lines = file_aux.readlines()

lines = stopword.read().replace("[","").replace("]","").replace(":","").replace("\n","").split('&')
lines = [" ".join(line.split()) for line in lines]
lines = [re.sub(' +', ' ',line) for line in lines]

file_aux.close()
data=[]
for line in range(1, len(lines)):
    arr= lines[line].split(' ')    
    data.append(arr)
arrayOfData=[]
for ar in range(0,len(data)):
    array=[]
    for el in range(1,len(data[ar])):
        array.append(float(data[ar][el]))
    arrayOfData.append(array)

avgMean_chor=arrayOfData[0]
avgMax_chor = arrayOfData[1]
stdMean_chor=arrayOfData[2]
stdMax_chor = arrayOfData[3]
upperStd_Mean_chor=np.add(avgMean_chor,stdMean_chor)
lowerStd_Mean_chor=np.subtract(avgMean_chor, stdMean_chor)
upperStd_Max_chor = np.add(avgMax_chor,stdMax_chor)
lowerStd_Max_chor = np.subtract(avgMax_chor,stdMax_chor)

file_aux  = open(experiment_name_2,'r')
stopword=open(experiment_name_2,"r")
lines = file_aux.readlines()

lines = stopword.read().replace("[","").replace("]","").replace(":","").replace("\n","").split('&')
lines = [" ".join(line.split()) for line in lines]
lines = [re.sub(' +', ' ',line) for line in lines]

file_aux.close()
data=[]
for line in range(1, len(lines)):
    arr= lines[line].split(' ')    
    data.append(arr)
arrayOfData=[]
for ar in range(0,len(data)):
    array=[]
    for el in range(1,len(data[ar])):
        array.append(float(data[ar][el]))
    arrayOfData.append(array)

avgMean_nn=arrayOfData[0]
avgMax_nn = arrayOfData[1]
stdMean_nn=arrayOfData[2]
stdMax_nn = arrayOfData[3]
upperStd_Mean_nn=np.add(avgMean_nn,stdMean_nn)
lowerStd_Mean_nn=np.subtract(avgMean_nn, stdMean_nn)
upperStd_Max_nn = np.add(avgMax_nn,stdMax_nn)
lowerStd_Max_nn = np.subtract(avgMax_nn,stdMax_nn)

fig, ax = plt.subplots()
ax.plot(avgMean_chor,"b-")
ax.fill_between(np.arange(0,len(avgMean_chor),dtype=int),upperStd_Mean_chor,lowerStd_Mean_chor, alpha=0.2)
ax.plot(avgMean_nn,"r-")
ax.fill_between(np.arange(0,len(avgMean_nn),dtype=int),upperStd_Mean_nn,lowerStd_Mean_nn, alpha=0.2)



plt.title('Mean (Enemies '+str(enemies)+')',fontsize=20)
plt.legend(["EA1", "Std EA1","EA2","Std EA2"], fontsize=14)
plt.ylabel('Fitness ',fontsize = 14)
plt.xlabel('Generations',fontsize = 14)
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.savefig('mean_enemies'+str(enemies),dpi=300 )
plt.show()

fig, ax = plt.subplots()
ax.plot(avgMax_chor,"b-")
ax.fill_between(np.arange(0,len(avgMax_chor),dtype=int),upperStd_Max_chor,lowerStd_Max_chor, alpha=0.2)
ax.plot(avgMax_nn,"r-")
ax.fill_between(np.arange(0,len(avgMax_nn),dtype=int),upperStd_Max_nn,lowerStd_Max_nn, alpha=0.2)


plt.title('Maximum (Enemies '+str(enemies)+')',fontsize=20)
plt.legend(["EA1", "Std EA1","EA2","Std EA2"], fontsize=14)
plt.ylabel('Fitness ',fontsize = 14)
plt.xlabel('Generations',fontsize = 14)
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.savefig('max_enemy'+str(enemies),dpi=300 )
plt.show()