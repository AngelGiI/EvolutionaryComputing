import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
import re

algorithm = 'EA1'
repetitions = 10
pop_size = 100
n_generations = 50
mutation_rate = 0.1

if algorithm == 'chor':
    experiments_dir = 'specialist_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_00'+str(int(mutation_rate*100))+'mut'
else:
    experiments_dir = 'specialistProbCross_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_0'+str(int(mutation_rate*10))+'mut'

experiment_name = experiments_dir+'/metrics_'+str(algorithm)+'_enemy'+str(enemy)+'.txt'

file_aux  = open(experiment_name,'r')
stopword=open(experiment_name,"r")
lines = file_aux.readlines()
#print(lines[0])

lines = stopword.read().replace("[","").replace("]","").replace(":","").replace("\n","").split('&')
lines = [" ".join(line.split()) for line in lines]
lines = [re.sub(' +', ' ',line) for line in lines]
file_aux.close()
data=[]
for line in range(1, len(lines)):
    arr= lines[line].split(' ')
    #print(arr)
    data.append(arr)
arrayOfData=[]
for ar in range(0,len(data)):
    array=[]
    for el in range(1,len(data[ar])):
        #print('\''+ str(data[ar][el])+'\'')
        array.append(float(data[ar][el]))
    arrayOfData.append(array)

avgMean=arrayOfData[0]
avgMax = arrayOfData[1]
stdMean=arrayOfData[2]
stdMax = arrayOfData[3]
upperStd_Mean=np.add(avgMean,stdMean)
lowerStd_Mean=np.subtract(avgMean, stdMean)
upperStd_Max = np.add(avgMax,stdMax)
lowerStd_Max = np.subtract(avgMax,stdMax)

fig, ax = plt.subplots()
ax.plot(avgMean,"r-")
ax.fill_between(np.arange(0,len(avgMean),dtype=int),upperStd_Mean,lowerStd_Mean, alpha=0.2)



#labels and legend
if algorithm == "chor":
    plt.title('Fitness Mean (Enemy '+str(enemy)+', choreography)')
else:
    plt.title('Fitness Mean (Enemy '+str(enemy)+', neural network)')
plt.legend(["Mean", "Standard deviation"])
plt.ylabel('Fitness (mean)')
plt.xlabel('Generations')
plt.grid(color='k', linestyle='-', linewidth=0.1)
#plt.savefig('mean_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_00'+str(int(mutation_rate*100))+'mut',dpi=300 )

plt.show()


fig, ax = plt.subplots()
ax.plot(avgMax,"r-")
ax.fill_between(np.arange(0,len(avgMax),dtype=int),upperStd_Max,lowerStd_Max, alpha=0.2)

#labels and legend
if algorithm == "chor":
    plt.title('Fitness Maximum (Enemy '+str(enemy)+', choreography)')
else:
    plt.title('Fitness Maximum (Enemy '+str(enemy)+', neural network)')
plt.legend(["Maximum", "Standard deviation"])
plt.ylabel('Fitness (maximum)')
plt.xlabel('Generations')
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.savefig('max_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_00'+str(int(mutation_rate*100))+'mut',dpi=300 )

plt.show()