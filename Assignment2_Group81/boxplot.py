import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
import re

algorithm = 'EA2'
repetitions = 10
pop_size = 100
n_generations = 50
enemies ='256'
mutation_rate = 0.16
experiments_dir = 'generalist_'+str(algorithm)

experiment_name = experiments_dir+'/test_runs_logs.txt'

file_aux  = open(experiment_name,'r')
stopword=open(experiment_name,"r")
lines = file_aux.readlines()
#print(lines[0])

lines = stopword.read().split('\n')
print(lines[7])
file_aux.close()
e=[]
p=[]
num_enemies = 8
gain_test = np.zeros(5)
gain = np.zeros(repetitions)
mark = 7
for j in range(0,repetitions):
    gain_test = np.zeros(5)
    print(j)
    for i in range(0,5):
        # iterates through tests
        ind_gains =[]
        p = []
        e = []
        for line in range(mark, (mark+num_enemies)+7,2):
            arr= lines[line].split(';') 
            print(arr)
            p.append(float(arr[2].split(':')[1]))
            e.append(float(arr[3].split(':')[1]))
        player_life = np.array(p)
        enemy_life = np.array(e)
        ind_gains.append(player_life - enemy_life) 
        #print(f"Player life: {player_life}")
        #print(f"Enemy life: {enemy_life}")
        #print(f"Ind gains: {ind_gains}")
        print(np.sum(np.array(ind_gains)))
        gain_test[i] = np.sum(np.array(ind_gains))
        #print(f"Gain {i} {gain_test[i]}")
        mark = line + 4
    #print(f"Gain test {gain_test}")
    gain[j] = np.mean(gain_test)
    mark +=6
print(f"GAIN: {gain}")
file = open(f'{experiments_dir}/gain_{algorithm}_{enemies}.txt','w')
print(f"Gain {gain}")
file.write(str(gain))
file.close()

fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(gain)
 
# show plot
plt.show()