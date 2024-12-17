import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
import re

algorithm = 'nn'
repetitions = 10
pop_size = 75
n_generations = 50
enemy= 3
mutation_rate = 0.2

if algorithm == 'chor':
    experiments_dir = 'specialist_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_00'+str(int(mutation_rate*100))+'mut'
else:
    experiments_dir = 'specialist_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_0'+str(int(mutation_rate*10))+'mut'

experiment_name = experiments_dir+'/best_tests.txt'

file_aux  = open(experiment_name,'r')
stopword=open(experiment_name,"r")
lines = file_aux.readlines()
#print(lines[0])

lines = stopword.read().split('\n')
file_aux.close()
e=[]
p=[]
for line in range(1, len(lines),2):
    arr= lines[line].split(';') 
    p.append(int(arr[2].split(':')[1]))
    e.append(int(arr[3].split(':')[1]))
player_life = np.array(p)
enemy_life = np.array(e)
individual_gain = player_life - enemy_life
print(f"Player life: {player_life}")
print(f"Enemy life: {enemy_life}")
file = open(f'{experiments_dir}/ind_gain_{algorithm}_enemy{enemy}.txt','w')
print(f"Individual gain {individual_gain}")
file.write(str(individual_gain))
file.close()

fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(individual_gain)
 
# show plot
plt.show()