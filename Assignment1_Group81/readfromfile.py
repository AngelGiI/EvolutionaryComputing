from numpy import loadtxt
import numpy as np
algorithm = 'nn'
repetitions = 10
pop_size = 100
n_generations = 30
enemy= 2
mutation_rate = 0.2


if algorithm == 'chor':
    experiments_dir = 'specialist_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_00'+str(int(mutation_rate*100))+'mut'
else:
    experiments_dir = 'specialist_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_0'+str(int(mutation_rate*10))+'mut'

mean_fitness = np.zeros((repetitions,n_generations)) 
maximum_fitness = np.zeros((repetitions,n_generations))
for i in range(0,repetitions):
    experiment_name = experiments_dir+'/specialist_'+str(algorithm)+'_'+'enemy'+str(enemy)+'_rep'+str(i)+'/results.txt'
    print(f"Repetition {i}")
    f  = open(experiment_name,'r')
    lines = f.readlines()
    for index, line in enumerate (lines):
        text = line.strip()
        text_splitted = text.split(" ")
        if (index>=3 and index <33):
            print(text.split(" ")[2])
            mean = float(text.split(" ")[2])
            best = float(text.split(" ")[1])
            gen = int(text.split(" ")[0])
            mean_fitness[i][gen] = round(mean,6)
            maximum_fitness[i][gen] = round(best,6)

#print(mean_fitness)
#print(maximum_fitness)
average_mean = np.mean(mean_fitness,0) #average of the mean of fitness function over 10 repetitions
average_maximum = np.mean(maximum_fitness,0) #average of maximum fitness function over 10 repetitions
std_mean = np.std(mean_fitness,0) #std of the mean of fitness function over 10 repetitions
std_maximum = np.std(maximum_fitness,0) #std of maximum fitness function over 10 repetitions
file_metrics  = open(experiments_dir+'/metrics_'+str(algorithm)+'_enemy'+str(enemy)+'.txt','w')
print( f'\n Saving results for enemy {enemy}:\naverage_mean: {average_mean}\naverage_maximum: {average_maximum}\nstd_mean: {std_mean}\nstd_maximum: {std_maximum}')
file_metrics.write(f'&average_mean: {average_mean}\n&average_maximum: {average_maximum}\n&std_mean: {std_mean}\n&std_maximum: {std_maximum}')
file_metrics.close()