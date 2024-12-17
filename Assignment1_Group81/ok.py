
for i in range(0,10):
    f = open('C:\Users\Angel\Desktop\EC\Assignment\FinalCodeT1\Assignment1_Group81\specialistProbCross_nn_100pop_30gen_enemy1_01mut\specialist_nn_enemy1_rep' +str(i)+ '/results.txt', 'r')
    for line in f:
        a = line.split(' ')






"""average_mean = np.mean(mean_fitness, 0)  # average of the mean of fitness function over 10 repetitions
average_maximum = np.mean(maximum_fitness, 0)  # average of maximum fitness function over 10 repetitions
std_mean = np.std(mean_fitness, 0)  # std of the mean of fitness function over 10 repetitions
std_maximum = np.std(maximum_fitness, 0)  # std of maximum fitness function over 10 repetitions
file_metrics = open(experiments_dir + '/metrics_' + str(algorithm) + '_enemy' + str(enemy) + '.txt', 'w')
print(
    f'\n Saving results for enemy {enemy}:\naverage_mean: {average_mean}\naverage_maximum: {average_maximum}\nstd_mean: {std_mean}\nstd_maximum: {std_maximum}')
file_metrics.write(
    f'&average_mean: {average_mean}\n&average_maximum: {average_maximum}\n&std_mean: {std_mean}\n&std_maximum: {std_maximum}')
file_metrics.close()"""