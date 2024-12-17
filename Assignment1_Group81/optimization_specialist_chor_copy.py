###############################################################################
# EvoMan FrameWork - Assignment 1			                                  #
# Coreography: brute-force evolutionary algorithm                             #
# Author: Group 81 			                                                  #
###############################################################################

#                                                                 IMPORTS
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from chor_controller import player_controller
import time
import numpy as np
import random
import glob, os
import matplotlib.pyplot as plt


#                                                                PARAMETERS
repetitions = 10 #number of repetitions for each enemy 
enemies = [3]
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

#                                                            AUXILIARY FUNCTIONS
# Runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# Evaluation of population
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# Tournament for parent selection
def tournament(pop):

    n_individuals = pop.shape[0]
    current_winner = random.randint(0, n_individuals-1) # random individual
    fit = fit_pop[current_winner] #fitness of the random individual
    for candidate in range(tournament_size-1): 
        candidate_index = random.randint(0, n_individuals-1)
        if fit_pop[candidate_index]> fit:
            current_winner = candidate_index
            fit = fit_pop[candidate_index]
    
    return pop[current_winner]  

# Mutation function
def mutate(indiv):
    for i in range(0,len(indiv)):
        if np.random.uniform(0 ,1)<=mutation_rate:
            indiv[i] = np.random.randint(0, 6)
    return indiv


# crossover
def crossover(pop):

    offspring = np.zeros((0,step_num))
    geneSteps=5
    for p in range(0,pop.shape[0], 2):
        # Parent selection with tournament
        parent_1 = tournament(pop)
        parent_2 = tournament(pop)
        child_1 = []
        child_2 = []
        # Each 5-gene block of the children is taken from one of the parents each time according to a uniform probability distribution
        for gene in range(0,len(parent_1),geneSteps): # 
            if 0.5 > np.random.uniform():
                for i in range(geneSteps):
                    child_1.append(parent_1[gene+i]) # take genes from first parent
                    child_2.append(parent_2[gene+i]) # take genes from second parent
            else:
                for i in range(geneSteps):
                    child_1.append(parent_2[gene+i])
                    child_2.append(parent_1[gene+i])
        # Mutate both children
        mutated_child1=mutate(child_1)
        mutated_child2=mutate(child_2)
        offspring = np.vstack((offspring, mutated_child1))
        offspring = np.vstack((offspring, mutated_child2)) 

    return offspring


#                                                                   MAIN

for enemy in enemies: # iterating over the enemies

    #Initialization of parameters
    pop_size = 75
    n_generations = 50
    mutation_rate = 0.05 # mutation rate
    tournament_size = int(pop_size/10) 
    algorithm = 'chor' # select algorithm: nn, choreo
    run_mode = 'test' # train or test
    step_num = 1000 #every individual is a sequence of 1000 moves.

    # Metrics initialization
    mean_fitness = np.zeros((repetitions,n_generations)) 
    maximum_fitness = np.zeros((repetitions,n_generations))

    #External directory
    experiments_dir = 'specialist_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_00'+str(int(mutation_rate*100))+'mut'
    if not os.path.exists(experiments_dir):
            os.makedirs(experiments_dir)

    
    for r in range(0,repetitions):
        previous_best = 0 # useful to compare best solutions

        # Creates directory to save experiments per repetition
        experiment_name = experiments_dir+'/specialist_'+str(algorithm)+'_'+'enemy'+str(enemy)+'_rep'+str(r)
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        # Environment initialization
        env = Environment(experiment_name=experiment_name,
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=player_controller(),
                        enemymode="static",
                        level=2,
                        speed="fastest")
        env.state_to_log() # checks environment state
        ini = time.time()  # sets time marker
        
        # Loads file with the best solution for testing
        if run_mode =='test':
            mean_best =[]
            file_best = open(experiments_dir + '/best_tests.txt','w') #Stores results in a file
            original_stdout = sys.stdout
            sys.stdout = file_best
            for r in range(0,repetitions):
                experiment_name = experiments_dir+'/specialist_'+str(algorithm)+'_'+'enemy'+str(enemy)+'_rep'+str(r)
                bsol =np.loadtxt(experiment_name+'/best.txt') #saves best from each repetition
                fitness_best = [] #stores the fitness values for the 5 tests
                sys.stdout = original_stdout
                env = Environment(experiment_name=experiment_name,
                                        enemies=[enemy],
                                        playermode="ai",
                                        player_controller=player_controller(),
                                        enemymode="static",
                                        level=2,
                                        speed="fastest")
                env.state_to_log() # checks environment state
                ini = time.time()  # sets time marker
                sys.stdout = file_best
                # env.update_parameter('speed','fastest')
                evaluate([bsol])
               # mean_best.append(np.mean(fitness_best)) #saves mean of tests for each repetition
               # print(f"\nMean fitness best: {mean_best}")
           # file_best.write(f"\nMeans of best solutions for {repetitions} repetitions: \n{mean_best}")
            file_best.close()
            sys.stdout = original_stdout

            #Box plots
            #fig = plt.figure(figsize =(7,7))
            #plt.boxplot(mean_best)
            #plt.title(f"Box plot Neural Network Enemy {enemy}")
            #plt.ylabel('Individual gain')
            #plt.xlabel('EA Neural Network')
            #plt.savefig(f"Box plot Neural Network Enemy {enemy}")
            #plt.show()

            sys.exit(0)



        if not os.path.exists(experiment_name+'/evoman_solstate'):
            # Initializes population loading old solutions or generating new ones
            print( '\nNEW EVOLUTION\n')

            pop = np.random.randint(0, 6, (pop_size, step_num))
            fit_pop = evaluate(pop) # computes fitness function
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)
            first_gen = 0
            mean_fitness[r][first_gen]=round(mean,6) # store mean for plot
            maximum_fitness[r][first_gen] =round(fit_pop[best],6) # store max for plot
            solutions = [pop, fit_pop]
            env.update_solutions(solutions)

        else:

            print( '\nCONTINUING EVOLUTION\n')

            env.load_state()
            pop = env.solutions[0]
            fit_pop = env.solutions[1]

            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)
            first_gen = 0
            mean_fitness[r][first_gen]=round(mean,6) #store mean for plot
            maximum_fitness[r][first_gen] =round(fit_pop[best],6) #store max for plot

            # finds last generation number
            file_aux  = open(experiment_name+'/gen.txt','r')
            first_gen = int(file_aux.readline())
            file_aux.close()




        # saves results for first population
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\n\ngen best mean std')
        print( '\n GENERATION '+str(first_gen)+' (rep '+str(r)+') '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(first_gen)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()


#                                                                 Evolution 
        for i in range(first_gen+1, n_generations):

            offspring = crossover(pop)  # crossover
            fit_offspring = evaluate(offspring)   # evaluation
            # Survival selection (mu,lambda): all parents are discarded, offspring becomes the new population
            pop = offspring
            fit_pop = fit_offspring

            best = np.argmax(fit_pop) #best solution in generation
            fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) 

            best = np.argmax(fit_pop) #maximum for each generation
            std  =  np.std(fit_pop)
            mean = np.mean(fit_pop) #mean for each generation 
            mean_fitness[r][i]=round(mean,6) #store mean for plot
            maximum_fitness[r][i] =round(fit_pop[best],6) #store max for plot

            # saves results
            file_aux  = open(experiment_name+'/results.txt','a')
            print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
            file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
            file_aux.close()

            # saves generation number
            file_aux  = open(experiment_name+'/gen.txt','w')
            file_aux.write(str(i))
            file_aux.close()

            # saves file with the best solution
            if previous_best <= fit_pop[best]: #checks if it is better than the previous saved best solution
                np.savetxt(experiment_name+'/best.txt',pop[best])
                previous_best = fit_pop[best]
                
            # saves simulation state
            solutions = [pop, fit_pop]
            env.update_solutions(solutions)
            env.save_state()

        fim = time.time() # prints total execution time for experiment
        print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

        file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
        file.close()

        env.state_to_log() # checks environment state

        
    #                                                       Results for plots

    average_mean = np.mean(mean_fitness,0) #average of the mean of fitness function over 10 repetitions
    average_maximum = np.mean(maximum_fitness,0) #average of maximum fitness function over 10 repetitions
    std_mean = np.std(mean_fitness,0) #std of the mean of fitness function over 10 repetitions
    std_maximum = np.std(maximum_fitness,0) #std of maximum fitness function over 10 repetitions
    file_metrics  = open(experiments_dir+'/metrics_'+str(algorithm)+'_enemy'+str(enemy)+'.txt','w')
    print( f'\n Saving results for enemy {enemy}:\naverage_mean: {average_mean}\naverage_maximum: {average_maximum}\nstd_mean: {std_mean}\nstd_maximum: {std_maximum}')
    file_metrics.write(f'&average_mean: {average_mean}\n&average_maximum: {average_maximum}\n&std_mean: {std_mean}\n&std_maximum: {std_maximum}')
    file_metrics.close()
