###############################################################################
# EvoMan FrameWork - Assignment 1			                                  #
# Neural network evolutionary algorithm                                       #
# Author: Group 81 			                                                  #
###############################################################################

#                                                                 IMPORTS
import random
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from nn_controller import player_controller
import time
import numpy as np
from math import fabs,sqrt
import glob, os

#                                                                PARAMETERS
repetitions = 10 #number of repetitions for each enemy 
enemies = [1]
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
    
    return pop[current_winner],fit_pop[current_winner]

# Mutation function
def mutate(indiv):
    for i in range(0,len(indiv)):
        if np.random.uniform(0 ,1)<=mutation_rate: #checks mutation rate
            indiv[i] = indiv[i]+np.random.normal(0, 1)
    return indiv


# Crossover function: uniform crossover
def uniform_crossover(pop):

    offspring = np.zeros((0,nn_params)) 

    for p in range(0,pop.shape[0], 2):
        # Parent selection with tournament
        parent_1,fit1 = tournament(pop)
        parent_2,fit2 = tournament(pop)
        f1 = 6+fit1/6+fit1+6+fit2
        child_1 = []
        child_2 = []
        # Each gene of the children is taken from one of the parents each time according to a uniform probability distribution
        for gene in range(0,len(parent_1)): 
            if f1 > np.random.uniform():
                child_1.append(parent_1[gene]) # take gene from first parent
                child_2.append(parent_2[gene]) # take gene from second parent
            else:
                child_1.append(parent_2[gene])
                child_2.append(parent_1[gene])
        # Mutate both children
        mutated_child1=mutate(child_1)
        mutated_child2=mutate(child_2)
        # Add them to total offspring
        offspring = np.vstack((offspring, mutated_child1))
        offspring = np.vstack((offspring, mutated_child2)) 

    return offspring

#                                                                   MAIN
for enemy in enemies: # iterating over the enemies

    #Initialization of parameters
    n_hidden_neurons = 10 # hidden neurons per layer
    upper_bound = 1
    lower_bound = -1
    pop_size = 50
    n_generations = 20
    previous_best = 0 # useful to compare best solutions
    mutation_rate = 0.02 # mutation rate
    tournament_size = int(pop_size/5) 
    algorithm = 'nn' # select algorithm: nn, choreo
    run_mode = 'train' # train or test
    output_size = 5 # output size of neural network

    # Metrics initialization
    mean_fitness = np.zeros((repetitions,n_generations)) 
    maximum_fitness = np.zeros((repetitions,n_generations))

    #External directory
    experiments_dir = 'specialist2_'+str(algorithm)+'_'+str(pop_size)+'pop_'+str(n_generations)+'gen_enemy'+str(enemy)+'_0'+str(int(mutation_rate*10))+'mut'
    if not os.path.exists(experiments_dir):
            os.makedirs(experiments_dir)

    
    for r in range(0,repetitions):
        # Creates directory to save experiments per repetition
        experiment_name = experiments_dir+'/specialist_'+str(algorithm)+'_'+'enemy'+str(enemy)+'_rep'+str(r)
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        
        # Environment initialization
        env = Environment(experiment_name=experiment_name,
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest")
        env.state_to_log() # checks environment state
        ini = time.time()  # sets time marker

        # Number of parameters of our neural network
        nn_params = (env.get_num_sensors()+1)*n_hidden_neurons  +(n_hidden_neurons+1)*n_hidden_neurons+ (n_hidden_neurons+1)*output_size


        # Loads file with the best solution for testing
        if run_mode =='test':

            bsol = np.loadtxt(experiment_name+'/best.txt')
            print( '\n RUNNING SAVED BEST SOLUTION \n')
            env.update_parameter('speed','normal')
            evaluate([bsol])

            sys.exit(0)


        if not os.path.exists(experiment_name+'/evoman_solstate'):
            # Initializes population loading old solutions or generating new ones
            print( '\nNEW EVOLUTION\n')

            pop = np.random.uniform(lower_bound, upper_bound, (pop_size, nn_params))
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

            offspring = uniform_crossover(pop)  #  crossover
            fit_offspring = evaluate(offspring)   # evaluation
            # Survival selection (mu,lambda): all parents are discarded, offspring becomes the new population
            pop = offspring 
            fit_pop = fit_offspring

            best = np.argmax(fit_pop) # best solution in generation
            fit_pop[best] = float(evaluate(np.array([pop[best]]))[0]) 

            best = np.argmax(fit_pop) #maximum for each generation
            std  =  np.std(fit_pop)
            mean = np.mean(fit_pop) #mean for each generation 
            mean_fitness[r][i]=round(mean,6) #store mean for plot
            maximum_fitness[r][i] =round(fit_pop[best],6) #store max for plot

            # saves results
            file_aux  = open(experiment_name+'/results.txt','a')
            print( '\n GENERATION '+str(i)+' (rep '+str(r)+') '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
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
    std_mean =np.std(mean_fitness,0) #std of the mean of fitness function over 10 repetitions
    std_maximum = np.std(maximum_fitness,0) #std of maximum fitness function over 10 repetitions
    file_metrics  = open(experiments_dir+'/metrics_'+str(algorithm)+'_enemy'+str(enemy)+'.txt','w')
    print( f'\n Saving results for enemy {enemy}:\naverage_mean: {average_mean}\naverage_maximum: {average_maximum}\nstd_mean: {std_mean}\nstd_maximum: {std_maximum}')
    file_metrics.write(f'&average_mean: {average_mean}\n&average_maximum: {average_maximum}\n&std_mean: {std_mean}\n&std_maximum: {std_maximum}')
    file_metrics.close()

