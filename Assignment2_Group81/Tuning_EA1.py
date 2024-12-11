###############################################################################
# EvoMan FrameWork - Assignment 2			                                  #
# Tuning Evolutionary algorithm n1                                            #
# Author: Group 81 			                                                  #
###############################################################################

#                                                                 IMPORTS
import random
from statistics import multimode
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import matplotlib.pyplot as plt

#TUNING RANGES
mutation_rate_range=[0.05,0.3]
mutation_par_range=[2,2]
tournament_size_range=[2,15]

n_tests=2

for n in range(n_tests): 

    

    mutation_rate = round(random.uniform(mutation_rate_range[0], mutation_rate_range[1]),2) # mutation rate
    tournament_size = int(random.uniform(tournament_size_range[0], tournament_size_range[1]))
    mutation_par=int(random.uniform(mutation_par_range[0], mutation_par_range[1]))
    algorithm = 'Tuning_EA1_'+str(n)+"_"+ str(mutation_rate)+ "_"+ str(tournament_size)+ "_"+ str(mutation_par) # select algorithm: nn, choreo
    
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        #Then paste the EA and be carefull of the variables above here:
    repetitions = 2 #number of repetitions for each enemy 

    #Initialization of parameters
    n_hidden_neurons = 10 # hidden neurons per layer
    upper_bound = 1
    lower_bound = -1
    pop_size = 100
    n_generations = 15
    run_mode = 'train' # train or test
    output_size = 5 # output size of neural network

    # Metrics initialization
    mean_fitness = np.zeros((repetitions,n_generations)) 
    maximum_fitness = np.zeros((repetitions,n_generations))

    #External directory
    experiments_dir = 'generalist_'+str(algorithm)
    if not os.path.exists(experiments_dir):
            os.makedirs(experiments_dir)

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
            if np.random.uniform(0 ,1)<=mutation_rate: #checks mutation rate
                newValue= indiv[i]+np.random.normal(0, 1)
                while (newValue>=upper_bound or newValue<=lower_bound):
                    newValue= indiv[i]+np.random.normal(0, 1)
                indiv[i] = newValue
        return indiv


    # Crossover function: 
    def crossover(pop,generation): 
    
        offspring = np.zeros((0,nn_params))  
        for p in range(0,pop.shape[0], 2): 
            # Parent selection with tournament 
            parent_1 = tournament(pop) 
            parent_2 = tournament(pop) 
            child_1 = [] 
            child_2 = [] 
            # Crossover is performed according to an adaptive crossover rate (1/generation). If above the crossover rate, a copy of the parents 
            # is made, and goes to mutation (asexual reproduction), otherwise a uniform crossover is applied before mutation.
            if (1/(mutation_par*(np.log10(generation+1)))) < np.random.uniform(): 
                child_1=parent_1 
                child_2=parent_2 
            else: 
            # Each gene of the children is taken from one of the parents each time according to a uniform probability distribution 
                for gene in range(0,len(parent_1)): 
                    
                    if 0.5 > np.random.uniform(): 
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


    def survivalSelection(pop, fit_pop):
        while (len(pop)!=pop_size):
            n_individuals = pop.shape[0]
            current_winner = random.randint(0, n_individuals-1) # random individual
            fit = fit_pop[current_winner] #fitness of the random individual
            for candidate in range(tournament_size-1): 
                candidate_index = random.randint(0, n_individuals-1)
                if fit_pop[candidate_index]< fit:
                    current_winner = candidate_index
                    fit = fit_pop[candidate_index]
            pop=np.delete(pop, current_winner, 0)
            fit_pop=np.delete(fit_pop, current_winner, 0)
        return pop, fit_pop
            

    #                                                                   MAIN



    for r in range(0,repetitions):
        previous_best = 0 # useful to compare best solutions

        # Creates directory to save experiments per repetition
        experiment_name = experiments_dir+'/generalist_'+str(algorithm)+'_rep'+str(r)
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        
        # Environment initialization
        env = Environment(experiment_name=experiment_name,
                        enemies=[1,5,6],
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest")
        env.state_to_log() # checks environment state
        ini = time.time()  # sets time marker

        # Number of parameters of our neural network
        nn_params = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

        # Loads file with the best solution for testing
        if run_mode =='test':
            mean_best =[]
            file_best = open(experiments_dir + '/test_runs_logs.txt','w') #Stores results in a file
            original_stdout = sys.stdout
            sys.stdout = file_best
            for r in range(0,repetitions):
                experiment_name = experiments_dir+'/generalist_'+str(algorithm)+'_rep'+str(r)
                bsol =np.loadtxt(experiment_name+'/best.txt') #saves best from each repetition
                print("="*100)
                print( '\n RUNNING SAVED BEST SOLUTION from '+ experiment_name+'/best.txt\n')
                fitness_best = [] #stores the fitness values for the 5 tests
                for t in range(0,5):
                    print("-"*100)
                    print(f"TEST {t}:")
                    # env.update_parameter('speed','fastest')
                    fitness_best.append( evaluate([bsol]))
                mean_best.append(np.mean(fitness_best)) #saves mean of tests for each repetition
                print(f"\nMean fitness best: {mean_best}")
            file_best.write(f"\nMeans of best solutions for {repetitions} repetitions: \n{mean_best}")
            file_best.close()
            sys.stdout = original_stdout
            metrics_test = open(experiments_dir +'/test_metrics_'+str(algorithm)+'.txt','w')
            metrics_test.write(str(mean_best))
            metrics_test.close()

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

            offspring = crossover(pop,i)  #  crossover
            fit_offspring = evaluate(offspring)   # evaluation
            pop = np.vstack((pop,offspring))
            fit_pop = np.append(fit_pop,fit_offspring)
            pop, fit_pop=survivalSelection(pop, fit_pop)

            best = np.argmax(fit_pop) # best solution in generation
            fit_pop[best] = float(evaluate(np.array([pop[best]]))[0]) 
            best_sol = fit_pop[best]

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
    file_metrics  = open(experiments_dir+'/metrics_'+str(algorithm)+'.txt','w')
    print( f'\n Saving results for EA1:\naverage_mean: {average_mean}\naverage_maximum: {average_maximum}\nstd_mean: {std_mean}\nstd_maximum: {std_maximum}')
    file_metrics.write(f'&average_mean: {average_mean}\n&average_maximum: {average_maximum}\n&std_mean: {std_mean}\n&std_maximum: {std_maximum}')
    file_metrics.close()

