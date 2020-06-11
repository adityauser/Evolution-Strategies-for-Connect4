import gc
import numpy
import random
import numpy as np
from pdb import set_trace as bb
from functools import reduce
import os
from colors import *
import argparse
import matplotlib.pyplot as plt
from termcolor import colored
import copy
import threading 
import concurrent.futures

parser = argparse.ArgumentParser()
#all command line options
parser.add_argument("--display", help="turn on rendering", action="store_true")
parser.add_argument("--mutation", help="whether to use regular mutations or SM",choices=['regular','SM-G-SUM','SM-G-ABS','SM-R','SM-G-SO'],default='regular')
parser.add_argument("--mutation_mag", help="magnitude of mutation operator",default=0.01)
parser.add_argument('--pop_size', help="population size",default=250)
parser.add_argument("--save", help="output file prefix for saving",default="out")
parser.add_argument("--hidden", help="number of hidden units per ann layer", default=15)
parser.add_argument("--name", help="path where plot needs to be saved", default="")
parser.add_argument("--save_reward", help="save the plot of avg reward", action="store_true")
parser.add_argument("--state_archive", help="create state archive", action="store_true")
parser.add_argument("--gen_length", help="number of evaluations in a generation", default=100)
parser.add_argument("--population_play", help="play matches against population", action="store_true")
parser.add_argument("--fast", help="don't run unnecessary part od code", action="store_true")
parser.add_argument("--init", help="init rule", default="xavier")
parser.add_argument("--message", help="description of the experiment", default="")
parser.add_argument("--celltype", help="recurrent cell type",default="lstm",choices=['lstm','gru','rnn'])
parser.add_argument("--layers", help="number of ann hidden layers",default=6)
parser.add_argument("--activation",help="ann activation function",default="relu")
parser.add_argument("--max_gen",help="total number of generation",default=100)
parser.add_argument("--domain",help="Experimental domain", default="connect_four",choices=['connect_four_CNN','breadcrumb_maze', 'connect_four'])
parser.add_argument("--frameskip",help="frameskip amount (i.e. query agent every X frames for action)", default="3")

#Parse arguments
args = parser.parse_args()
print(args)

#domain selection (whether the recurrent parity task or the breadcrumb hard maze)
if args.domain=='connect_four':
    import connect_domain as evolution_domain
elif args.domain=='connect_four_CNN':
    import connect_four_CNN as evolution_domain

else:
	raise("This file doesn't support "+args.domain)

#pop up rendering display (for breadcrumb maze domain)
do_display = args.display

#make save directory 
os.system("mkdir -p %s" % args.save)

#define dictionary describing ann
params = {'size':int(args.hidden),'af':args.activation,'layers':int(args.layers),'init':args.init,'celltype':args.celltype} 

#define dictionary describing domain
domain = {'name':args.domain,'difference_frames':False,'frameskip':int(args.frameskip),'history':1,'rgb':False,'incentive':'fitness'}

#initialize domain
evolution_domain.setup(domain,params)

#fire up pygame for visualization
import pygame
from pygame.locals import *


def find_fitness(agent):
    if agent==None:
        return 0
    else :
        return agent.fitness


#Maze rendering call (visualize behavior of population)
def render_maze(pop):
    pass

#takes population and gives out a child
def get_child(population, mutation_mag, evals, greedy_select=5):
	parents = random.sample(population, greedy_select)
	parent = reduce(lambda x, y: x if x.fitness > y.fitness else y, parents)
	child = parent.copy((evals, parent.identity[1]))
	child.mutate(mutation=args.mutation, mag=mutation_mag, fitness=parent.fitness)
	return child

#evaluation
def evaluate(individual, test_opponents, **kwargs):
	r = 0
	for against in test_opponents:
		#this returns reward, terminal_state, broken, the_game
		returns = individual.map(against=against, **kwargs)
		r+=returns[0]
	return r, returns


if (__name__ == '__main__'):

    #initialize empty population
    population = []

    #number of threads
    threads = 4  

    #for testing pourpose. Here ice means we are re-evaluting.
    #Re-evaluation of performance
    ice = []
    #Re-evaluation of fitness
    ice_population_fitness = []

    #performance check of population
    performance = []

    #average population fitness check after every 100 evals
    population_fitness = []

    #Aditya: This code is not relevent for us. 
    #placeholders to hold champion
    best_fit = -1e9
    best_ind = None
    best_beh = None

    #highest ice_population_fitness
    best_pop_fitness = -1
    best_eval = 0

    #grab population size
    psize = int(args.pop_size)

    #number of trials
    trials = 100

    # Opponents for the tournament
    opponents = []
    perfo_allOppo = []
    opponent_fitness = []
    random_opponent = 10
    for _ in range(trials):
        opponents.append(None)
    all_opponents = copy.deepcopy(opponents)

    #broken
    broken = False

    #initialize population
    for k in range(psize):

        threads = []
        robot = evolution_domain.individual((1, k+1))

        #initialize random parameter vector
        robot.init_rand()

        #evaluate in domain
        for _ in range(trials):
            thread = threading.Thread(target=robot.map, args=(args.state_archive, )) 
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        robot.parent = None

        #add to population
        population.append(robot)

    print("population_fitness ", np.mean([x.fitness for x in population]))

    #Dominance
    dominance = np.ones(psize)

    #solution flag
    solved = False

    #broken
    broken = False

    #we spent generations looking at the population
    evals = 0

    population_play = args.population_play

    #number of tournaments in each generation
    gen_length = int(args.gen_length)

    #parse max evaluations
    max_gen = int(args.max_gen)

    #tournament size
    greedy_kill = 5
    greedy_select = 2
    #parse mutation intensity parameter
    mutation_mag = float(args.mutation_mag)

    #evolutionary loop
    while evals < max_gen:        

        print('Evaluation no. ' ,evals)

        trial_reward = 0

        broken = True

        evals += 1

        tournament_played = 0

        #initialize empty population for childs
        young_ones = []


        # Evaluate population performance against random agents
        test = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
        	for test_agent in population:
        		future = executor.submit(evaluate, individual=test_agent, test_opponents=opponents, push_all=args.state_archive, test=True)
        		r, _ = future.result()
        		test.append(r/len(opponents))
        performance.append(test)
        if test[-1]<=-1:
            raise('check2')


        
        # Evaluate population performance against all_opponents, basicly this is the fitness in case of population_play.

        test = []
        if args.population_play and not args.fast:
        	with concurrent.futures.ThreadPoolExecutor() as executor:
        		for test_agent in population:
        			future = executor.submit(evaluate, individual=test_agent, test_opponents=all_opponents, push_all=args.state_archive, test=True)
        			r, _ = future.result()
        			test.append(r/len(all_opponents))
        		perfo_allOppo.append(test)
        		if test[-1]<=-1:
        			raise('check3')            

        #Get the average fitness of population
        print (bcolors.WARNING)

        if args.population_play:
            print('opponent_fitness ', np.mean([find_fitness(x) for x in opponents]))
            opponent_fitness.append(np.mean([find_fitness(x) for x in opponents]))

        print("population_fitness ", np.mean([x.fitness for x in population]))
        print (bcolors.ENDC)
        population_fitness.append(np.mean([x.fitness for x in population]))

        #Re-evaluate the fitness of every agent in the population
        #[print(x.fitness, x.identity) for x in population]
        print (bcolors.WARNING)
        print('Recalculating the fitness')
        print (bcolors.ENDC)
        for parent in population:
            #parent.matchs_played = 0
            #parent.net_reward = 0
            #parent.fitness = 0
            threads = []
            for against in opponents:
            	thread = threading.Thread(target=parent.map, args=(args.state_archive, against))
            	thread.start()
            	threads.append(thread)
            for thread in threads:
            	thread.join()
           
            if parent.fitness<=-1:
                print(broken, parent.fitness,"check4")
                raise('check4')
        print (bcolors.WARNING)
        print("ice_population_fitness ", np.mean([x.fitness for x in population]))
        print (bcolors.ENDC)
        ice_population_fitness.append(np.mean([x.fitness for x in population]))

            
        #creation of new off springs, tournament selection and evalute in domain
        with concurrent.futures.ThreadPoolExecutor() as executor:
        	while tournament_played < gen_length:
        		tournament_played += 1
        		future = executor.submit(get_child, population=population, mutation_mag=mutation_mag, evals=evals, greedy_select=greedy_select)
        		child = future.result()
        		future = executor.submit(evaluate, individual=child, test_opponents=opponents, push_all=args.state_archive)
        		print('Generation no. {} and Tournament no. {}'.format(evals, tournament_played))
        		trial_reward, terminal_state = future.result()
        		if trial_reward/trials<=-1:
        			raise('check1')
        		print('Child average reward ' , trial_reward/trials)
        		print('child_fitness ', child.fitness)

        		young_ones.append(child)

        		if do_display:
        			print(terminal_state[0])
       
        # Add young_ones to population
        for child in young_ones:
        	dominance[child.identity[1]-1] += 1
        	population.append(child)

        #remove individuals from the pop using a tournament of same size
        for _ in range(gen_length):
        	to_kill = random.sample(population, greedy_kill)
        	to_kill = reduce(lambda x, y: x if x.fitness < y.fitness else y, to_kill)
        	population.remove(to_kill)
        	to_kill.kill()
        	dominance[to_kill.identity[1]-1] -= 1
        	del to_kill

        #Add some new members in the oppents team.
        if args.population_play and ice_population_fitness[-1]>0.80:
            for _ in range(int(trials/6)):
                new_ops = random.sample(population, int(psize/25))
                new_op = reduce(lambda x, y: x if x.fitness > y.fitness else y, parents)
                opponents[int(np.random.rand()*(len(opponents)-random_opponent))] = new_op
                all_opponents.append(new_op)

        if evals%20==0:
            #trials = trials*2 # exponential trials are currently not for population play
            #mutation_mag = mutation_mag*0.80
            print(colored('Updated the trials to', 'red'), trials)
            print(colored('Updated the mutation_mag to', 'red'), mutation_mag)

            #For population play
            for _ in range(trials - len(all_opponents)):
            	all_opponents.append(None)
				
        
        file = open(args.name+'_dominance.txt','a') 
        [file.write(str(x) + " ") for x in dominance]
        file.write('\n')
        file.close() 

        

    #Print the losing games
    for _ in range(5):
    	good_agents = random.sample(population, int(len(population)/10))
    	good_agent = reduce(lambda x, y: x if x.fitness > y.fitness else y, good_agents)
    	print(colored("Agent identity: ", 'green'), good_agent.identity)
    	print("Agent fitness: ", good_agent.fitness)
    	for against in opponents:
    		reward, terminal_state, broken, the_game = good_agent.map(against=against, test=True, print_game=True)
    		if reward<0:
    			print(colored("New opponent: ", 'yellow'))
    			for a_row in the_game:
    				[print(element, end=' ') for element in a_row]
    				print()

    # Plotting different varients of fitness.
    performance = np.array(performance)
    ice = np.array(ice)
    plt.plot(range(max_gen), np.mean(performance, axis=1))
    plt.plot(range(max_gen), population_fitness)
    plt.plot(range(max_gen), ice_population_fitness)
    if args.population_play:
        plt.plot(range(max_gen), opponent_fitness)
        if not args.fast:
        	plt.plot(range(max_gen), np.mean(np.array(perfo_allOppo), axis=1))
        	plt.legend(['perfo', 'fitness', 'ice_fitness', 'opponent_fitness', 'perfo_allOppo'], loc='lower right')
        else:
        	plt.legend(['perfo', 'fitness', 'ice_fitness', 'opponent_fitness'], loc='lower right')
    else:
        plt.legend(['perfo', 'fitness', 'ice_fitness'], loc='lower right')
    plt.ylabel('Average performance of population')
    plt.xlabel('Evaluations') 
    plt.fill_between(range(max_gen), np.amin(performance, axis=1), np.amax(performance, axis=1), color='pink', alpha='1')
    if args.save_reward:
        if args.population_play:
        	if not os.path.exists('PopulationPlay/'+args.name):
        		os.makedirs('PopulationPlay/'+args.name)
        	plt.savefig('PopulationPlay/'+args.name+'/evaluation_'+str(args.mutation)+'.svg', format='svg')
        	file = open('PopulationPlay/'+args.name+'/message.txt','w') 
        else:
        	if not os.path.exists('test/'+args.name):
        		os.makedirs('test/'+args.name)
        	plt.savefig('test/'+args.name+'/evaluation_'+str(args.mutation)+'.svg', format='svg')
        	file = open('test/'+args.name+'/message.txt','w') 
        file.write(args.message) 
        file.close() 
    plt.show()

    [print(x.fitness, x.identity) for x in population]
    #print(population_fitness)
    print("SOLVED!")
    fname = args.save + "_EVAL%d" % evals
    child.save(fname)
    print("saved locally")

    
