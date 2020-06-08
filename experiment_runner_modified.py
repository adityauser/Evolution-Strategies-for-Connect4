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
parser.add_argument("--celltype", help="recurrent cell type",default="lstm",choices=['lstm','gru','rnn'])
parser.add_argument("--layers", help="number of ann hidden layers",default=6)
parser.add_argument("--activation",help="ann activation function",default="relu")
parser.add_argument("--max_evals",help="total number of evaluations",default=100000)
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


if (__name__ == '__main__'):

    #initialize empty population
    population = []

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
    trials = 30

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
        robot = evolution_domain.individual((1, k+1))

        #initialize random parameter vector
        robot.init_rand()

        #evaluate in domain
        for _ in range(trials):
            terminal, broken = robot.map(push_all=args.state_archive)
            while broken:
                robot.init_rand()
                terminal, broken = robot.map(push_all=args.state_archive)
        robot.parent = None

        #add to population
        population.append(robot)

    print("population_fitness ", np.mean([x.fitness for x in population]))

    #solution flag
    solved = False

    #broken
    broken = False

    #we spent evals looking at the population
    evals = psize

    population_play = args.population_play

    gen_length = int(args.gen_length)

    #parse max evaluations
    max_evals = int(args.max_evals)

    #tournament size
    greedy_kill = 5
    greedy_select = 5

    #parse mutation intensity parameter
    mutation_mag = float(args.mutation_mag)

    #evolutionary loop
    while evals < max_evals:        

        print('Evaluation no. ' ,evals)

        trial_reward = 0

        broken = True

        evals += 1

        if evals % 50 == 0:
            gc.collect()

        #Aditya: You can ignore this if component
        if evals % 500 == 0:
            #logging progress to text file
            print("saving out...",evals)
            f = "%s.progress"%args.save

            outline = str(evals)+" "+str(best_fit)

            #write out addtl info if we have it
    
            open(f,"a+").write(outline+"\n")
            f2 = "%s_best.npy" % args.save
            best_ind.save(f2)

        #tournament selection and evalute in domain
        while broken:
            trial_reward = 0
            parents = random.sample(population, greedy_select)
            parent = reduce(lambda x, y: x if x.fitness > y.fitness else y, parents)
            child = parent.copy((evals, parent.identity[1]))
            child.mutate(mutation=args.mutation, mag=mutation_mag, fitness=parent.fitness) 
            for against in all_opponents:
                terminal_state, broken = child.map(push_all=args.state_archive, against=against)
                if broken:
                    break
                trial_reward+=child.reward

        population.append(child)

        if trial_reward/trials<=-1:
            raise('check1')
        
        print('Child average reward ' , trial_reward/trials)
        print('child_fitness ', child.fitness)

        #Aditya: You can ignore this if component
        if child.reward > best_fit:
            best_fit = child.reward
            best_ind = child.copy((evals, parent.identity[1]))

            print (bcolors.WARNING)
            print ("new best fit: ", best_fit)
            print (bcolors.ENDC)

        if do_display:
            print(terminal_state)

        if (child.solution()):
         #   solved = True
            pass

        # Evaluate population performance against random agents
        if ((evals - psize)  % gen_length == 0):
            test = []
            for test_agent in population:
                r = 0
              #  print(id(test_agent.genome))
                for _ in range(len(opponents)):
                    reward, terminal_state, broken, the_game = test_agent.map(test = True)
                    while broken:
                        reward, terminal_state, broken, the_game = test_agent.map(test = True)
                    r+=reward
                test.append(r/len(opponents))
            performance.append(test)
            if test[-1]<=-1:
                raise('check2')

        # Evaluate population performance against all_opponents, basicly this is the fitness in case of population_play.

        if args.population_play and ((evals - psize)  % gen_length == 0) and not args.fast:
            test = []
            for test_agent in population:
                r = 0
              #  print(id(test_agent.genome))
                for against in all_opponents:
                    reward, terminal_state, broken, the_game = test_agent.map(against=against, test=True)
                    r+=reward
                test.append(r/len(all_opponents))
            perfo_allOppo.append(test)
            if test[-1]<=-1:
                raise('check3')

        #Get the average fitness of population
        if ((evals - psize)  % gen_length == 0):

            print (bcolors.WARNING)

            if args.population_play:
                print('opponent_fitness ', np.mean([find_fitness(x) for x in opponents]))
                opponent_fitness.append(np.mean([find_fitness(x) for x in opponents]))

            print("population_fitness ", np.mean([x.fitness for x in population]))
            print (bcolors.ENDC)
            population_fitness.append(np.mean([x.fitness for x in population]))

            #Aditya: You can ignore this if component
            idx = 0
            save_all = False
            if save_all:
                for k in population:
                    k.save("%s/child%d" % (args.save,idx))
                    idx += 1

        #Re-evaluate the fitness of every agent in the population
        if (evals-psize)%gen_length==0:
            print (bcolors.WARNING)
            print('Recalculating the fitness')
            print (bcolors.ENDC)
            for parent in population:
                broken = True
                while broken:
                    #parent.matchs_played = 0
                    #parent.net_reward = 0
                    #parent.fitness = 0
                    for against in opponents:
                        terminal_state, broken = parent.map(push_all=args.state_archive, against=against)
                        if broken:
                            break
                if parent.fitness<=-1:
                    print(broken, parent.fitness,"check4")
                    raise('check4')
            print (bcolors.WARNING)
            print("ice_population_fitness ", np.mean([x.fitness for x in population]))
            print (bcolors.ENDC)
            ice_population_fitness.append(np.mean([x.fitness for x in population]))

            
            #Add some new members in the oppents team.
            if args.population_play and ice_population_fitness[-1]>0.80:
                for _ in range(int(trials/6)):
                    new_ops = random.sample(population, int(psize/25))
                    new_op = reduce(lambda x, y: x if x.fitness > y.fitness else y, parents)
                    opponents[int(np.random.rand()*(len(opponents)-random_opponent))] = new_op
                    all_opponents.append(new_op)

            #This is reductant
            if best_pop_fitness<ice_population_fitness[-1]:
            	best_pop_fitness = ice_population_fitness[-1]
            	best_eval = evals

            if (evals-psize)%(gen_length*20)==0:
            	trials = trials*2 # exponential trials are currently not for population play
            	mutation_mag = mutation_mag*0.80
            	print(colored('Updated the trials to', 'red'), trials)
            	print(colored('Updated the mutation_mag to', 'red'), mutation_mag)

            	#This is reductant
            	#To make sure this if statement does not runs for next 100 evals
            	best_eval = evals
            	
            	#For population play
            	for _ in range(trials - len(all_opponents)):
            		all_opponents.append(None)
				

        #remove individual from the pop using a tournament of same size
        to_kill = random.sample(population, greedy_kill)
        to_kill = reduce(lambda x, y: x if x.fitness < y.fitness else y,
                         parents)
        population.remove(to_kill)
        to_kill.kill()
        del to_kill
        del terminal_state

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
    plt.plot(range(int(args.pop_size)+gen_length, evals, gen_length), np.mean(performance, axis=1))
    plt.plot(range(int(args.pop_size)+gen_length, evals, gen_length), population_fitness)
    plt.plot(range(int(args.pop_size)+gen_length, evals, gen_length), ice_population_fitness)
    if args.population_play:
        plt.plot(range(int(args.pop_size)+gen_length, evals, gen_length), opponent_fitness)
        if not args.fast:
        	plt.plot(range(int(args.pop_size)+gen_length, evals, gen_length), np.mean(np.array(perfo_allOppo), axis=1))
        	plt.legend(['perfo', 'fitness', 'ice_fitness', 'opponent_fitness', 'perfo_allOppo'], loc='lower right')
        else:
        	plt.legend(['perfo', 'fitness', 'ice_fitness', 'opponent_fitness'], loc='lower right')
    else:
        plt.legend(['perfo', 'fitness', 'ice_fitness'], loc='lower right')
    plt.ylabel('Average performance of population')
    plt.xlabel('Evaluations') 
    plt.fill_between(range(int(args.pop_size)+gen_length, evals, gen_length), np.amin(performance, axis=1), np.amax(performance, axis=1), color='pink', alpha='1')
    if args.save_reward:
        if args.population_play:
            plt.savefig('PopulationPlay/'+args.name+'evaluation_'+str(args.mutation)+'_population_play'+'.svg', format='svg')
        else:
            plt.savefig('test/'+args.name+'evaluation_'+str(args.mutation)+'.svg', format='svg')
    plt.show()

    [print(x.net_reward, x.matchs_played) for x in population]

    #print(population_fitness)
    print("SOLVED!")
    fname = args.save + "_EVAL%d" % evals
    child.save(fname)
    print("saved locally")

    
