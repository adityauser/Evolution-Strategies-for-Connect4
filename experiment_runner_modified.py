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
parser.add_argument("--population_play", help="play matches against population", action="store_true")
parser.add_argument("--decay", help="population fitness reduced in 100 tournaments", action="store_true")
parser.add_argument("--init", help="init rule", default="xavier")
parser.add_argument("--celltype", help="recurrent cell type",default="lstm",choices=['lstm','gru','rnn'])
parser.add_argument("--layers", help="number of ann hidden layers",default=6)
parser.add_argument("--activation",help="ann activation function",default="relu")
parser.add_argument("--max_evals",help="total number of evaluations",default=100000)
parser.add_argument("--domain",help="Experimental domain", default="connect_four",choices=['connect_four_CNN','breadcrumb_maze', 'connect_four'])
#Ignore this argument
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

    #grab population size
    psize = int(args.pop_size)

    #number of trials
    trials = 30

    # Opponents for the tournament
    opponents = []
    for _ in range(trials):
        opponents.append(None)

    #broken
    broken = False

    #initialize population
    for k in range(psize):
        robot = evolution_domain.individual((1, k+1))

        #initialize random parameter vector
        robot.init_rand()

        #evaluate in domain
        for _ in range(trials):
            terminal, broken = robot.map()
            while broken:
                robot.init_rand()
                terminal, broken = robot.map()
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

    #parse max evaluations
    max_evals = int(args.max_evals)

    #tournament size
    greedy_kill = 2
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
        if  not args.population_play:
            while broken:
                trial_reward = 0
                parents = random.sample(population, greedy_select)
                parent = reduce(lambda x, y: x if x.fitness > y.fitness else y, parents)
                child = parent.copy((evals, parent.identity[1]))
                child.mutate(mutation=args.mutation,mag=mutation_mag) 
                for _ in range(trials):
                    terminal_state, broken = child.map()
                    if broken:
                        break
                    trial_reward+=child.reward

                
        else:
            while broken:
                trial_reward = 0
                parents = random.sample(population, greedy_select)
                parent = reduce(lambda x, y: x if x.fitness > y.fitness else y, parents)
                child = parent.copy((evals, parent.identity[1]))
                child.mutate(mutation=args.mutation,mag=mutation_mag) 
                for against in opponents:
                    terminal_state, broken = child.map(against = against)
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
        if ((evals - psize)  % 100 == 0):
            test = []
            for test_agent in population:
                r = 0
              #  print(id(test_agent.genome))
                for _ in range(trials):
                    reward, terminal_state, broken = test_agent.map(test = True)
                    while broken:
                        reward, terminal_state, broken = test_agent.map(test = True)
                    r+=reward
                test.append(r/trials)
            performance.append(test)
            if test[-1]<=-1:
                raise('check2')

        # Evaluate population performance against opponents, basicly this is the fitness in case of population_play.
        if args.population_play and ((evals - psize)  % 100 == 0):
            test = []
            for test_agent in population:
                r = 0
              #  print(id(test_agent.genome))
                for against in opponents:
                    reward, terminal_state, broken = test_agent.map(against=against, test = True)
                    while broken:
                        reward, terminal_state, broken = test_agent.map(against=against, test = True)
                    r+=reward
                test.append(r/trials)
            ice.append(test)
            if test[-1]<=-1:
                raise('check3')

        #Get the average fitness of population
        if ((evals - psize)  % 100 == 0):

            print (bcolors.WARNING)

            if args.population_play:
                print('opponent_fitness ', np.mean([find_fitness(x) for x in opponents]))

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
        if (evals-psize)%100==0:
            print (bcolors.WARNING)
            print('Recalculating the fitness')
            print (bcolors.ENDC)
            for parent in population:
                broken = True
                while broken:
                    parent.matchs_played = 0
                    parent.net_reward = 0
                    parent.fitness = 0
                    for against in opponents:
                        terminal_state, broken = parent.map(against=against)
                        if broken:
                            break
                if parent.fitness<=-1:
                    print(broken,parent.fitness,"check4")
                    raise('check4')
            print (bcolors.WARNING)
            print("ice_population_fitness ", np.mean([x.fitness for x in population]))
            print (bcolors.ENDC)
            ice_population_fitness.append(np.mean([x.fitness for x in population]))

            #Add some new members in the oppents team.
            if args.population_play:
                for _ in range(int(trials/4)):
                    new_ops = random.sample(population, int(psize/15))
                    new_op = reduce(lambda x, y: x if x.fitness > y.fitness else y, parents)
                    opponents[int(np.random.rand()*len(opponents))] = new_op
                    


        #remove individual from the pop using a tournament of same size
        to_kill = random.sample(population, greedy_kill)
        to_kill = reduce(lambda x, y: x if x.fitness < y.fitness else y,
                         parents)
        population.remove(to_kill)
        to_kill.kill()
        del to_kill
        del terminal_state


    # Plotting different varients of fitness.
    performance = np.array(performance)
    ice = np.array(ice)
    plt.plot(range(int(args.pop_size)+100, evals, 100), np.mean(performance, axis=1))
    plt.plot(range(int(args.pop_size)+100, evals, 100), population_fitness)
    plt.plot(range(int(args.pop_size)+100, evals, 100), ice_population_fitness)
    if args.population_play:
        plt.plot(range(int(args.pop_size)+100, evals, 100), np.mean(ice, axis=1))
        plt.legend(['perfo', 'ice_perfo', 'fitness', 'ice_fitness'], loc='upper left')
    else:
        plt.legend(['perfo', 'fitness', 'ice_fitness'], loc='upper left')
    plt.ylabel('Average performance of population')
    plt.xlabel('Evaluations') 
    plt.fill_between(range(int(args.pop_size)+100, evals, 100), np.amin(performance, axis=1), np.amax(performance, axis=1), color='pink', alpha='1')
    if args.save_reward:
        if args.population_play:
            plt.savefig('PopulationPlay/'+args.name+'evaluation_'+str(args.mutation)+'_population_play'+'.png')
        else:
            plt.savefig('test/'+args.name+'evaluation_'+str(args.mutation)+'.png')
    plt.show()


    print("SOLVED!")
    fname = args.save + "_EVAL%d" % evals
    child.save(fname)
    print("saved locally")

    
