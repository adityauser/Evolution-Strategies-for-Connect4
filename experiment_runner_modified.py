import gc
import numpy
import operator
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
import pickle

from sklearn.cluster import KMeans


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
parser.add_argument("--layers", help="number of ann hidden layers",default=6)
parser.add_argument("--activation",help="ann activation function",default="relu")
parser.add_argument("--max_gen",help="total number of generation",default=100)
parser.add_argument("--domain",help="Experimental domain", default="connect_four",choices=['connect_four_CNN','breadcrumb_maze', 'connect_four'])

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
params = {'size':int(args.hidden),'af':args.activation,'layers':int(args.layers),'init':args.init} 

#define dictionary describing domain
domain = {'name':args.domain, 'incentive':'fitness'}

#initialize domain
evolution_domain.setup(domain,params)


def find_fitness(agent):
	''' Wrapper over fitness object over agent class

	Args:
		agent: the individual
	Returns:
		0 if random agent else the fitness
	'''

	if agent==None:
        	return 0
	else :
        	return agent.fitness


#Maze rendering call (visualize behavior of population)
def render_maze(pop):
    pass

#takes population and gives out a child
def get_child(population, mutation_mag, evals, greedy_select=5):
	''' Select the parent, mutate the parent to make a child.
	Agrs:
		population: colletion of agent from which parent need to be sample
		mutation_mag: mutation magnitude
		evals: current generation number
		greedy_select: degree of exploitation vs exploration
	Returns: 
		child: agent made after the mutation
	'''
	parents = random.sample(population, greedy_select)
	parent = reduce(lambda x, y: x if x.fitness > y.fitness else y, parents)
	child = parent.copy((evals, parent.identity[1]))
	child.mutate(mutation=args.mutation, mag=mutation_mag, fitness=parent.fitness)
	return child

#evaluation
def evaluate(individual, test_opponents, **kwargs):
	''' Individual plays against opponents
	Args:
		individual: individual whose evaluation needs to be done
		test_opponents: opponents against which the individual plays
	'''
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

    #performance record of population
    performance = []
    #performance record of elite agent
    performance_elite = []

    #average population fitness
    population_fitness = []

    #average elite fitness
    population_elite_fitness = []
    elite = [None]*10

    #grab population size
    psize = int(args.pop_size)

    # To take account of each family
    '''
    individual_hierarchy = {}
    for i in range(psize):
    	individual_hierarchy[i+1] = []
    '''

    #number of trials in each tournament
    trials = 30

    # flag: Do average fitness of agent in all generations
    reset_score = False

    #cluster model
    cluster_model = KMeans(n_clusters=int(trials/6), n_jobs=-1)

    # Opponents for the tournament
    opponents = []
    #performance record of all opponents
    perfo_allOppo = []
    #average opponent fitness
    opponent_fitness = []
    # number of random opponents in the team
    random_opponent = 10
    for _ in range(trials):
        opponents.append(None)
    # all the oppoenets we have
    all_opponents = copy.deepcopy(opponents)

    #broken
    broken = False

    #initialize population
    for k in range(psize):

        threads = []
        robot = evolution_domain.individual((1, k+1))
        # individual_hierarchy[k+1].append(robot)
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

    # novelty vertors for all individuals
    X = [indv.get_novelty_vector() for indv in population]
    cluster_model.fit(X)
    
    print("population_fitness ", np.mean([x.fitness for x in population]))

    # Selecting the elite agents from the population
    values = [[copy.deepcopy(ind), ind.fitness] for ind in population]
    values.sort(key = operator.itemgetter(1))
    for x in enumerate(values[-(1+len(elite)):-1]):
        elite[x[0]] =  x[1][0]

    print("population_elite_fitness ", np.mean([x.fitness for x in elite]))
    
    #Dominance: Which geneome is the most dominant in the population
    dominance = np.ones(psize)

    #solution flag
    solved = False

    #broken flag
    broken = False

    #number of generations
    evals = 0

    # population play flag
    population_play = args.population_play

    #number of tournaments in each generation
    gen_length = int(args.gen_length)
    #parse max evaluations
    max_gen = int(args.max_gen)

    #degree of exploitation vs exploration
    greedy_kill = 5
    greedy_select = 5

    #parse mutation intensity parameter
    mutation_mag = float(args.mutation_mag)

    #flags 
    update_eval = 0
    update_novelty = 0

    # Creating folder
    if args.save_reward:
        if args.population_play:
            if not os.path.exists('PopulationPlay/'+args.name):
                os.makedirs('PopulationPlay/'+args.name)
        else:
            if not os.path.exists('test/'+args.name):
                os.makedirs('test/'+args.name)

    #evolutionary loop
    while evals < max_gen:        

        print('Evaluation no. ' ,evals)

        broken = True

        evals += 1
        if evals % 2 == 0:
            gc.collect()
        # number of tournament played
        tournament_played = 0

        #initialize empty population for childs
        young_ones = []


        # Evaluate population performance against random agents
        test = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
        	for test_agent in population:
        		future = executor.submit(evaluate, individual=test_agent, test_opponents=[None]*30, push_all=args.state_archive, test=True)
        		r, _ = future.result()
        		test.append(r/len(opponents))
                
        # To check everything is working fine
        performance.append(test)
        if test[-1]<-1:
            raise('check2')

        # Evaluate elite performance against random agents
        test = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for test_agent in elite:
                future = executor.submit(evaluate, individual=test_agent, test_opponents=[None]*30, push_all=args.state_archive, test=True)
                r, _ = future.result()
                test.append(r/len(opponents))

        # To check everything is working fine                
        performance_elite.append(test)
        if test[-1]<-1:
            raise('check5')
        
        # Evaluate population performance against all_opponents, basicly this is the fitness in case of population_play.
        test = []
        if args.population_play and not args.fast:
        	with concurrent.futures.ThreadPoolExecutor() as executor:
        		for test_agent in population:
        			future = executor.submit(evaluate, individual=test_agent, test_opponents=all_opponents, push_all=args.state_archive, test=True)
        			r, _ = future.result()
        			test.append(r/len(all_opponents))
        		perfo_allOppo.append(test)
        		if test[-1]<-1:
        			raise('check3')            

        
        print (bcolors.WARNING)

        if args.population_play:
            print('opponent_fitness ', np.mean([find_fitness(x) for x in all_opponents]))
            opponent_fitness.append(np.mean([find_fitness(x) for x in all_opponents]))

        #Get the average fitness of population
        print("population_fitness ", np.mean([x.fitness for x in population]))
        print("population_elite_fitness ", np.mean([x.fitness for x in elite]))
        print (bcolors.ENDC)
        population_fitness.append(np.mean([x.fitness for x in population]))
        population_elite_fitness.append(np.mean([x.fitness for x in elite]))
        
        #Re-evaluate the fitness of every agent in the population
        #[print(x.fitness, x.identity) for x in population]
        print (bcolors.WARNING)
        print('Recalculating the fitness')
        print (bcolors.ENDC)
        for parent in population:
            if reset_score:
            		parent.matchs_played = 0
            		parent.net_reward = 0
            		parent.fitness = 0
            	
            threads = []
            for against in all_opponents:
            	thread = threading.Thread(target=parent.map, args=(args.state_archive, against))
            	thread.start()
            	threads.append(thread)
            for thread in threads:
            	thread.join()
           
            if parent.fitness<-1:
                print(broken, parent.fitness,"check4")
                raise('check4')
        print (bcolors.WARNING)
        print("re_population_fitness ", np.mean([x.fitness for x in population]))
        print("population_elite_fitness ", np.mean([x.fitness for x in elite]))
        print (bcolors.ENDC)

        
        #creation of new off springs, tournament selection and evalute in domain
        with concurrent.futures.ThreadPoolExecutor() as executor:
        	while tournament_played < gen_length:
        		tournament_played += 1
        		print('Generation no. {} and Tournament no. {}'.format(evals, tournament_played))
        		future = executor.submit(get_child, population=population, mutation_mag=mutation_mag, evals=evals, greedy_select=greedy_select)
        		child = future.result()
        		#individual_hierarchy[child.identity[1]].append(child)
        		future = executor.submit(evaluate, individual=child, test_opponents=all_opponents, push_all=args.state_archive)
        		trial_reward, terminal_state = future.result()
        		if trial_reward/len(all_opponents)<-1:
        			raise('check1')
        		print('Child average reward ' , trial_reward/len(all_opponents))
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
        # Finding new elite agents
        values = [[copy.deepcopy(ind), ind.fitness] for ind in population]
        values.sort(key = operator.itemgetter(1))
        for x in enumerate(values[-(1+len(elite)):-1]):
            elite[x[0]] =  x[1][0]

        #Add some new members in the oppents team.
        if args.population_play and population_fitness[-1]>0.80:
            update_eval = evals
            update_novelty = evals

            # Making clusters of the population
            X = [indv.get_novelty_vector() for indv in population]
            yhat = cluster_model.predict(X)
            pop_np = np.array(population)

            # Add opponent to the opponents team
            for cluster in range(int(trials/6)):
            	pop_cluster = list(pop_np[np.where(yhat == cluster)])
            	print(len(pop_cluster))
            	if len(pop_cluster)>0:
            		new_ops = random.sample(pop_cluster, int(np.ceil(len(pop_cluster)/5)))
            	else:
            		new_ops = random.sample(population, int(psize/25))
            	new_op = copy.deepcopy(reduce(lambda x, y: x if x.fitness > y.fitness else y, new_ops))
            	# Replace opponent
            	opponents[int(np.random.rand()*(len(opponents)-random_opponent))] = new_op
            	# Add opponent
            	all_opponents.append(new_op)
            del pop_np
            '''
            for _ in range(int(trials/6)):
                new_ops = random.sample(population, int(psize/25))
                new_op = copy.deepcopy(reduce(lambda x, y: x if x.fitness > y.fitness else y, new_ops))
                opponents[int(np.random.rand()*(len(opponents)-random_opponent))] = new_op
                all_opponents.append(new_op)
            '''
        # Train the KNN agent
        if (evals-update_novelty)%3==0:
        	#cluster_model = KMeans(n_clusters=int(trials/6))
            X = [indv.get_novelty_vector() for indv in population]
            cluster_model.fit(X)

        # Geometric decrease for mutation magnitude and increase of numbers of trails
        if (evals-update_eval)%20==0:
            update_eval = evals
            #trials = trials*2 # exponential trials are currently not for population play
            #mutation_mag = mutation_mag*0.95
            print(colored('Updated the trials to', 'red'), trials)
            print(colored('Updated the mutation_mag to', 'red'), mutation_mag)

            #For population play
            for _ in range(trials - len(all_opponents)):
            	all_opponents.append(None)
				
		# Save the data
        if args.population_play:
            file = open('PopulationPlay/'+args.name+'/dominance.txt','a') 
            #dbfile = open('PopulationPlay/'+args.name+'/elite_agent', 'ab') 
            np.save('PopulationPlay/'+args.name+'/performance', performance)
            np.save('PopulationPlay/'+args.name+'/performance_elite', performance_elite)
            np.save('PopulationPlay/'+args.name+'/population_fitness', population_fitness)
            np.save('PopulationPlay/'+args.name+'/population_elite_fitness', population_elite_fitness)
            np.save('PopulationPlay/'+args.name+'/opponent_fitness', opponent_fitness)
            if not args.fast:
                np.save('PopulationPlay/'+args.name+'/performanceAgaisntAllOppo', np.array(perfo_allOppo))
        else:
            np.save('test/'+args.name+'/performance', performance)
            np.save('test/'+args.name+'/performance_elite', performance_elite)
            np.save('test/'+args.name+'/population_fitness', population_fitness)
            np.save('test/'+args.name+'/population_elite_fitness', population_elite_fitness)
           # dbfile = open('test/'+args.name+'/elite_agent', 'ab') 
            file = open('test/'+args.name+'/dominance.txt','a') 
        [file.write(str(x) + " ") for x in dominance]
        file.write('\n')
        file.close() 
        
        #pickle.dump(elite, dbfile)
       # dbfile.close() 

        

    #save elite agent
    if args.save_reward:
        for i, agent in enumerate(elite):
            if args.population_play:
                agent.save('PopulationPlay/'+args.name+'/elite_agent'+str(i))
            else:
                agent.save('test/'+args.name+'/elite_agent'+str(i))


    #Print the losing games
    for i in range(5):
        #good_agents = random.sample(population, int(len(population)/10))
        #good_agent = reduce(lambda x, y: x if x.fitness > y.fitness else y, good_agents)
        good_agent = elite[i]
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
    plt.plot(range(max_gen), np.mean(performance, axis=1))
    plt.plot(range(max_gen), np.mean(performance_elite, axis=1))
    plt.plot(range(max_gen), population_fitness)
    plt.plot(range(max_gen), population_elite_fitness)
    if args.population_play:
        plt.plot(range(max_gen), opponent_fitness)
        if not args.fast:
        	plt.plot(range(max_gen), np.mean(np.array(perfo_allOppo), axis=1))
        	plt.legend(['test', 'test_elite', 'avg_pop_fitness', 'avg_elite_fitness', 'avg_opponent_fitness', 'test_against_allopponent'], loc='lower right')
        else:
        	plt.legend(['test', 'test_elite', 'avg_pop_fitness', 'avg_elite_fitness', 'avg_opponent_fitness'], loc='lower right')
    else:
        plt.legend(['test', 'test_elite', 'avg_pop_fitness', 'avg_elite_fitness'], loc='lower right')
    plt.ylabel('Average performance of population')
    plt.xlabel('Evaluations') 
    plt.fill_between(range(max_gen), np.amin(performance, axis=1), np.amax(performance, axis=1), color='pink', alpha='1')
    if args.save_reward:
        if args.population_play:        	
        	plt.savefig('PopulationPlay/'+args.name+'/evaluation_'+str(args.mutation)+'.svg', format='svg')
        	file = open('PopulationPlay/'+args.name+'/message.txt','w') 
        else:        	
        	plt.savefig('test/'+args.name+'/evaluation_'+str(args.mutation)+'.svg', format='svg')
        	file = open('test/'+args.name+'/message.txt','w') 
        file.write(args.message) 
        file.close() 
    plt.show()
    '''
    for i in range(psize):
    	family_performance = [x.fitness for x in individual_hierarchy[i+1]]
    	plt.plot(range(len(family_performance)), family_performance)
    	plt.ylabel('Fitness of individual')
    	plt.xlabel('Generation Number') 
    	if args.population_play:
    		plt.savefig('PopulationPlay/'+args.name+'/family_'+str(i+1)+'.svg', format='svg')
    	else:
    		plt.savefig('test/'+args.name+'/family_'+str(i+1)+'.svg', format='svg')
    	plt.close()
    '''
    [print(x.fitness, x.identity) for x in population]
    #print(population_fitness)
    print("SOLVED!")
    fname = args.save + "_EVAL%d" % evals
    child.save(fname)
    print("saved locally")
