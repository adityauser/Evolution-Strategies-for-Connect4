#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from functools import partial
import matplotlib.pyplot as plt

import os
import argparse


parser = argparse.ArgumentParser()
#all command line options

parser.add_argument("--mutation_mag", help="magnitude of mutation operator",default='0.01')
parser.add_argument('--pop_size', help="population size",default='200')
parser.add_argument("--hidden", help="number of hidden units per ann layer", default='15')
parser.add_argument("--name", help="path where plot needs to be saved", default="")
parser.add_argument("--save_result", help="save the plot of avg reward", action="store_true")
parser.add_argument("--noise", help="add noise to the output", action="store_true")
parser.add_argument("--trials", help="number of trials in a generation", default='100')
parser.add_argument("--message", help="description of the experiment", default="")
parser.add_argument("--layers", help="number of ann hidden layers",default='6')
parser.add_argument("--max_gen",help="total number of generation",default='100')
parser.add_argument("--arms",help="total number of arms",default='10')


#Parse arguments
args = parser.parse_args()


if args.save_result:
    try:
        os.makedirs('results/'+args.layers + 'x' + args.hidden + 'mag' + args.mutation_mag + 'arms' + args.arms + 'psize' + args.pop_size + 'trials' + args.trials + args.name)
    except:
        pass


class Enviroment:
    def __init__(self, is_bernoulli, no_arms=10):
        self.no_arms = no_arms
        
        self.pull_arm = self.pull_arm_bernaulli if is_bernoulli else self.pull_arm_gaussion
            
        self.arms = {}
        for arm in range(self.no_arms):
            self.arms[arm] = np.random.rand()
        
        self.best_arm = np.argmax([[self.arms[arm] for arm in range(self.no_arms)]])
        
    def reset(self):
        for arm in range(self.no_arms):
            self.arms[arm] = np.random.rand()
        self.best_arm = np.argmax([[self.arms[arm] for arm in range(self.no_arms)]])
        
    def pull_arm_bernaulli(self, arm):
        return 1 if self.arms[arm] > np.random.rand() else 0
    
    def pull_arm_gaussion(self, arm):
        return np.random.normal(self.arms[arm], scale=0.1)
    
        


# In[2]:


class AgentUCB:
    def __init__(self, no_arms=10):
        self.no_arms = no_arms
        self.info = {}
        for arm in range(self.no_arms):
            self.info[arm] = {}
            self.info[arm]['likely'] = 0
            self.info[arm]['no_visited'] = 0
            self.info[arm]['is_visited'] = False
            self.info[arm]['value'] = 0
        self.round = 0
        self.score = 0
        
    def pull(self):
        self.round += 1
        
        for arm in range(self.no_arms):
            if not self.info[arm]['is_visited']:
                self.info[arm]['is_visited'] = True
                self.info[arm]['no_visited'] += 1
                return arm
            
        arm = np.argmax(np.asarray([self.info[arm]['likely'] for arm in range(self.no_arms)]))
        self.info[arm]['no_visited'] += 1
        
        return arm
    
    def reset(self):
        self.info = {}
        for arm in range(self.no_arms):
            self.info[arm] = {}
            self.info[arm]['likely'] = 0
            self.info[arm]['no_visited'] = 0
            self.info[arm]['is_visited'] = False
            self.info[arm]['value'] = 0
        self.round = 0    
    
    def update(self, action, reward):
        
        self.info[action]['value'] += (reward-self.info[action]['value'])/self.info[action]['no_visited']
        
        if all([self.info[arm]['is_visited'] for arm in range(self.no_arms)]):
            for arm in range(self.no_arms):
                self.info[arm]['likely'] = self.info[arm]['value'] + np.sqrt((2*np.log(self.round))/self.info[arm]['no_visited'])
                
            
        


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable,grad

import copy
import operator
import collections
from sklearn.preprocessing import normalize

state_archive = collections.deque([], 100)

class Model(torch.nn.Module):
    def __init__(self, num_inputs, action_space,settings={}):
        super(Model, self).__init__()

        #should hidden units have biases?
        _b = True
        hid_layers = int(args.layers) #how many hidden layers
        sz = int(args.hidden) #how many neurons per layer

        num_outputs = action_space

        #overwritable defaults
        af = nn.ReLU 
        oaf = nn.Sigmoid
        

        afunc = {'relu':nn.ReLU,'tanh':nn.Tanh,'linear':lambda : linear,'sigmoid':nn.Sigmoid}
        
        self.stddev = 0.1
        self._af = af
        self.af = af()
        self.sigmoid = oaf()
        self.softmax = nn.Softmax(dim=1)

        #first fully-connected layer changing from input-size representation to hidden-size representation
        self.fc1 = nn.Linear(num_inputs, sz, bias=_b) 

        self.hidden_layers = []

        #create all the hidden layers
        for x in range(hid_layers):
            self.hidden_layers.append(nn.Linear(sz, sz, bias=True))
            self.add_module("hl%d"%x,self.hidden_layers[-1])

        #create the hidden -> output layer
        self.fc_out = nn.Linear(sz, num_outputs, bias=_b)

        self.train()

    def forward(self, inputs, intermediate=None, debug=False):

        x = inputs
        x = torch.cat(x, axis=1)
        #fully connection input -> hidden
        x = self.fc1(x.float())
            
        x = self._af()(x)
        
        #propagate signal through hidden layers
        for idx,layer in enumerate(self.hidden_layers):

            #do fc computation
            x = layer(x)

            #run it through activation function
            x = self._af()(x)

            if intermediate == idx:
                cached = x
        
        #output layer
        x = self.fc_out(x)
        #Add noise
        if args.noise:            
            x = x+torch.autograd.Variable(torch.randn(x.size()).cpu() * self.stddev)
        #softmax
        x = self.softmax(x)

        if intermediate!=None:
            return x,cached

        return x
    
    #function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for param in self.parameters():
            sz = param.data.numpy().flatten().shape[0]
            pvec[count:count + sz] = param.data.numpy().flatten()
            count += sz
        return pvec.copy()
    
    #function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        tot_size = self.count_parameters()
        count = 0

        for param in self.parameters():
            sz = param.cpu().data.numpy().flatten().shape[0]
            raw = pvec[count:count + sz]
            reshaped = raw.reshape(param.cpu().data.numpy().shape)
            param.data = torch.from_numpy(reshaped).float()
            count += sz

        return pvec

    #count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for param in self.parameters():
            #print param.data.numpy().shape
            count += param.cpu().data.numpy().flatten().shape[0]
        return count

class Individual:
    def __init__(self, no_arms=10, mag=0.1, trials=50):
        self.no_arms = no_arms
        self.mag = mag
        self.trials = trials
        self.net_rwd = 0
        self.score = 0
        self.fitness = 0
        self.matches = 0
        self.played = False
        self.buffer = []
        self.model = Model(no_arms*2, no_arms)
        self.info = {}
        for arm in range(self.no_arms):
            self.info[arm] = {}
            self.info[arm]['no_visited'] = 0
            self.info[arm]['value'] = 0
        
    def pull(self):
        visits = torch.Tensor([normalize([np.array([self.info[arm]['no_visited'] for arm in range(self.no_arms)])]).ravel()])
        values = torch.Tensor([np.array([self.info[arm]['value'] for arm in range(self.no_arms)])])
        actions = self.model([visits, values])
        self.buffer.append([visits[0].numpy(), values[0].numpy()])
        try:
            #arm = np.random.choice(self.no_arms, 1, p=actions[0].detach().numpy())[0]
            arm  = np.argmax(actions[0].detach().numpy())
        except:
            raise TypeError('actions:', actions)
        self.info[arm]['no_visited'] += 1
        return arm
    
    def reset(self):
        self.buffer = []
        self.info = {}
        for arm in range(self.no_arms):
            self.info[arm] = {}
            self.info[arm]['no_visited'] = 0
            self.info[arm]['value'] = 0

    
    def mutate(self, mag=None, **kwargs):
        if mag==None:
            mag = self.mag
        parameters = self.model.extract_parameters()
        child = Individual(self.no_arms, self.mag, self.trials)
        genome = self.mutate_sm_g('SM-G-SO', parameters, child.model, states=None, mag=mag, **kwargs)
        child.model.inject_parameters(genome.astype(float)) 
        return child
    
    def mutate_sm_g(self, mutation, params, model, verbose=False, states=None, mag=0.1, **kwargs):

        global state_archive

        #inject parameters into current model
        model.inject_parameters(params.copy())

        #if no states passed in, use global state archive
        if states == None:
            states = state_archive

        #sub-sample experiences from parent
     #   print(states[-1], states)
        sz = min(100,len(states))


        np_obs = random.sample(states, sz)
       # print(len(np_obs[0]))
        verification_states = Variable(
            torch.from_numpy(np.array([i[0] for i in np_obs])), requires_grad=False)
        verification_mask = Variable(
            torch.from_numpy(np.array([i[1] for i in np_obs])), requires_grad=False)
       # print("verification_mask", verification_mask.size(), verification_mask.squeeze(1))

        #run experiences through model
        #NOTE: for efficiency, could cache these during actual evalution instead of recalculating
        old_policy = model([verification_states, verification_mask])
        num_outputs = old_policy.size()[1]

        abs_gradient=False 
        avg_over_time=False
        second_order=False

        if mutation.count("ABS")>0:
            abs_gradient=True
            avg_over_time=True
        if mutation.count("SO")>0:
            second_order=True

        #generate normally-distributed perturbation
        delta = np.random.randn(*params.shape).astype(np.float32)*mag

        if second_order:
            if verbose:
                print ('SM-G-SO')
            np_copy = np.array(old_policy.data.numpy(),dtype=np.float32)
            _old_policy_cached = Variable(torch.from_numpy(np_copy), requires_grad=False)
            loss =  ((old_policy-_old_policy_cached)**2).sum(1).mean(0)
            loss_gradient = grad(loss, model.parameters(), create_graph=True)
            flat_gradient = torch.cat([grads.view(-1) for grads in loss_gradient]) #.sum()

            direction = (delta/ np.sqrt((delta**2).sum()))
            direction_t = Variable(torch.from_numpy(direction),requires_grad=False)
            grad_v_prod = (flat_gradient * direction_t).sum()
            second_deriv = torch.autograd.grad(grad_v_prod, model.parameters())
            sensitivity = torch.cat([g.contiguous().view(-1) for g in second_deriv])
            scaling = torch.sqrt(torch.abs(sensitivity).data)

        elif not abs_gradient:
            print ("SM-G-SUM")
            tot_size = model.count_parameters()
            jacobian = torch.zeros(num_outputs, tot_size)
            grad_output = torch.zeros(*old_policy.size())

            for i in range(num_outputs):
                model.zero_grad()   
                grad_output.zero_()
                grad_output[:, i] = 1.0
                old_policy.backward(grad_output, retain_graph=True)
                jacobian[i] = torch.from_numpy(model.extract_grad())

            scaling = torch.sqrt(  (jacobian**2).sum(0) )

        else:
            print ("SM-G-ABS")
            #NOTE: Expensive because quantity doesn't slot naturally into TF/pytorch framework
            tot_size = model.count_parameters()
            jacobian = torch.zeros(num_outputs, tot_size, sz)
            grad_output = torch.zeros([1,num_outputs]) #*old_policy.size())

            for i in range(num_outputs):
                for j in range(sz):
                    old_policy_j = model([verification_states[j:j+1], verification_mask.squeeze(1)[j:j+1]])
                    model.zero_grad()   
                    grad_output.zero_()

                    grad_output[0, i] = 1.0

                    old_policy_j.backward(grad_output, retain_graph=True)
                    jacobian[i,:,j] = torch.from_numpy(model.extract_grad())

            mean_abs_jacobian = torch.abs(jacobian).mean(2)
            scaling = torch.sqrt( (mean_abs_jacobian**2).sum(0))

        scaling = scaling.numpy()

        #Avoid divide by zero error 
        #(intuition: don't change parameter if it doesn't matter)
        scaling[scaling==0]=1.0

        #Avoid straying too far from first-order approx 
        #(intuition: don't let scaling factor become too enormous)
        scaling[scaling<0.01]=0.01

        #rescale perturbation on a per-weight basis
        delta /= scaling

        #delta should be less if fitness is high
        #delta *= -np.log((fitness+1)/2)
        #print("Sum of delta changed from {} to {}".format(sum(delta/np.log((fitness+1)/2)), sum(delta)))

        #generate new perturbation
        new_params = params+delta

        model.inject_parameters(new_params)
        old_policy = old_policy.data.numpy()

        #restrict how far any dimension can vary in one mutational step
        weight_clip = 0.2

        #currently unused: SM-G-*+R (using linesearch to fine-tune)
        mult = 0.05

        if mutation.count("R")>0:
            linesearch=True
            threshold = mag
        else:
            linesearch=False

        if linesearch == False:
            search_rounds = 0
        else:
            search_rounds = 15

        def search_error(x,raw=False):
            final_delta = delta*x
            final_delta = np.clip(final_delta,-weight_clip,weight_clip)
            new_params = params + final_delta
            model.inject_parameters(new_params)

            output = model([verification_states, verification_mask.squeeze(1)]).data.numpy()
            change = np.sqrt(((output - old_policy)**2).sum(1)).mean()

            if raw:
                return change

            return (change-threshold)**2

        if linesearch:
            mult = minimize_scalar(search_error,bounds=(0,0.1,3),tol=(threshold/4)**2,options={'maxiter':search_rounds,'disp':True})
            if verbose:
                print ("linesearch result:",mult)
            chg_amt = mult.x
        else:
            #if not doing linesearch
            #don't change perturbation
            chg_amt = 1.0

        final_delta = delta*chg_amt
        if verbose:
            print ('perturbation max magnitude:',final_delta.max())

        final_delta = np.clip(delta,-weight_clip,weight_clip)
        new_params = params + final_delta

        if verbose:
            print ('max post-perturbation weight magnitude:',abs(new_params).max())

        if verbose:
            print("divergence:", search_error(chg_amt,raw=True))

        diff = np.sqrt(((new_params - params)**2).sum())
        if verbose:
            print("mutation size: ", diff)

        return new_params
    
    def run(self, envs):
        global state_archive
        for env in envs:
            for _ in range(self.trials):
                arm = self.pull()
                r = env.pull_arm(arm)
                self.info[arm]['value'] += (r-self.info[arm]['value'])/self.info[arm]['no_visited']
                self.net_rwd += r 
                self.score += int(arm==env.best_arm)
                self.matches += 1
            state_archive.appendleft(random.choice(self.buffer))
            self.reset()
        self.fitness = self.score/self.matches
        self.matches = 0
        self.score = 0
        
        


if (__name__ == "__main__"):

	'''

	paramaters = {
	    'psize': [10, 25, 50, 100, 200, 500],
	    'mag': [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.2, 0.5],
	    'greedy_kill': [1, 3, 5, 10],
	    'size': [4, 8, 16, 32, 64],
	    'layers': [1, 2, 3, 4, 5],
	    'arms': [2, 4, 8, 16, 32, 64]
	}

	'''

	# In[ ]:

	print('Start:', args)

	import random
	from functools import reduce


	# population size
	psize = int(args.pop_size)
	#mutation magnitude
	mag = float(args.mutation_mag)
	# number of generations
	generations = int(args.max_gen)
	#number of arms
	no_arms = int(args.arms)

	#Number of samples for each bandit problem
	trials = int(args.trials)

	greedy_kill = 5
	greedy_select = 5

	#setting the enviroment
	all_envs = []
	for i in range(psize):
	    envs  = [Enviroment(is_bernoulli=True, no_arms=no_arms) for _ in range(10)]
	    all_envs.append(envs)

	#set of elite agent
	elite = [None]*int(psize/2)


	# Average rewards for ES
	avg_elite_rwd = 0
	avg_rwd = 0
	rwds = []
	elite_rwds = []

	#initilizing population
	population = []
	for _ in range(psize):
	    population.append(Individual(no_arms=no_arms, mag=mag, trials=trials))  
	    
	for gen in range(generations):
	    
	    [indv.run(all_envs[i]) for i, indv in enumerate(population)]
	    
	    #Getting elite agent        
	    for i in range(len(elite)):
	        parents = random.sample(population, greedy_select)
	        parent = reduce(lambda x, y: x if x.fitness > y.fitness else y, parents)
	        elite[i] = copy.deepcopy(parent)
		
	    rwds.append(np.mean([ind.fitness for ind in population]))
	    elite_rwds.append(np.mean([ind.fitness for ind in elite]))
            if gen%50==0:
	        print('Generation: ', gen)
	        print('pop fitness', np.mean([ind.fitness for ind in population]))
	        print('elite fitness', np.mean([ind.fitness for ind in elite]))
	    
	    
	    
	    for j, parent in enumerate(elite):
	        for i in range(int(psize/len(elite))):
	            child = parent.mutate()
	            child.run(all_envs[j+i])
	            population.append(child)

	    
	    #Killing
	    for _ in range(int(psize/len(elite))*len(elite)):
	        to_kill = random.sample(population, greedy_kill)
	        to_kill = reduce(lambda x, y: x if x.fitness < y.fitness else y, to_kill)
	        population.remove(to_kill)
		
	   # [env.reset() for env in envs]
	  #  print('test fitness', np.mean([ind.fitness for ind in population]))
	  #  [ind.run(env) for ind in elite]
	   # print('test_elite fitness', np.mean([ind.fitness for ind in elite]))

	'''
	#for agent in elite:
	#    print(agent.model(Variable(torch.Tensor(np.array([agent.info[arm]['no_visited'] for arm in range(no_arms)])))))
	'''

	#setting UCB agent
	ucb = AgentUCB(no_arms)

	#Average reward of UCB agent
	ucb_rwd = 0
	for env in envs:
	    for _ in range(trials):    
	        arm = ucb.pull()
	        r = env.pull_arm(arm)
	        ucb_rwd += int(env.best_arm==arm)
	        ucb.update(arm, r)
	    ucb.reset()

	ucb_rwd /= len(envs)*trials
		
		

	plt.plot(range(len(rwds)), rwds)
	plt.plot(range(len(rwds)), elite_rwds)
	plt.plot(range(len(rwds)), [ucb_rwd]*len(rwds))



	plt.ylabel('Returns')
	plt.xlabel('Generations') 
	plt.legend(['pop_avg_return', 'elite_avg_return', 'UCB return'], loc='lower right')



	if args.save_result:
            if not args.noise:
                plt.savefig('results/'+args.layers + 'x' + args.hidden + 'mag' + args.mutation_mag + 'arms' + args.arms + 'psize' + args.pop_size + 'trials' + args.trials + args.name + '/training'+'.svg', format='svg')
            else:
                plt.savefig('results/'+args.layers + 'x' + args.hidden + 'mag' + args.mutation_mag + 'arms' + args.arms + 'psize' + args.pop_size + 'trials' + args.trials + args.name + '/noise_training'+'.svg', format='svg')
			

	plt.close()


	# In[ ]:

	'''
	data = np.array([[round(a, 2) ,round(b, 2)] for a,b in zip(rwds, elite_rwds)])
	teams_list = ['Pop return', 'elite_return']
	row_format ="{:>15}" * (len(teams_list) + 1)
	print(row_format.format("", *teams_list))
	for team, row in zip(range(len(rwds)), data):
	    print(row_format.format(team, *row))
	    
	'''

	# In[ ]:


	test_envs  = [Enviroment(is_bernoulli=True, no_arms=no_arms) for _ in range(10)]
	test_pop = []
	test_elite = []
	test_ucb = []
	for env in test_envs:
	    test_rwd = 0
	    for indv in population:
                indv.played = False
                indv.score = 0
                indv.fitness = 0
                indv.run([env])
                test_rwd += indv.fitness
	    test_pop.append(round(test_rwd/psize, 3))
	    
	    test_rwd = 0
	    for indv in elite:
                indv.played = False
                indv.score = 0
                indv.fitness = 0
                indv.run([env])
                test_rwd += indv.fitness
	    test_elite.append(round(test_rwd/psize, 3))
	    
	    test_rwd = 0
	    for _ in range(trials):    
                arm = ucb.pull()
                r = env.pull_arm(arm)
                test_rwd += int(env.best_arm==arm)
                ucb.update(arm, r)
	    ucb.reset()
	    test_ucb.append(round(test_rwd/psize, 3))

	    


	# In[ ]:


	labels = ['Env ' + str(i) for i in range(len(test_envs))]

	x = np.arange(len(labels))  # the label locations
	width = 0.5  # the width of the bars

	fig, ax = plt.subplots()


	rects1 = ax.bar(x - width/2, test_pop, width/2, label='Population')
	rects2 = ax.bar(x, test_elite, width/2, label='Elite')
	rects3 = ax.bar(x + width/2, test_ucb, width/2, label='UCB')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('Scores')
	ax.set_title('Scores for different envirnoments')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.legend()


	def autolabel(rects):
	    for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
		            xy=(rect.get_x() + rect.get_width() / 2, height),
		            xytext=(0, 3),  # 3 points vertical offset
		            textcoords="offset points",
		            ha='center', va='bottom')


	autolabel(rects1)
	autolabel(rects2)
	autolabel(rects3)

	fig.tight_layout()

	if args.save_result:
            if not args.noise:
                plt.savefig('results/'+args.layers + 'x' + args.hidden + 'mag' + args.mutation_mag + 'arms' + args.arms + 'psize' + args.pop_size + 'trials' + args.trials + args.name + '/testing'+'.svg', format='svg')
            else:
                plt.savefig('results/'+args.layers + 'x' + args.hidden + 'mag' + args.mutation_mag + 'arms' + args.arms + 'psize' + args.pop_size + 'trials' + args.trials + args.name + '/noise_testing'+'.svg', format='svg')
			


	plt.close()
	print('Done:' )
	print(args)

