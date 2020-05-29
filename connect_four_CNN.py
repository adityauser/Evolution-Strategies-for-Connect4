import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,grad
import torch
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms

import time
from scipy.optimize import minimize_scalar
import weakref
from pdb import set_trace as bb
import collections
import numpy as np
import random
import uuid
import pyspiel
import re

do_cuda = False 
if not do_cuda:
    torch.backends.cudnn.enabled = False

#global environment
env = None
max_episode_length = 600

#global archive to keep track of states sampled from environment
state_archive = collections.deque([], 200)

def set_maze(config):
    global env
    env = pyspiel.load_game("connect_four")

from pytorch_helpers import *

#default feed-forward net architecture
class netmodel(torch.nn.Module):
    def __init__(self, num_inputs, action_space, settings={}, in_channels=3, num_features=4):
        super(netmodel, self).__init__()

        #should hidden units have biases?
        num_outputs = action_space

        self.main = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(in_channels, num_features, 4,     #[7,6]
                      stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),

            nn.Conv2d(num_features, num_features * 2, 4,  #[6,5]
                      stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),

            nn.Conv2d(num_features * 2, num_features * 4,    #[5,4]
                      4, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),

            nn.Conv2d(num_features * 4, num_features * 8,  #[4,3]
                      4, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),

          #  nn.Conv2d(num_features * 8, num_features *
           #           16, 4, stride=1, padding=1, bias=False),  #
         #   nn.ELU(inplace=True),

            nn.Conv2d(num_features * 8, num_outputs, 4,  #[3,2]
                      stride=1, padding=1, bias=False),

            nn.Flatten(),

            nn.Linear(num_outputs * 2, num_outputs),

            nn.Softmax(1)
        )

    def forward(self, inputs,intermediate=None,debug=False):
      #  print(help(inputs))
        main = self.main(inputs.float())
        return main

    #function to return current pytorch gradient in same order as genome's flat vector theta
    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for param in self.parameters():
            sz = param.grad.data.numpy().flatten().shape[0]
            pvec[count:count + sz] = param.grad.data.numpy().flatten()
            count += sz
        return pvec.copy()

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

    #function to take a flat vector and reshape it to resemble neural network weights
    def reshape_parameters(self,pvec):
        count = 0
        numpy_params = []

        for name,param in self.named_parameters():
            sz = param.data.numpy().flatten().shape[0]
            raw = pvec[count:count + sz]
            reshaped = raw.reshape(param.data.numpy().shape)
            numpy_params.append((name,reshaped))
            count += sz

        #print ([ (r[0],(r[1]**2).sum().mean()) for r in numpy_params])
        return numpy_params

    #function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        tot_size = self.count_parameters()
        count = 0

        for param in self.parameters():
            sz = param.data.numpy().flatten().shape[0]
            raw = pvec[count:count + sz]
            reshaped = raw.reshape(param.data.numpy().shape)
            param.data = torch.from_numpy(reshaped)
            count += sz

        return pvec

    #count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for param in self.parameters():
            #print param.data.numpy().shape
            count += param.data.numpy().flatten().shape[0]
        return count


#genome class that can be mutated, selected, evaluated in domain
#substrate for evolution
class individual:
    env = None  #perhaps turn this into an env_generator? or pass into constructor? for parallelization..
    model_generator = None
    global_model = None
    rollout = None
    instances = []

    def __init__(self, identity):
        self.noise = 0.05
        self.smog = False
        self.id = uuid.uuid4().int
        self.live_descendants = 0
        self.alive = True
        self.dead_weight = False
        self.parent= None
        self.percolate = False
        self.selected = 0
        self.matchs_played = 0
        self.net_reward = 0
        self.identity = identity

        if self.percolate:
            self.__class__.instances.append(weakref.proxy(self))

    def copy(self, identity, percolate=False): 
        new_ind = individual(identity)
        new_ind.genome = self.genome.copy()
        new_ind.states = self.states

        if self.percolate:
            new_ind.parent = self

        #update live descendant count
        if self.percolate:
            self.live_descendants += 1
            if hasattr(self,'parent'):
                p_pointer = self.parent
            else:
                p_pointer = None
                self.parent = None
            while p_pointer != None:
                p_pointer.live_descendants += 1 
                p_pointer = p_pointer.parent
        
        return new_ind

    def kill(self):
        self.alive=False
        if self.live_descendants <= 0:
                self.dead_weight=True
        self.remove_live_descendant()

    def remove_live_descendant(self):
        p_pointer = self.parent
        while p_pointer != None:
            p_pointer.live_descendants -= 1
            if p_pointer.live_descendants <= 0 and not p_pointer.alive:
                p_pointer.dead_weight=True

            p_pointer = p_pointer.parent

    def mutate(self, mutation='regular', **kwargs):

        #plain mutation is normal ES-style mutation
        if mutation=='regular':
            self.genome = mutate_plain(self.genome, states=self.states,**kwargs)
        elif mutation.count("SM-G")>0:
            #smog_target is target-based smog where we aim to perturb outputs
            self.genome = mutate_sm_g(
                    mutation,
                    self.genome,
                    individual.global_model,
                    individual.env,
                    states=self.states,
                    **kwargs)
            #smog_grad is TRPO-based smog where we attempt to induce limited policy change
        elif mutation.count("SM-R")>0:
                self.genome = mutate_sm_r(
                    self.genome,
                    individual.global_model,
                    individual.env,
                    states=self.states,
                    **kwargs)
        else:
            assert False

    #randomly initialize genome using underlying ANN's random init
    def init_rand(self):
        global controller_settings
        model = individual.model_generator
        env = individual.env
        state = env.new_initial_state()
        
        newmodel = model(len(state.observation_tensor()), len(state.legal_actions()), controller_settings)
        self.genome = newmodel.extract_parameters()

    def render(self, screen):
        pass

    #evaluate genome in environment with a roll-out
    def map(self, push_all=True, trace=False, against = None, test = False):
        global state_archive
        individual.global_model.inject_parameters(self.genome)
        reward, state_buffer, terminal_state, broken = individual.rollout({}, individual.global_model, individual.env, against)

        if test:
            return reward, terminal_state, broken


        if push_all:
            state_archive = state_buffer
            self.states = state_buffer
        else:
            print("not using all states")
            state_archive.appendleft(random.choice(state_buffer))
            self.states = None

        self.broken = broken

        if not broken:
            self.reward = reward
            self.net_reward+=reward
            self.solved = reward==1
            self.matchs_played+=1
        else:
            self.matchs_played+=1
            self.net_reward = -1e8
        _c4_fitness(self)

        return terminal_state, broken

    def prepare(self):
        pass

    #does individual solve the task?
    def solution(self):
        return self.solved

    #save genome out
    def save(self, fname):
        if fname.count(".npy")==0:
            fname_new=fname+".npy"
        else:
            fname_new=fname
        np.save(fname_new, self.genome)

    #load genome in
    def load(self, fname):
        if fname.count(".npy")==0:
            fname_new=fname+".npy"
        else:
            fname_new=fname
        self.genome = np.load(fname_new)
        print (self.genome.shape)
        print (model.extract_parameters().shape )

#Method to conduct maze rollout
@staticmethod
def do_rollout(args, model, env, against, render=False, screen=None, trace=False):
    state_buffer = collections.deque([], 400)

    transform = transforms.Compose([transforms.ToTensor()])

    action_repeat = 3

    #TODO: Replace the random selection of first turn with some deterministic method.
    turn = random.sample([1, 0], 1)[0]

    state = env.new_initial_state()
    obs = state.observation_tensor()
    obs = np.transpose(np.reshape(np.array(state.observation_tensor()), (3, int(len(state.observation_tensor())/(3*7)), 7)))
    this_model_return = 0

    broken = False
    action = None

    if not against==None:
        opponent_model = copy.deepcopy(against.global_model)
        opponent_model.inject_parameters(against.genome)

    # Rollout
    for step in range(max_episode_length):     

        player = state.current_player()

        if len(state.legal_actions(player))<1:
            print("No move left",state.legal_actions(player))

        obs = transform(obs).unsqueeze(0)
        

        if (player + turn)%2 == 0:
            if against ==None:
                action = random.choice(state.legal_actions(player))
            else:
                logit = opponent_model(Variable(obs, volatile=True))
                actions = logit.data.numpy()[0]

                if np.isnan(actions).any():
                    print("check_nan1",actions)

                for a in range(len(actions)):
                    actions[a] = int(a in state.legal_actions(player))*actions[a]

                if np.isnan(actions).any():
                    print("check_nan2",actions)

                actions = actions/sum(actions)

                if np.isnan(actions).any():
                    print("check_nan3",actions)
                    print(state)
                    broken=True
                    break
                action = np.random.choice(len(actions), 1, p=actions)[0]

                
        elif (player + turn)%2 == 1:
            logit = model(Variable(obs, volatile=True))
            actions = logit.data.numpy()[0]

            if np.isnan(actions).any():
                print("check_nan1",actions)

            for a in range(len(actions)):
                actions[a] = int(a in state.legal_actions(player))*actions[a]

            if np.isnan(actions).any():
                print("check_nan2",actions)

            actions = actions/sum(actions)

            if np.isnan(actions).any():
                print("check_nan3",step,actions)
                print(state)
                broken=True
                break

            action = np.random.choice(len(actions), 1, p=actions)[0]
        else:
            raise("Illegal player")

            

        if action not in state.legal_actions(player):
            print("ab kya kru")
            print(state.legal_actions(player), actions)
        #print(action)
            

        state.apply_action(action)
        
        reward = state.returns()[(1 + turn)%2]
    

        done = state.is_terminal()

        this_model_return += reward

        if True:  #(random.random()<0.05):
            state_buffer.appendleft(obs.squeeze(0).numpy())  

        if done:
            break

        obs = np.transpose(np.reshape(np.array(state.observation_tensor()), (3, int(len(state.observation_tensor())/(3*7)), 7)))


    return this_model_return, state_buffer, state, broken


def mutate_plain(params, mag=0.05,**kwargs):
    do_policy_check = False

    delta = np.random.randn(*params.shape).astype(np.float32)*np.array(mag).astype(np.float32)
    new_params = params + delta

    diff = np.sqrt(((new_params - params)**2).sum())

    if do_policy_check:
        output_dist = check_policy_change(params,new_params,kwargs['states'])
        print("mutation size: ", diff, "output distribution change:",output_dist)
    else:
        print("mutation size: ", diff)

    return new_params


def mutate_sm_r(params,
                     model,
                     env,
                     verbose=True,
                     states=None,
                     mag=0.01):
    global state_archive

    model.inject_parameters(params.copy())

    if states == None:
        states = state_archive

    delta = np.random.randn(*(params.shape)).astype(np.float32)
    delta = delta / np.sqrt((delta**2).sum())

    sz = min(100,len(state_archive))
    np_obs = np.array(random.sample(state_archive, sz), dtype=np.float32)
    verification_states = Variable(
        torch.from_numpy(np_obs), requires_grad=False)

    output = model(verification_states)
    old_policy = output.data.numpy()

    #do line search
    threshold = mag
    search_rounds = 15
    
    def search_error(x,raw=False):
        new_params = params + delta * x
        model.inject_parameters(new_params)

        output = model(verification_states).data.numpy()

        change = ((output - old_policy)**2).mean()
        if raw:
            return change
        return (change-threshold)**2
    
    mult = minimize_scalar(search_error,tol=0.01**2,options={'maxiter':search_rounds,'disp':True})
    new_params = params+delta*mult.x

    print ('Distribution shift:',search_error(mult.x,raw=True))
    print ("SM-R scaling factor:",mult.x)
    diff = np.sqrt(((new_params - params)**2).sum())
    print("mutation size: ", diff)
    return new_params

def check_policy_change(p1,p2,states):
    model.inject_parameters(p1.copy())
    #TODO: check impact of greater accuracy
    sz = min(100,len(states))

    verification_states = np.array(random.sample(states, sz), dtype=np.float32)
    verification_states = Variable(torch.from_numpy(verification_states), requires_grad=False)
    old_policy = model(verification_states).data.numpy()
    old_policy = Variable(torch.from_numpy(old_policy), requires_grad=False)

    model.inject_parameters(p2.copy())
    model.zero_grad()
    new_policy = model(verification_states)
    divergence_loss_fn = torch.nn.MSELoss(size_average=True)
    divergence_loss = divergence_loss_fn(new_policy,old_policy)

    return divergence_loss.data[0]


def mutate_sm_g(mutation, 
                       params,
                       model,
                       env,
                       verbose=False,
                       states=None,
                       mag=0.1,
                        **kwargs):

    global state_archive

    #inject parameters into current model
    model.inject_parameters(params.copy())

    #if no states passed in, use global state archive
    if states == None:
        states = state_archive

    #sub-sample experiences from parent
 #   print(states[-1], states)
    sz = min(100,len(states))
    verification_states = np.array(random.sample(states, sz), dtype=np.float32)
    verification_states = Variable(torch.from_numpy(verification_states), requires_grad=False)

    #run experiences through model
    #NOTE: for efficiency, could cache these during actual evalution instead of recalculating
    old_policy = model(verification_states)

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
                old_policy_j = model(verification_states[j:j+1])
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

        output = model(verification_states).data.numpy()
        change = np.sqrt(((output - old_policy)**2).sum(1)).mean()

        if raw:
            return change

        return (change-threshold)**2

    if linesearch:
        mult = minimize_scalar(search_error,bounds=(0,0.1,3),tol=(threshold/4)**2,options={'maxiter':search_rounds,'disp':True})
        print ("linesearch result:",mult)
        chg_amt = mult.x
    else:
        #if not doing linesearch
        #don't change perturbation
        chg_amt = 1.0

    final_delta = delta*chg_amt
    print ('perturbation max magnitude:',final_delta.max())

    final_delta = np.clip(delta,-weight_clip,weight_clip)
    new_params = params + final_delta

    print ('max post-perturbation weight magnitude:',abs(new_params).max())

    if verbose:
        print("divergence:", search_error(chg_amt,raw=True))

    diff = np.sqrt(((new_params - params)**2).sum())
    print("mutation size: ", diff)

    return new_params


model = None
controller_settings = None

def setup(maze,_controller_settings):
    global model,controller_settings
    set_maze(maze)
    controller_settings = _controller_settings
    state = env.new_initial_state()
    model = netmodel(len(state.observation_tensor()), len(state.legal_actions()), controller_settings)

    if do_cuda:
        model.cuda()

    individual.env = env
    individual.model_generator = netmodel
    individual.rollout = do_rollout
    individual.global_model = model

'''
#breadcrumb finess calculation (given breadcrumb array)
def _breadcrumb_fitness(ind):
    global breadcrumb
    pos_x = int(ind.behavior[-2])
    pos_y = int(ind.behavior[-1])
    ind.fitness = -breadcrumb[pos_x,pos_y]
    if ind.broken:
     ind.fitness = -1e8
'''

def _c4_fitness(ind):
    ind.fitness = ind.net_reward/ind.matchs_played

    #All broken rollouts are ignored so, basicly this if statement is redundant.
    if ind.broken:
     ind.fitness = -1e8


if __name__ == '__main__':

    solution_file="solution.npy"

    setup({'maze':"connect_four.txt"},{'layers':64,'af':'tanh','size':125,'residual':True})
    robot = individual()
    robot.load(solution_file)
    robot.render(screen)
