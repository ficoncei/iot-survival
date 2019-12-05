# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import random
import torchvision
import matplotlib.pyplot as plt
from collections import Counter 
from random import randint

print(torchvision.__version__)

device = torch.device("cpu")

#tunable params
alpha=0.8
beta=0.6
bmax=384
packetsizemax=80
gAC=0.9
episodes=10
epochs=50
inputSize, hiddenSize, outputSize= 2, 6, 7
muenergy, sigmaenergy, mupacketsize, sigmapacketsize = 63, 1, 40, 1

#init params
cn=4
lengthcn=4
cnmax=4
preferencesmapping=np.empty(cnmax)

#define dictionary and action set
eprime=np.zeros((packetsizemax,cnmax))
actionSet = [-3,-2,-1,0,1,2,3]

#useful "python variables"
maxStates=(bmax+1)*cn
visitsCount=np.zeros((maxStates, len(actionSet)))
blockofstatesperlevel=math.floor((maxStates+1)/lengthcn)

#define performance metrics Deep leraning approach + full security approach
ep_rewards = np.zeros(episodes)
ep_discpackets = np.zeros(episodes)
ep_rewardsmaxSecurity = np.zeros(episodes)
ep_discpacketsmaxSecurity = np.zeros(episodes)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputSize,hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.fc10 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc10(x))
        return x

model=  Net()
    
#security overhead dictionary defintion
for idn in range(packetsizemax):
    for icn in range(cnmax):
        if (idn==0):
            eprime[idn][icn]=0
        elif( icn == 0 ):
            overhead=0
        elif( icn == 1 ):
            overhead=5
        elif( icn == 2 ):
            overhead=16
        elif( icn == 3 ):
            overhead=21
        if (idn!=0):
            eprime[idn][icn]=math.ceil((idn+overhead)/88*100)
        else:
            eprime[idn][icn]=0


#returns battery status from state number and security level
def derivebnfromstate(istate,icn):
    return istate-(bmax+1)*(icn-1)

#returns allowed action set from state number
def getupperlowercn(istate,blockofstatesperlevel):
    if (istate<blockofstatesperlevel):
        upper=len(actionSet)-1;
        lower=3;
        icn=1
    elif (istate>=blockofstatesperlevel and istate<2*blockofstatesperlevel):
        upper=len(actionSet)-2;
        lower=2;
        icn=2
    elif (istate>=2*blockofstatesperlevel and istate<3*blockofstatesperlevel):
        upper=len(actionSet)-3;
        lower=1;
        icn=3
    elif istate>=3*blockofstatesperlevel:
        upper=len(actionSet)-4;
        lower=0;
        icn=4
        
    return upper,lower,icn


#selects a random action from the allowed action set, given the state number
#for an exploration strategy in action selection
def selectRandomAction(istate,blockofstatesperlevel):
    upper,lower,icn = getupperlowercn(istate,blockofstatesperlevel)
    ibn=derivebnfromstate(istate,icn)
    inputLayer = torch.tensor([ibn,icn]).float()
    outputLayer_pred = model(inputLayer)
    preferencesmapping=F.softmax(outputLayer_pred[lower:upper+1],dim=0)
    rndnumber=np.random.rand()
    if rndnumber<=preferencesmapping[0]:
        action=lower;
    elif (rndnumber>preferencesmapping[0] and rndnumber<=preferencesmapping[0]+preferencesmapping[1]):
        action=lower+1;
    elif (rndnumber>preferencesmapping[0]+preferencesmapping[1] and rndnumber<preferencesmapping[0]+preferencesmapping[1]+preferencesmapping[2]):
        action=lower+2;
    else:
        action=upper;
    return action;

#selects the action with highest value from the allowed action set, given the state number
#for a greedy strategy in action selection
def selectmaxargaction(istate,blockofstatesperlevel):
    upper,lower,icn = getupperlowercn(istate,blockofstatesperlevel)
    ibn=derivebnfromstate(istate,icn)
    inputLayer = torch.tensor([ibn,icn]).float()
    outputLayer_pred = model(inputLayer)
    aux=F.softmax(outputLayer_pred[lower:upper+1],dim=0)
    aux=aux.tolist()
    ind=lower
    for elem in aux:
        if (aux.count(elem)>1 and elem==max(aux)):
            randnumber=randint(1,aux.count(elem))
            for elem2 in aux:
                if randnumber==1 and elem2==elem:
                    ind+=1
                    return lower+aux.index(elem2)
                elif elem2==elem:
                    ind+=1
                    randnumber-=1
                else:
                    ind+=1
        elif (aux.count(elem)>1 and elem!=max(aux)):
            randnumber=randint(1,aux.count(elem))
        elif (aux.count(elem)==1 and elem==max(aux)):
            return lower+aux.index(max(aux))

#auxiliary function to break ties for actions with the same value
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 

#outputs the action value of an action, given the current state and its action
def estimateActionValue(istate,iaction,icn,bmax):
    ibn=derivebnfromstate(istate,icn)
    inputLayer = torch.tensor([ibn,icn]).float()
    outputLayer_pred = model(inputLayer)
    return outputLayer_pred[iaction]

#outputs the probability of a specific action, given an input state
def estimateActionProb(istate,iaction,icn,bmax):
    ibn=istate-(bmax+1)*(icn-1);
    inputLayer = torch.tensor([ibn,icn]).float()
    outputLayer_pred = model(inputLayer)
    outputLayer_pred=F.softmax(outputLayer_pred,dim=0)
    return outputLayer_pred[iaction]
    
#trains the NN 
def trainNN(delta,target,reward,istate,curAction,alpha,beta):
    #feed forward and loss calculation
    upper,lower,icn = getupperlowercn(istate,blockofstatesperlevel)
    ibn=derivebnfromstate(istate,icn)
    inputLayer = torch.tensor([ibn,icn]).float()
    outputLayer = model(inputLayer)
    logprobs = F.log_softmax(outputLayer[lower:upper+1],dim=0)
    policy_loss = -logprobs[curAction-lower]*delta
    
    #NN training with ADAM algorithm backpropagation
    loss_function = torch.nn.MSELoss()
    value_loss=loss_function(outputLayer[curAction],target)*delta
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    value_loss.backward()
    optimizer.step()
   
#reward function - security component
def securityf(cn,alpha):
    if cn==1:
        return 0
    elif cn==2:
        return 1-alpha
    elif cn==3:
        return alpha
    elif cn==4:
        return 1
    else:
        print('cn',cn)

#reward function - energy component
def rewardfunction(batteryratio,cn,alpha,beta):
    sec=securityf(cn,alpha)
    return (1-beta)*sec+beta*(batteryratio)

#plots results from obtained policy at the end of each episode
def plot_results():
    fig, ax = plt.subplots()
    ax.plot(ep_rewards[0:episodes], 'b', label='Deep Q-learning policy')
    ax.plot(ep_rewardsmaxSecurity[0:episodes], 'r', label='Fixed Lv4 Policy (SotA)')
    ax.legend();
    plt.title('Cumulative Rewards after each episode')
    plt.ylabel('Avg Rewards')
    plt.xlabel('Episodes')



#simulates current policy
def simulatePolicyDLapproach():
    runtimes=5000
    discpackets_sim=0
        
    for runtime in range(runtimes):
        episoderewards_sim=0
        nextState=random.randint(1,maxStates-1)  #Initialize state
        upper,lower,cn=getupperlowercn(nextState,blockofstatesperlevel)
        bn=derivebnfromstate(nextState,cn)
        
        for epoch in range (epochs):
            currentState=nextState
            upper,lower,cn=getupperlowercn(currentState,blockofstatesperlevel)
            currentAction=selectmaxargaction(currentState,blockofstatesperlevel)
            nextcn=cn+currentAction-3
    
            if (dnruntime[epoch]<=0):
                packetsize=0;
            elif (dnruntime[epoch]>=80):
                packetsize=80;
            else:
                packetsize=dnruntime[epoch];
           
            bn=derivebnfromstate(currentState,cn)
            bnold=bn
            bn=bn+hnruntime[epoch]-eprime[packetsize][cn-1] 
            if (bn>bmax):
                bn=bmax
                batteryratio=bn*1.0/bmax
                reward_sim=rewardfunction(batteryratio,nextcn,alpha,beta) #observe R
            elif (bn<=0):
                bn=bnold
                reward_sim=0  #observe R
                discpackets_sim=+1 #discard packet
            else:
                batteryratio=bn*1.0/bmax
                reward_sim=rewardfunction(batteryratio,nextcn,alpha,beta) #observe R
            
            episoderewards_sim+=reward_sim
            nextState=(bmax+1)*(nextcn-1)+bn
            cn=nextcn
        
        ep_discpackets[iepisode]+=discpackets_sim
        ep_rewards[iepisode]+=episoderewards_sim

    ep_rewards[iepisode]=ep_rewards[iepisode]/runtimes/epochs
    ep_discpackets[iepisode]=ep_discpackets[iepisode]/runtimes/epochs
    #print(ep_rewards[iepisode])
    #print(ep_discpackets[iepisode])
    
    
def simulatePolicyfullsec():
    runtimes=5000
    discpackets_sim=0
        
    for runtime in range(runtimes):
        episoderewards_sim=0
        nextState=random.randint(1,maxStates-1)  #Initialize state
        upper,lower,cn=getupperlowercn(nextState,blockofstatesperlevel)
        bn=derivebnfromstate(nextState,4)
        
        for epoch in range (epochs):
            currentState=nextState
            upper,lower,cn=getupperlowercn(currentState,blockofstatesperlevel)
            nextcn=4
    
            if (dnruntime[epoch]<=0):
                packetsize=0;
            elif (dnruntime[epoch]>=80):
                packetsize=80;
            else:
                packetsize=dnruntime[epoch];
           
            bn=derivebnfromstate(currentState,cn)
            bnold=bn
            bn=bn+hnruntime[epoch]-eprime[packetsize][cn-1] 
            if (bn>bmax):
                bn=bmax
                batteryratio=bn*1.0/bmax
                reward_sim=rewardfunction(batteryratio,nextcn,alpha,beta) #observe R
            elif (bn<=0):
                bn=bnold
                reward_sim=0  #observe R
                discpackets_sim=+1 #discard packet
            else:
                batteryratio=bn*1.0/bmax
                reward_sim=rewardfunction(batteryratio,nextcn,alpha,beta) #observe R
            
            episoderewards_sim+=reward_sim
            nextState=(bmax+1)*(nextcn-1)+bn
            cn=nextcn
        
        
        ep_discpacketsmaxSecurity[iepisode]+=discpackets_sim
        ep_rewardsmaxSecurity[iepisode]+=episoderewards_sim

    ep_rewardsmaxSecurity[iepisode]=ep_rewards[iepisode]/runtimes/epochs
    ep_discpacketsmaxSecurity[iepisode]=ep_discpackets[iepisode]/runtimes/epochs
    print(ep_rewardsmaxSecurity[iepisode])
    print(ep_discpacketsmaxSecurity[iepisode])

#run episodes and simulate the current policy to obtain the current average cumulative rewards
for iepisode in range (episodes):
    nextState=random.randint(1,maxStates-1)   #(bmax+1)*cn-1  #Initialize S
    upper,lower,cn=getupperlowercn(nextState,blockofstatesperlevel)
    bn=derivebnfromstate(nextState,cn)
    i=1.;
    episoderewards=0
    #generate random energy and packet size inputs for each episode
    hnruntime=np.random.normal(muenergy, sigmaenergy, epochs)
    dnruntime=np.random.normal(mupacketsize, sigmapacketsize, epochs)
    hnruntime=hnruntime.round()
    hnruntime=hnruntime.astype(int)
    dnruntime=dnruntime.round()
    dnruntime=dnruntime.astype(int)

    for epoch in range (epochs):
        currentState=nextState
        if epoch==0:
            currentAction=selectRandomAction(currentState,blockofstatesperlevel)
        else:
            currentAction=nextaction
        visitsCount[int(currentState)][currentAction]=visitsCount[int(currentState)][currentAction]+1
        nextcn=cn+currentAction-3
        if (dnruntime[epoch]<=0):
            packetsize=0;
        elif (dnruntime[epoch]>=80):
            packetsize=80;
        else:
            packetsize=dnruntime[epoch];
        bnold=bn
        bn=bn+hnruntime[epoch]-eprime[packetsize][cn-1]     

        if (bn>=bmax):
            bn=bmax
            batteryratio=bn*1.0/bmax
            reward=rewardfunction(batteryratio,nextcn,alpha,beta) #observe R
        elif (bn<=0):
            bn=bnold
            reward=0  #observe R
        else:
            batteryratio=bn*1.0/bmax
            reward=rewardfunction(batteryratio,nextcn,alpha,beta) #observe R

        nextState=(bmax+1)*(nextcn-1)+bn
        nextaction=selectmaxargaction(nextState,blockofstatesperlevel)
    
        q_sa=estimateActionValue(currentState,currentAction,cn,bmax);
        nextcn=cn+currentAction-3
        
        q_sprime_aprime=estimateActionValue(nextState,nextaction,nextcn,bmax);
        target=reward+gAC*q_sprime_aprime
        delta=reward+gAC*q_sprime_aprime-q_sa
        trainNN(delta,target,reward,currentState,currentAction,alpha,beta)
        
        cn=nextcn        
        i=gAC*i;


    print("Episode",iepisode, "ending\n")

    simulatePolicyDLapproach()

simulatePolicyfullsec()

plot_results()
