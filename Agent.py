"""
@author: Abel Mariam
"""

import sys
import numpy as np
from abc import ABCMeta, abstractmethod
from State_Action import *

#Abstract base class which outlines the basic functionality of all agent
class Agent:
    __metaclass__ = ABCMeta
    
    #Specifies the agent's strategy so that given a state, the agent's policy 
    #will return an action to perform in that state
    @abstractmethod
    def policy(self,state): pass

#Agent which selects its actions at random
class Random(Agent):
    def policy(self,state):
        return np.random.choice(state.getActions())

#User interface so that the user can participate in the game
class Human(Agent):
    def policy(self,state):
        userInput = input("Please enter your move: ")
     
        while True:
            try: 
                userTokens = list(map(int, userInput.split(",")))
                if userInput.strip().lower() == "quit":
                    sys.exit()
                elif not state.isValid(Action(userTokens)):
                    raise ValueError
                else:
                    return Action(userInput)
                    
            except ValueError:
                print("Woops! That is not a valid move. Try again..")

#Agent which selects the optimal action when possible
class Optimal(Agent):
    def policy(self,currState): 
        for i in [0,1,2]:
            sub = [currState.state[x] for x in [0,1,2] if x != i]
            nimSum = sub[0]^sub[1]
            
            if (currState.state[i] - np.abs(nimSum-currState.state[i])) is nimSum:
                return Action((i,abs(nim-currState.heap[i])))
                
        return np.random.choice(currState.getActions())
            
#Agent that adaptively learns through the Q-Learning algorithm
class Q_Learning(Agent):
    #Q is a function f: State x Action -> R and is internally represented as a Map.

    #Alpha is the learning rate and determines to what extent the newly acquired 
    #information will override the old information

    #Gamma is the discount rate and determines the importance of future rewards

    #Epsilon serves as the exploration rate and determines the probability 
    #that the agent, in the learning process, will randomly select an action
  
    Q = {}
    prevState = prevAction = None
    WIN_REWARD, LOSS_REWARD = 1.0, -1.0 
    
    def __init__(self, alpha, gamma, epsilon):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
    
    #If a given state is not in our hashmap, we will add all possible state,
    #action pair and initialize the values by randomly sampling from the uniform(0,1)
    #(so we can avoid having two actions which attain the maximum Q-value)
    def makeKey(self,currState):
        possActions = currState.getActions()
        someAction = possActions[0].action

        if (currState.state, someAction) not in self.Q:
            for i in possActions:
                self.Q[(currState.state, i.action)] = np.random.uniform(0.0,0.01)
    
    def policy(self,currState):
        possActions = currState.getActions()
        if np.random.random() > self.epsilon:
            #Returns the action associated with the max Q value
            qVal = [self.Q[(currState.state, a.action)] for a in possActions]
            return possActions[np.argmax(qVal)]
        else:
            #Selects the action at random 
            return np.random.choice(possActions)
    
    #Updates the Q-table as specified by the standard Q-learning algorithm
    def updateQ(self,currState):
        if currState.isTerminal():
            self.Q[(self.prevState.state, self.prevAction.action)] += \
                self.alpha * (self.LOSS_REWARD - self.Q[(self.prevState.state, self.prevAction.action)])
            currAction = self.prevState = self.prevAction = None
        else:
            self.makeKey(currState)
            currAction = self.policy(currState)
            
            if self.prevAction is not None:
                nextState = currState.peekAction(currAction)
                reward = 0 if not nextState.isTerminal() else self.WIN_REWARD
                maxQ = max([self.Q[(currState.state,a.action)] for a in currState.getActions()])
                
                self.Q[(self.prevState.state, self.prevAction.action)] += \
                    self.alpha * (reward + (self.gamma * maxQ) - \
                    self.Q[(self.prevState.state, self.prevAction.action)])
            self.prevState, self.prevAction = State(currState.state), currAction
            
        return currAction