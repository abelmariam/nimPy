"""
@author: Abel Mariam
"""

import numpy as np

#An action in the game of Nim is an ordered pair (x,y) where x is in {0,1,2} and
#represents which pile the agent is going to remove objects from and y is the 
#number of objects they are going to remove.
class Action:
    def __init__(self, action):
        if type(action) is (not list or not tuple):
            raise TypeError("Argument not a list or tuple")
        else:
            self.action = tuple(action)

#Positions in the game of Nim represented internally as ordered pair (x,y,z) where x,y,z are
#non-negative integers which represent how many objects are in each respective pile.
class State:
    def __init__(self, state):
        if type(state) is (not list or not tuple):
            raise TypeError("Argument not a list or tuple")
        else:
            self.state = tuple(state)
        
    def isTerminal(self):
        return np.array_equal(self.state,[0,0,0])  
        
    #Return a list with all valid actions which are valid from the given state     
    def getActions(self):
        if self.isTerminal():
            return [None]
        else:
            return [Action([x,y]) for x in {0,1,2} for y in np.arange(1,self.state[x]+1) if self.state[x]>0]
    
    def nimSum(self):
        return self.state[0]^self.state[1]^self.state[2]
        
    #Performs the given action on the state and thus transforms it into a new state.    
    def doAction(self,kAction):
        if kAction is not None:
            self.state = list(self.state)
            self.state[kAction.action[0]] -= kAction.action[1]
            self.state = tuple(self.state)
         
    #Performs the given action on this state and and returns the new state.
    def peekAction(self,kAction):
        newState = State(list(self.state))
        newState.doAction(kAction)
         
        return newState
         