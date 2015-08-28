"""
@author: Abel Mariam
"""
import numpy as np

from Agent import *

class Nim:
    wins = losses = 0
    
    def __init__(self, player1, player2, nGames):
        self.player1 = player1
        self.player2 = player2
        self.nGames = nGames
    
    def play(self, initState):
        
        for i in np.arange(self.nGames):
            currState = State(initState.state)
            
            while True:
                action_p1 = self.player1.updateQ(currState) if type(self.player1) is Q_Learning \
                                                            else self.player1.policy(currState)
                currState.doAction(action_p1)
                
                if action_p1 is None:
                    self.losses += 1
                    break
                elif currState.isTerminal():
                    self.wins += 1
                    break
                
                action_p2 = self.player2.policy(currState)
                currState.doAction(action_p2)            
    
    def main():
        #FIXME: Add command line interface so that you can change various paramaters
        #of the game such as which kinds of agents will be playing, if the q-learning
        #agent is playing then what are the learning parameters: alpha, gamma, epsilon.
    
        #FIXME: Test user agent gameplay
            
        #ADDITIONAL FUNCTIONALITY: 
        #1. How many iterations will they be trained for, how many will they test for,
        #2. Allow the user to input the Q-table so that we can skip training.
        #3. Option for saving the results of the training, testing. 
    
        qAgent = Q_Learning(0.35, 0.9, 0.35)
        optAgent = Optimal()
        
        q_vs_opt_train = Nim(qAgent,optAgent,100000)
        q_vs_opt_train.play(State(np.random.random_integers(1,10,3)))
        print("Train - Wins: ", q_vs_opt_train.wins, "Losses: ", q_vs_opt_train.losses)
        
        qAgent.epsilon = 0
        q_vs_opt_test = Nim(qAgent,optAgent,100000)
        q_vs_opt_test.play(State(np.random.random_integers(1,10,3)))
        print("Test - Wins: ", q_vs_opt_test.wins, "Losses: ", q_vs_opt_test.losses)
        
if __name__ == "__main__":
    Nim.main()
        