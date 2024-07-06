# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        foodList = newFood.asList()

        if foodList:
            minFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
            score += 20 / (minFoodDistance + 1)

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        '''
        recall psuedocode:
        if the state is a terminal state: return state's utility
        if the agent is MAX: return max-value(state)
        if the agent is MIN: return max-value(state)
        '''

        bestAction = None
        maxValue = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = self.minValue(successor, 1, 0)
            if value > maxValue:
                maxValue = value
                bestAction = action
        
        return bestAction

    def maxValue(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        value = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = max(value, self.minValue(successor, 1, depth))
        
        return value

    def minValue(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        value = float('inf')
        numAgents = gameState.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth + (1 if nextAgent == 0 else 0)

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if nextAgent == 0:
                value = min(value, self.maxValue(successor, nextDepth))
            else:
                value = min(value, self.minValue(successor, nextAgent, nextDepth))
        
        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        maxValue = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = self.minValue(successor, 1, 0, alpha, beta)
            if value > maxValue:
                maxValue = value
                bestAction = action
                alpha = value
            alpha = max(alpha, value)
        
        return bestAction

    def maxValue(self, gameState, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        value = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = max(value, self.minValue(successor, 1, depth, alpha, beta))
            if value > beta:
                return value
            alpha = max(alpha, value)
        
        return value

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        
        value = float('inf')
        numAgents = gameState.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth + (1 if nextAgent == 0 else 0)

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if nextAgent == 0:
                value = min(value, self.maxValue(successor, nextDepth, alpha, beta))
            else:
                value = min(value, self.minValue(successor, nextAgent, nextDepth, alpha, beta))
            
            if value < alpha:
                return value
            beta = min(beta, value)
        
        return value
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            if agentIndex == 0:  
                return max(expectimax(state.generateSuccessor(agentIndex, action), depth, 1)
                           for action in state.getLegalActions(agentIndex))
            else:  
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth - 1 if nextAgent == 0 else depth
                actions = state.getLegalActions(agentIndex)
                probabilities = [1.0 / len(actions)] * len(actions)  
                return sum(prob * expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                           for action, prob in zip(actions, probabilities))

        bestAction = max(gameState.getLegalActions(0),
                         key=lambda action: expectimax(gameState.generateSuccessor(0, action), self.depth, 1))
        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodList = foodGrid.asList()
    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodList]
    ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates]

    if foodDistances:
        closestFoodDist = min(foodDistances)
    else:
        closestFoodDist = 1 

    if ghostDistances:
        closestGhostDist = min(ghostDistances)
    else:
        closestGhostDist = 1  

    foodScore = -1.5 * closestFoodDist
    ghostScore = 2 * (1.0 / closestGhostDist if closestGhostDist > 0 else 100)

    scaredScore = sum(scaredTimes)

    finalScore = currentGameState.getScore() + foodScore + ghostScore + scaredScore

    return finalScore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
