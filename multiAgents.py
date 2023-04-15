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


from builtins import range
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        # Create food coordinates based on food grid
        currFood = currentGameState.getFood()

        # Get distances from Pacman to food
        foodDistances = []
        hitFood = False
        closestFood = 0.01 # set default to really small number (used in inverse, so cannot be 0)
        for y in range(0, successorGameState.data.layout.height):
          for x in range(0, successorGameState.data.layout.width):
            # If action causes Pacman to eat food, instead break loop since action is confirmed ideal
            if newFood[x][y] != currFood[x][y]:
              hitFood = True
              break

            elif newFood[x][y] == True:
              foodDistances.append(manhattanDistance(newPos, (x, y)))
        
        # Get distances from Pacman to ghosts
        ghostDistances = []
        for ghost in newGhostStates:
          ghostDistances.append(manhattanDistance(newPos, ghost.configuration.pos))

        # Define closest ghost and food
        closestGhost = min(ghostDistances)

        # If food wasn't hit, set closest to min value
        if hitFood == False:
          closestFood = min(foodDistances)

        # Initialize evaluation
        evaluation = 0

        # If ghost is far away, focus on food
        if closestGhost > 6:
          evaluation = (2.0 / closestFood)

        # If ghost is somewhat close, include that distance in evaluation
        elif closestGhost > 3:
          evaluation = (1.0 / closestFood) + closestGhost

        # If ghost is really close, focus on getting away
        else:
          evaluation = (2.0 * closestGhost)

        # Return evaluation 
        return evaluation

def scoreEvaluationFunction(currentGameState):
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
    # Actual search agent for MiniMax
    def miniMax(self, gameState, agentIndex, maxAgents, currDepth, targetDepth):
      
      # Get legal actions
      agentActions = gameState.getLegalActions(agentIndex)

      # Terminal state or target depth reached
      if (len(agentActions) == 0) or (currDepth == targetDepth):
        return self.evaluationFunction(gameState)

      # Normal Pacman turn: return max heuristic of every successor
      if agentIndex == 0:
        # Initialize scores for comparison
        scores = []

        # Get scores based on actions
        for action in agentActions:
          scores.append(self.miniMax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, maxAgents, currDepth, targetDepth))

        # If done searching, return action
        if currDepth == 0:
          bestScore = max(scores)
          for i in range(0, len(scores)):
            if scores[i] == bestScore:
              return agentActions[i]
        
        # Otherwise, return best score
        else:
          return max(scores)

      # Normal Ghost turn: return min heuristic of every successor
      else:
        # Initialize scores for comparison
        scores = []

        # Determine heuristics for each action
        if agentIndex != (maxAgents - 1): # If current ghost isn't last agent
          for action in agentActions:
            scores.append(self.miniMax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, maxAgents, currDepth, targetDepth))
        
        else: # If current ghost is last agent
          for action in agentActions:
            scores.append(self.miniMax(gameState.generateSuccessor(agentIndex, action), 0, maxAgents, currDepth + 1, targetDepth))

        # Find and return min score
          # (Only return score, Ghost action not needed and Pacman action is determined in getAction)
        return min(scores)

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        # Return action
        return self.miniMax(gameState, 0, gameState.getNumAgents(), 0, self.depth)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    # Actual search agent for AlphaBeta
    def alphaBeta(self, gameState, agentIndex, maxAgents, currDepth, targetDepth, alpha, beta):
      
      # Get legal actions
      agentActions = gameState.getLegalActions(agentIndex)

      # Terminal state or target depth reached
      if (len(agentActions) == 0) or (currDepth == targetDepth):
        return self.evaluationFunction(gameState)

      # Normal Pacman turn: return max heuristic of every successor
      if agentIndex == 0:
        scores = [] # Keep track of scores in case no obvious best

        # Determine heuristics for each action
        bestScore = -50000
        for action in agentActions:
          heuristic = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), 1, maxAgents, currDepth, targetDepth, alpha, beta)
          scores.append(heuristic)
          bestScore = max(heuristic, bestScore)
          if bestScore > beta:
            if currDepth == 0:
              return action
            
            return bestScore
          alpha = max(bestScore, alpha)
        
        # If no obvious best found, return best of okay actions
        if currDepth == 0:
          bestScore = max(scores)
          for i in range(0, len(scores)):
            if scores[i] == bestScore:
              return agentActions[i]
        
        # If no obvious best found, return best of okay scores
        else:
          return bestScore

      # Normal Ghost turn: return min heuristic of every successor
      else:
       # Determine heuristics for each action
        bestScore = 50000

        if agentIndex != (maxAgents - 1): # If current ghost isn't last agent     
          for action in agentActions:
            heuristic = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, maxAgents, currDepth, targetDepth, alpha, beta)
            bestScore = min(heuristic, bestScore)
            if bestScore < alpha:
              break
            beta = min(bestScore, beta)
        
        else: # If current ghost is last agent
          for action in agentActions:
            heuristic = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), 0, maxAgents, currDepth + 1, targetDepth, alpha, beta)
            bestScore = min(heuristic, bestScore)
            if bestScore < alpha:
              break
            beta = min(bestScore, beta)
        
        return bestScore

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestScore = -50000
        alpha = -50000
        beta = 50000
        
        # Return action
        return self.alphaBeta(gameState, 0, gameState.getNumAgents(), 0, self.depth, alpha, beta)
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    # Actual search agent for ExpectiMax
    def expectiMax(self, gameState, agentIndex, maxAgents, currDepth, targetDepth):
      
      # Get legal actions
      agentActions = gameState.getLegalActions(agentIndex)

      # Terminal state or target depth reached
      if (len(agentActions) == 0) or (currDepth == targetDepth):
        return self.evaluationFunction(gameState)

      # Normal Pacman turn: Find max score like before
      if agentIndex == 0:
        scores = [] # Keep track of scores in case no obvious best

        # Determine heuristics for each action
        bestScore = -50000
        for action in agentActions:
          scores.append(self.expectiMax(gameState.generateSuccessor(agentIndex, action), 1, maxAgents, currDepth, targetDepth))
        
        # If done searching, return best action
        if currDepth == 0:
          bestScore = max(scores)
          for i in range(0, len(scores)):
            if scores[i] == bestScore:
              return agentActions[i]
        
        # Otherwise, return best score
        else:
          return max(scores)

      # Normal Ghost turn: average all scores and return
      else:
        numActions = len(agentActions) * 1.0 # Get number of actions for calculating average
        totalScore = 0.0

        if agentIndex != (maxAgents - 1): # If current ghost isn't last agent     
          for action in agentActions:
            totalScore += 1.0 * self.expectiMax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, maxAgents, currDepth, targetDepth)
        
        else: # If current ghost is last agent
          for action in agentActions:
            totalScore += 1.0 * self.expectiMax(gameState.generateSuccessor(agentIndex, action), 0, maxAgents, currDepth + 1, targetDepth)
                    
        # Return average of scores
        return (totalScore / numActions)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Return action
        return self.expectiMax(gameState, 0, gameState.getNumAgents(), 0, self.depth)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This evaluation function calculates an evaluation for a current state based
      on current score, proximity to closest ghost, and proximity to closest piece of food. This
      is done by using the current score as a base value and adding different values to it based
      on the distance from the closest ghost. For example, a far away ghost will have no impact
      on the evaluation while one that is very close is the main factor in the final evaluation.
      
      This function is based on the evaluation used in the ReflexAgent, but has some differences
      in the specific ratios and base values used, making it work for states rather than actions.
    """
    "*** YOUR CODE HERE ***"
    # Get states for evaluation
    pacmanPosition = currentGameState.getPacmanPosition()
    foodStates = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    # Initialize distance lists
    foodDistances = []
    ghostDistances = []

    # Get distances from Pacman to food
    for y in range(0, currentGameState.data.layout.height):
      for x in range(0, currentGameState.data.layout.width):
        if foodStates[x][y] == True:
          foodDistances.append(manhattanDistance(pacmanPosition, (x, y)))

    # Get distances from Pacman to ghosts
    for ghost in ghostStates:
      ghostDistances.append(manhattanDistance(pacmanPosition, ghost.configuration.pos))

    # Use current score as a base
    evaluation = currentGameState.getScore()

    # Then, assuming there is at least one ghost and one piece of food...
    if (len(foodDistances) > 0) and (len(ghostDistances) > 0):
      
      # Define closest ghost and food
      closestGhost = min(ghostDistances)
      closestFood = min(foodDistances)

      # If closest food is 0, set closest to really small number
      if closestFood == 0:
        closestFood = 0.01

      # If ghost is far away, focus on food
      if closestGhost > 6:
        evaluation += (2.0 / closestFood)

      # If ghost is somewhat close, include that distance in evaluation
      elif closestGhost > 3:
        evaluation += (0.7 * (2.0 / closestFood)) + (0.3 * closestGhost)

      # If ghost is really close, focus on getting away, but also look for food
      else:
        evaluation += (0.25 * (2.0 / closestFood)) + (0.75 * closestGhost)

    # Return evaluation 
    return evaluation
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

