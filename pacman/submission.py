

import random, util, numpy
from game import Agent


#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return scoreEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
  """

  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """
  return gameState.getScore() #TODO: IMPLEMENT this heuristic!!!

#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################

def getNextIndexAgent(agentIndex, gameState):
  return (agentIndex + 1) % gameState.getNumAgents()

# c: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    legalMoves = gameState.getLegalActions()

    states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]

    min_max_values = [self.getMinMaxValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth) for state in states]
    bestScore = max(min_max_values)
    bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
    chosenIndex = random.choice(bestIndices)
    return legalMoves[chosenIndex]


  def getMinMaxValue(self, gameState, agentIndex, depth):
      if (agentIndex == 0 and depth == 1) or gameState.isWin() or gameState.isLose():
          return betterEvaluationFunction(gameState)

      legalMoves = gameState.getLegalActions(agentIndex)

      if agentIndex == 0:
          return max(self.getMinMaxValue(state, getNextIndexAgent(agentIndex, gameState), depth - 1) for state in [gameState.generatePacmanSuccessor(action) for action in legalMoves])
      else:
          return min(self.getMinMaxValue(state, getNextIndexAgent(agentIndex, gameState), depth) for state in [gameState.generateSuccessor(agentIndex, action) for action in legalMoves])




######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    a = - numpy.inf
    b = numpy.inf

    legalMoves = gameState.getLegalActions()

    states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]

    min_max_values = [self.getAlphaBetaValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth, a, b) for state in
                    states]
    bestScore = max(min_max_values)
    bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
    chosenIndex = random.choice(bestIndices)
    return legalMoves[chosenIndex]

  def getAlphaBetaValue(self, gameState, agentIndex, depth, a, b):
    if (agentIndex == 0 and depth == 1) or gameState.isWin() or gameState.isLose():
      self.function = betterEvaluationFunction
      return self.function(gameState)

    legalMoves = gameState.getLegalActions(agentIndex)

    if agentIndex == 0:
      current_max = - numpy.inf
      for state in [gameState.generatePacmanSuccessor(action) for action in legalMoves]:
          value = self.getAlphaBetaValue(state, getNextIndexAgent(agentIndex, gameState), depth - 1, a, b)
          current_max = max(current_max, value)
          a = max(current_max, a)
          if current_max >= b:
              return numpy.inf
      return current_max

    else:
      current_min = numpy.inf
      for state in [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]:
          value = self.getAlphaBetaValue(state, getNextIndexAgent(agentIndex, gameState), depth, a, b)
          if value == - numpy.inf:
              return - numpy.inf
          current_min = min(current_min, value)
          b = min(current_min, b)
          if current_min <= a:
              return - numpy.inf
      return current_min



######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

    legalMoves = gameState.getLegalActions()

    states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]

    min_max_values = [self.getExpectimaxValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth) for state
                      in states]
    bestScore = max(min_max_values)
    bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
    chosenIndex = random.choice(bestIndices)
    return legalMoves[chosenIndex]

  def getExpectimaxValue(self, gameState, agentIndex, depth):
      if (agentIndex == 0 and depth == 1) or gameState.isWin() or gameState.isLose():
          return betterEvaluationFunction(gameState)

      legalMoves = gameState.getLegalActions(agentIndex)

      if agentIndex == 0:
          return max(self.getExpectimaxValue(state, getNextIndexAgent(agentIndex, gameState), depth - 1) for state in
                     [gameState.generatePacmanSuccessor(action) for action in legalMoves])
      else:
          return numpy.average(list((self.getExpectimaxValue(state, getNextIndexAgent(agentIndex, gameState), depth) for state in
                     [gameState.generateSuccessor(agentIndex, action) for action in legalMoves])))


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """

  def getAction(self, gameState):
    """
      Returns the action using self.depth and self.evaluationFunction

    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE



