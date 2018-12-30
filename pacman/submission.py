
import random, util, numpy, math
from game import Agent
from ghostAgents import DirectionalGhost


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

class stateDate:

    def __init__(self):
        self.apathetic = False
        self.last_moves = []
        self.food_direction = [0, 0]

    def update(self, gameState, legalMoves):
        self.apathetic = self.isApatheticState(gameState, legalMoves)
        if self.apathetic:
            self.last_moves.append(gameState.getPacmanPosition())
            self.food_direction = self.getFoodDirection(gameState)
            if len(self.last_moves) > numpy.random.randint(0, len(self.last_moves) * 10):
                self.last_moves = []
        else:
            self.last_moves = []

    def isApatheticState(self, gameState, legalMoves, bound_value=5):
        """
          get pacman turn game state and return an answer if this state is apathetic.
          apathetic state: state that all the changes in the score of his successor states,
                           bound under @bound_value, meaning: there is no significantly  good or bad action for this state.
        """
        scores = [self.evaluate_score(gameState, action) for action in legalMoves]

        current_score = gameState.getScore()

        # list of all the changes in the score.
        diff_scores = list(map(lambda score: score - current_score, scores))

        # list of boolean values that indicate if the value is bounded under bound_value.
        diff_scores_boolean = list(map(lambda score: (-bound_value < score < bound_value), diff_scores))

        if False in diff_scores_boolean:
            return False  # the state is not apathetic
        else:
            return True  # the state is apathetic.

    def evaluate_score(self, currentGameState, action):
        """
        evaluate score for state after pacmen play.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return successorGameState.getScore()

    def getFoodDirection(self, gameState):
        """
        get nearest food vector according  to pacman position by iterate all food positions.
        """
        X_VALUE = 0
        Y_VALUE = 1
        pacman_position = gameState.getPacmanPosition()

        # initial values for loop.
        nearest_food_position = [0, 0]
        nearest_food_distance = gameState.data.layout.height * gameState.data.layout.width

        # find nearest_food_position according to minimal nearest_food_distance.
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                if gameState.hasFood(x,y):
                    distance = self.getDistanveFromFoodDirection(pacman_position, (x, y))
                    if distance < nearest_food_distance:
                        nearest_food_position = [x, y]

        return nearest_food_position

    def getDistanveFromFoodDirection(self, pacman_pos, food_pos):
        """
        calculate the (food_pos - pacman_pos) vector length which represent the distance between pacman and the nearest food.
        """
        X_VALUE = 0
        Y_VALUE = 1
        food_vector = [pacman_pos[X_VALUE] - food_pos[X_VALUE], pacman_pos[Y_VALUE] - food_pos[Y_VALUE]]
        return numpy.sqrt(food_vector[X_VALUE] ** 2 + food_vector[Y_VALUE] ** 2)


state_data = stateDate()

def betterEvaluationFunction(gameState):
  global state_data # type: stateDate
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

  score = gameState.getScore()

  if state_data.apathetic:
      position = gameState.getPacmanPosition()
      long_path = (gameState.data.layout.height + gameState.data.layout.width)
      distance_from_food = state_data.getDistanveFromFoodDirection(position, state_data.food_direction)

      # if we close to food add to score.
      score += 7 * ((long_path - distance_from_food) / long_path)

      # if we visit that position decremante 10 points from score.
      if position in state_data.last_moves:
          score -= 10

  return score


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
    legalMoves = gameState.getLegalActions(agentIndex=0)

    states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]

    state_data.update(gameState, legalMoves)

    min_max_values = [self.getMinMaxValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth) for state in states]
    bestScore = max(min_max_values)
    bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
    chosenIndex = random.choice(bestIndices)
    return legalMoves[chosenIndex]


  def getMinMaxValue(self, gameState, agentIndex, depth):
      if (agentIndex == 0 and depth == 1) or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

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

    state_data.update(gameState, legalMoves)

    min_max_values = [self.getAlphaBetaValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth, a, b) for state in states]
    bestScore = max(min_max_values)
    bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
    chosenIndex = random.choice(bestIndices)

    return legalMoves[chosenIndex]

  def getAlphaBetaValue(self, gameState, agentIndex, depth, a, b):
    if (agentIndex == 0 and depth == 1) or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

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

        state_data.update(gameState, legalMoves)

        min_max_values = [self.getExpectimaxValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth)
                          for state
                          in states]
        bestScore = max(min_max_values)
        bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def getExpectimaxValue(self, gameState, agentIndex, depth):
        if (agentIndex == 0 and depth == 1) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:
            return max(self.getExpectimaxValue(state, getNextIndexAgent(agentIndex, gameState), depth - 1) for state in
                       [gameState.generatePacmanSuccessor(action) for action in legalMoves])
        else:
            return numpy.average(
                list((self.getExpectimaxValue(state, getNextIndexAgent(agentIndex, gameState), depth) for state in
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

        legalMoves = gameState.getLegalActions()

        states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]

        state_data.update(gameState, legalMoves)

        min_max_values = [
            self.getDirectionalExpectimaxValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth) for
            state
            in states]
        bestScore = max(min_max_values)
        bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def getDirectionalExpectimaxValue(self, gameState, agentIndex, depth):
        if (agentIndex == 0 and depth == 1) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:
            return max(
                self.getDirectionalExpectimaxValue(state, getNextIndexAgent(agentIndex, gameState), depth - 1) for state in
                [gameState.generatePacmanSuccessor(action) for action in legalMoves])
        else:
            ghost = DirectionalGhost(index=agentIndex)
            act_prob_dict = ghost.getDistribution(gameState)
            val_prob_dict = util.Counter()
            for action in legalMoves:
                state = gameState.generateSuccessor(agentIndex, action)
                val = self.getDirectionalExpectimaxValue(state, getNextIndexAgent(agentIndex, gameState), depth)
                val_prob_dict[val] = act_prob_dict[action]
            val_prob_dict.normalize()
            return util.chooseFromDistribution(val_prob_dict)


######################################################################################
# I: implementing competition agent

class CompetitionAgent(RandomExpectimaxAgent):
    def __init__(self, evalFn = 'betterEvaluationFunction', depth='3'):
        super(self.__class__, self).__init__(evalFn, depth)