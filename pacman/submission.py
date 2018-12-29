

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
        self.food_directions_positions = []
        self.not_visited_yet_directions_positions = []
    def update(self, gameState, legalMoves):
        self.food_directions_positions = []
        self.not_visited_yet_directions_positions = []
        self.apathetic = self.isApatheticState(gameState, legalMoves)
        if self.apathetic:
            food_directions = self.getFavoriteDirectionByFood(gameState, legalMoves)
            not_visited_yet_directions = self.getFavoriteDirectionByCache(gameState, legalMoves)
            for action in legalMoves:
                if action not in food_directions and action not in not_visited_yet_directions:
                    continue
                state = gameState.generatePacmanSuccessor(action)
                if action in food_directions:
                    self.food_directions_positions.append(state.getPacmanPosition())
                if action in not_visited_yet_directions:
                    self.not_visited_yet_directions_positions.append(state.getPacmanPosition())
            if len(self.last_moves) > max(gameState.data.layout.width, gameState.data.layout.height) - 2:
                self.last_moves = []
            self.last_moves.append(gameState.getPacmanPosition())
            self.last_moves = self.last_moves
        else:
            self.last_moves = []
        # print(self.apathetic)
        # print(self.last_moves)
        # print(self.food_directions_positions)
        # print(self.not_visited_yet_directions_positions)

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

    def getFavoriteDirectionByFood(self, gameState, legalMoves):
        X_VALUE = 0
        Y_VALUE = 1
        favorite_direction = None
        """
        get favorite direction by calculating the food central mass vector direction. 
        """
        pacman_position = gameState.getPacmanPosition()
        num_food = 0
        food_Central_mass_vactor = [0,0]
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                if gameState.hasFood(x, y):
                    num_food += 1
                    food_Central_mass_vactor[X_VALUE] += x
                    food_Central_mass_vactor[Y_VALUE] += y


        # if the food_Central_mass_vector in pacman position.
        if food_Central_mass_vactor == [0, 0]:
            return favorite_direction

        food_Central_mass_vactor[X_VALUE] = food_Central_mass_vactor[X_VALUE] / num_food - pacman_position[X_VALUE]
        food_Central_mass_vactor[Y_VALUE] = food_Central_mass_vactor[Y_VALUE] / num_food - pacman_position[Y_VALUE]

        angel = numpy.degrees(

            math.atan2(food_Central_mass_vactor[Y_VALUE], food_Central_mass_vactor[X_VALUE])) % 360
        # print(angel, food_Central_mass_vactor)
        if 0 <= angel <= 45:
            favorite_direction = ['East', 'North']
        if 45 <= angel <= 90:
            favorite_direction = ['North', 'East']
        if 90 <= angel <= 135:
            favorite_direction = ['North', 'West']
        if 135 <= angel <= 180:
            favorite_direction = ['West', 'North']
        if 180 <= angel <= 225:
            favorite_direction = ['West', 'South']
        if 225 <= angel <= 270:
            favorite_direction = ['South', 'West']
        if 270 <= angel <= 315:
            favorite_direction = ['South', 'East']
        if 315 <= angel <= 360:
            favorite_direction = ['East', 'South']

        if favorite_direction[1] not in legalMoves:
            favorite_direction.remove(favorite_direction[1])
        if favorite_direction[0] not in legalMoves:
            favorite_direction.remove(favorite_direction[0])

        return favorite_direction


    def getFavoriteDirectionByCache(self, gameState, legalMoves):
        """
        return list of direction to places we haven't visited yet
        """
        states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]
        favorite_directions = []
        for state, action in zip(states, legalMoves):
            pacman_pos = state.getPacmanPosition()
            if pacman_pos not in self.last_moves:  # we didn't visit there yet
                favorite_directions.append(action)
        return favorite_directions


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
      if position in state_data.food_directions_positions:
          score += 3
      if position in state_data.not_visited_yet_directions_positions:
          score += 5
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

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '4'):
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

    min_max_values = [self.getMinMaxValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth) for state in states]
    bestScore = max(min_max_values)
    bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
    chosenIndex = random.choice(bestIndices)
    return legalMoves[chosenIndex]


  def getMinMaxValue(self, gameState, agentIndex, depth):
      if gameState.isWin() or gameState.isLose():
          return gameState.getScore()
      if agentIndex == 0 and depth == 1:
          legalMoves = gameState.getLegalActions(agentIndex)
          state_data.update(gameState, legalMoves)
          return max([self.evaluationFunction(state) for state in
                      [gameState.generatePacmanSuccessor(action) for action in legalMoves]])

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
    if gameState.isWin() or gameState.isLose():
      return gameState.getScore()
    if agentIndex == 0 and depth == 1:
      legalMoves = gameState.getLegalActions(agentIndex)
      state_data.update(gameState, legalMoves)
      return max([self.evaluationFunction(state) for state in [gameState.generatePacmanSuccessor(action) for action in legalMoves]])

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
      if gameState.isWin() or gameState.isLose():
          return gameState.getScore()
      if agentIndex == 0 and depth == 1:
          legalMoves = gameState.getLegalActions(agentIndex)
          state_data.update(gameState, legalMoves)
          return max([self.evaluationFunction(state) for state in
                      [gameState.generatePacmanSuccessor(action) for action in legalMoves]])

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

    legalMoves = gameState.getLegalActions()

    states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]

    min_max_values = [self.getDirectionalExpectimaxValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth) for
                      state
                      in states]
    bestScore = max(min_max_values)
    bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
    chosenIndex = random.choice(bestIndices)
    return legalMoves[chosenIndex]

  def getDirectionalExpectimaxValue(self, gameState, agentIndex, depth):
      if gameState.isWin() or gameState.isLose():
          return gameState.getScore()
      if agentIndex == 0 and depth == 1:
          legalMoves = gameState.getLegalActions(agentIndex)
          state_data.update(gameState, legalMoves)
          return max([self.evaluationFunction(state) for state in
                      [gameState.generatePacmanSuccessor(action) for action in legalMoves]])

      legalMoves = gameState.getLegalActions(agentIndex)

      if agentIndex == 0:
          return max(self.getDirectionalExpectimaxValue(state, getNextIndexAgent(agentIndex, gameState), depth - 1) for state in
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

class CompetitionAgent(AlphaBetaAgent):

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        a = - numpy.inf
        b = numpy.inf

        legalMoves = gameState.getLegalActions()

        state_data.update(gameState, legalMoves)

        states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]

        if state_data.apathetic:
            self.evaluationFunction = betterEvaluationFunction
            self.depth = 4
        else:
            self.evaluationFunction = scoreEvaluationFunction
            self.depth = 2
        min_max_values = [
            self.getAlphaBetaValue(state, getNextIndexAgent(agentIndex=0, gameState=gameState), self.depth, a, b) for state
            in
            states]
        bestScore = max(min_max_values)
        bestIndices = [index for index in range(len(min_max_values)) if min_max_values[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def getAlphaBetaValue(self, gameState, agentIndex, depth, a, b):
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
        if agentIndex == 0 and depth == 1:
            legalMoves = gameState.getLegalActions(agentIndex)
            state_data.update(gameState, legalMoves)
            return max([self.evaluationFunction(state) for state in
                        [gameState.generatePacmanSuccessor(action) for action in legalMoves]])

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





