import os
from collections import deque
import random

import numpy as np
import pickle
from sklearn.linear_model import SGDRegressor


from .agent_setting import ACTIONS, FINAL_MODEL_NAME, EPSILON_END, EPSILON_START
from .lib import CustomRegressor, state_to_features, get_valid_actions

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 7)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    #
    self.action_history = deque([], 20)
    self.actions = ACTIONS
    self.epsilon = EPSILON_START # 0.01
    # setup models
    model_file = os.path.join('./models', FINAL_MODEL_NAME)
    if os.path.isfile(model_file):
        self.logger.info("Loading model from saved state.")
        with open(model_file, "rb") as file:
            self.model = pickle.load(file)
        self.model_is_fitted = True
    if self.train:
        self.logger.info("Setting up model from scratch.")
        self.model = CustomRegressor(SGDRegressor(alpha=0.0001, warm_start=True), self)
        self.model_is_fitted = False

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 7)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.action_history = deque([], 20)



def act(self, game_state):
    """
    Picking action according to model
    Args:
        self:
        game_state:

    Returns:
    """
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    _, _, bombs_left, (x, y) = game_state['self']
    self.coordinate_history.append((x, y))
    mask, valid_actions = get_valid_actions(game_state)
    if self.train:
        random_prob = self.epsilon
    else:
        random_prob = EPSILON_END
    if random.random() < random_prob or not self.model_is_fitted:
        #execute_action = np.random.choice(valid_actions) if len(valid_actions) > 0 else "WAIT"
        execute_action = np.random.choice(valid_actions)
        #self.logger.debug(f"Choose action uniformly at random. valid action {valid_actions}, execute action {execute_action}")
    else:
        q_values = self.model.predict(state_to_features(game_state, self.coordinate_history))[0]
        execute_action = valid_actions[np.argmax(q_values[mask])]
        #self.logger.info(f'Choose action according to model: {self.actions[np.argmax(q_values)]}, valid execution action: {execute_action}')
        #self.logger.info(f'q_values: {q_values}, mask: {mask}')
    self.action_history.append((execute_action, x, y))
    return execute_action

