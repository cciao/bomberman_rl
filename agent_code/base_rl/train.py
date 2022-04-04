from collections import namedtuple, deque, Counter
import copy

import pickle
from typing import List
import random

import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-GUI backend, needed for plotting in non-main thread.
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1.0

import events as e
import settings as s
from .lib import state_to_features, check_direction
from .agent_setting import (ACTIONS, NUMBER_OF_FEATURE, NUMBER_OF_ACTIONS,
                           UPDATE_TARGET_MODEL, SAVE_MODEL, TRAINING_ROUNDS,
                           BATCH_SIZE, TRANSITION_BUFFER_SIZE_MAX,
                           STEP_ALPHA, DECAY_GAMMA,
                           EVALUATION_PLOT)

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
CLOSER_TO_ESCAPE = "CLOSER_TO_ESCAPE"
FURTHER_FROM_ESCAPE = "FURTHER_FROM_ESCAPE"
BOMBED_GOAL = "BOMBED_GOAL"
MISSED_GOAL = "MISSED_GOAL"
WAITED_NECESSARILY = "WAITED_NECESSARILY"
WAITED_UNNECESSARILY = "WAITED_UNNECESSARILY"
CLOSER_TO_OTHERS = "CLOSER_TO_OTHERS"
FURTHER_FROM_OTHERS = "FURTHER_FROM_OTHERS"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"
CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
FURTHER_FROM_CRATE = "FURTHER_FROM_CRATE"
SURVIVED_STEP = "SURVIVED_STEP"
BOMBED_1_TO_2_CRATES = "BOMBED_1_TO_2_CRATES"
BOMBED_3_TO_5_CRATES = "BOMBED_3_TO_5_CRATES"
BOMBED_5_PLUS_CRATES = "BOMBED_5_PLUS_CRATES"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_BUFFER_SIZE_MAX)
    # evaluation initialization
    self.game_nr = 0
    self.historic_data = {
        'rewards': [],
        'coins': [],
        'crates': [],
        'enemies': [],
        'exploration': [],
        'games': [],
        'offline_acc': [],
        'offline_rewards': []
    }
    # Initialization
    self.rewards    = 0
    self.coins   = 0
    self.crates  = 0
    self.enemies    = 0
    self.exploration = 0
    self.model_action = []
    self.online_action = []
    self.model_rewards = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards

    # state_to_features is defined in callbacks.py
    new_events = events
    self_coor = None
    if self_action is not None:
        self_coor = self.coordinate_history.pop()
    old_feature = state_to_features(old_game_state, self.coordinate_history)
    if self_coor is not None:
        self.coordinate_history.append(self_coor)
    if old_game_state is not None and self_action is not None:
        next_feature = state_to_features(new_game_state, self.coordinate_history)
        new_events = get_events(old_feature, self_action, events)
        self.transitions.append(Transition(old_feature, self_action, next_feature, reward_from_events(self, new_events)))

    # collect evaluation data
    evaluate_collection(self, new_events, False)
    evaluate_offline(self, old_feature, self_action, events, False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.logger.debug(f'end of round {last_game_state is None}, {last_action is None} in final step')
    self.game_nr += 1

    new_events = events
    self_coor = None
    if last_action is not None:
        self_coor = self.coordinate_history.pop()
    last_feature = state_to_features(last_game_state, self.coordinate_history)
    if self_coor is not None:
        self.coordinate_history.append(self_coor)
    #last_feature = state_to_features(last_game_state)
    if last_game_state is not None and last_action is not None:
        new_events = get_events(last_feature, last_action, events)
        self.transitions.append(Transition(last_feature, last_action, None, reward_from_events(self, new_events)))

    # train and update the model
    if len(self.transitions) > BATCH_SIZE:
        # mini_batch sampling
        #batch = []
        #for i in range(len(self.transitions)):
        #    if self.transitions[i].action == 'WAIT':
        #        batch.append(self.transitions[i])
        batch = random.sample(self.transitions, BATCH_SIZE)
        # Initialization.
        X = [[] for i in range(len(self.actions))]  # Feature matrix for each action
        y = [[] for i in range(len(self.actions))]  # Target vector for each action
        for old_state, action, next_state, reward in batch:
            if old_state is not None:
                # Index of action taken in 'state'.
                action_idx = self.actions.index(action)
                # Q-value for the given state and action.
                if self.model_is_fitted and next_state is not None:
                    # Non-terminal next state and pre-existing model.
                    maximal_response = np.max(self.model.predict(next_state))
                    q_update = (reward + DECAY_GAMMA * maximal_response)
                else:
                    # Either next state is terminal or a model is not yet fitted.
                    q_update = reward # Equivalent to a Q-value of zero for the next state.

                # Append feature data and targets for the regression,
                # corresponding to the current action.
                X[action_idx].append(old_state)
                y[action_idx].append(q_update)
        # update model
        self.logger.info(f'current round training data info{[np.array(X[i]).shape for i in range(len(self.actions))]}')
        #self.logger.info(np.array(X[0]))
        self.model.partial_fit(X, y)
        self.model_is_fitted = True
        #if self.game_nr % 99 == 0:
        #    self.logger.info(f'training data print: {X}')
        #    self.logger.info(f'training data print: {y}')
    evaluate_collection(self, new_events, True)
    evaluate_offline(self, last_feature, last_action, events, True)

    if not last_game_state["round"] % SAVE_MODEL:
        with open("./models/model-"+ str(last_game_state["round"]), "wb") as file:
            pickle.dump(self.model, file)

    if last_game_state["round"] == TRAINING_ROUNDS:
        with open("./models/model-"+ str(last_game_state["round"]), "wb") as file:
            pickle.dump(self.model, file)


def reward_from_events_dump(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    # escape > kill > coin > crate

    # Base rewards:
    kill = 5
    coin = 2
    crate = 0.2

    escape_movement = 0.5
    bombing = 0.5
    waiting = 0.5
    offensive_movement = 0.25
    coin_movement = 0.1
    crate_movement = 0.05

    passive = 0

    # Game reward dictionary:
    game_rewards = {
        # ---- CUSTOM EVENTS ----
        # escape movement
        CLOSER_TO_ESCAPE: escape_movement,
        FURTHER_FROM_ESCAPE: -escape_movement,

        # bombing
        BOMBED_GOAL: bombing,
        MISSED_GOAL: -4 * escape_movement,  # Needed to prevent self-bomb-laying loops.
        BOMBED_1_TO_2_CRATES: 0.5,
        BOMBED_3_TO_5_CRATES: 0.3,
        BOMBED_5_PLUS_CRATES: 0.1,

        # waiting
        WAITED_NECESSARILY: waiting,
        WAITED_UNNECESSARILY: -1,

        # offensive movement
        CLOSER_TO_OTHERS: offensive_movement,
        FURTHER_FROM_OTHERS: -offensive_movement,

        # coin movement
        CLOSER_TO_COIN: 5*coin_movement,
        FURTHER_FROM_COIN: -coin_movement,

        # crate movement
        CLOSER_TO_CRATE: crate_movement,
        FURTHER_FROM_CRATE: -crate_movement,

        # passive
        SURVIVED_STEP: passive,

        # ---- DEFAULT EVENTS ----
        # movement
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,
        e.INVALID_ACTION: -1,

        # bombing
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,

        # crates, coins
        e.CRATE_DESTROYED: 0.2,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 2,

        # kills
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -5,
        e.OPPONENT_ELIMINATED: 0,

        # passive
        e.SURVIVED_ROUND: 0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def get_events(old_feature, self_action, events_src)->list:
    """
    get events
    """
    events = copy.deepcopy(events_src)
    # Extract the lethal indicator from the old state.
    islethal_old = old_feature[0] == 1

    if islethal_old:
        # ---- WHEN IN LETHAL ----
        # When in lethal danger, we only care about escaping. Following the
        # escape direction is rewarded, anything else is penalized.
        escape_dir_old = old_feature[2:4]
        if check_direction(escape_dir_old, self_action):
            events.append(CLOSER_TO_ESCAPE)
        else:
            events.append(FURTHER_FROM_ESCAPE)

    else:
        # ---- WHEN IN NON-LETHAL ----
        # When not in lethal danger, we are less stressed to make the right
        # decision. Our order of prioritization is: others > coins > crates.

        # Extracting information from the old game state.
        target_acq_old = old_feature[1]
        others_dir_old = old_feature[4:6]
        coins_dir_old = old_feature[6:8]
        crates_dir_old = old_feature[8:10]
        current_position_old = old_feature[10]


        # If we chose to bomb in the previous state:
        if self_action == 'BOMB':
            if current_position_old == 1:
                events.append(BOMBED_1_TO_2_CRATES)
            elif current_position_old == 2:
                events.append(BOMBED_3_TO_5_CRATES)
            elif current_position_old == 3:
                events.append(BOMBED_5_PLUS_CRATES)
            # Reward if we successfully bombed the target, else penalize.
            if target_acq_old == 1:
                events.append(BOMBED_GOAL)
            else:
                events.append(MISSED_GOAL)


        elif self_action == 'WAIT':
            # Reward the agent if the waiting was neccesary, but penalize
            # if a direction for others, coins or crates was suggested.
            if (all(others_dir_old == (0, 0)) and
                    all(coins_dir_old == (0, 0)) and
                    all(crates_dir_old == (0, 0)) and
                    target_acq_old == 0):
                events.append(WAITED_NECESSARILY)
            else:
                events.append(WAITED_UNNECESSARILY)
        # Reward the agent if the waiting was neccesary, but penalize
        # if a direction for others, coins or crates was suggested.

        # If we chose to move in the previous state:
        else:
            # Penalize if we were standing at the bomb goal, but moved.
            if target_acq_old == 1:
                events.append(MISSED_GOAL)

            # Movement priority: others > coins > crates.

            # Reward/penalize for moving towards/away from the offensive target.
            if not all(others_dir_old == (0, 0)):
                if check_direction(others_dir_old, self_action):
                    events.append(CLOSER_TO_OTHERS)
                else:
                    events.append(FURTHER_FROM_OTHERS)
            elif not all(coins_dir_old == (0, 0)):
                if check_direction(coins_dir_old, self_action):
                    events.append(CLOSER_TO_COIN)
                else:
                    events.append(FURTHER_FROM_COIN)

                # Reward/penalize for moving towards/away from the crate target.
            elif not all(crates_dir_old == (0, 0)):
                if check_direction(crates_dir_old, self_action):
                    events.append(CLOSER_TO_CRATE)
                else:
                    events.append(FURTHER_FROM_CRATE)
    if not 'GOT_KILLED' in events:
        events.append(SURVIVED_STEP)
    return events

def evaluate_collection(self, events: List[str], data_reset=False):
    """
    collect the game data
    """
    # Get the numbers from the last round.
    if 'COIN_COLLECTED' in events:
        self.coins += 1
    if 'CRATE_DESTROYED' in events:
        self.crates += events.count('CRATE_DESTROYED')
    if 'KILLED_OPPONENT' in events:
        self.enemies += events.count('KILLED_OPPONENT')
    self.rewards += reward_from_events(self, events)

    # Total score in this game.
    #score = np.sum(self.score_in_round)

    if data_reset:
        # Append results to each specific list.
        self.historic_data['rewards'].append(self.rewards)
        self.historic_data['coins'].append(self.coins)
        self.historic_data['crates'].append(self.crates)
        self.historic_data['enemies'].append(self.enemies)
        self.historic_data['games'].append(self.game_nr)
        self.rewards = 0
        self.coins = 0
        self.crates = 0
        self.enemies = 0
        self.exploration = 0

        # Plot training progress every n:th game.
        if self.game_nr % EVALUATION_PLOT == 0:
            # Incorporate the full training history.
            games_list = self.historic_data['games']
            reward_list = self.historic_data['rewards']
            coins_list = self.historic_data['coins']
            crate_list = self.historic_data['crates']
            other_list = self.historic_data['enemies']
            explr_list = self.historic_data['exploration']

            # Plotting
            fig, ax = plt.subplots(4, figsize=(7.2, 5.4), sharex=True)

            # Total score per game.
            ax[0].plot(games_list, reward_list)
            ax[0].set_title('Total rewards per game')
            ax[0].set_ylabel('Rewards')

            # Collected coins per game.
            ax[1].plot(games_list, coins_list)
            ax[1].set_title('Collected coins per game')
            ax[1].set_ylabel('Coins')

            # Destroyed crates per game.
            ax[2].plot(games_list, crate_list)
            ax[2].set_title('Destroyed crates per game')
            ax[2].set_ylabel('Crates')

            # Eliminiated opponents per game
            ax[3].plot(games_list, other_list)
            ax[3].set_title('Eliminated opponents per game')
            ax[3].set_ylabel('Kills')

            # Export the figure.
            fig.tight_layout()
            plt.savefig(f'./models/plot.pdf')
            plt.close('all')

            with open("./models/evaluation", "wb") as file:
                pickle.dump(self.historic_data, file)


def evaluate_offline(self, old_feature, self_action, events, data_reset=False):
    """
    evaluate the model and rule-based model
    """
    if old_feature is not None and self_action is not None:
        self.model_action.append(old_feature)
        self.online_action.append(self.actions.index(self_action))
        self.model_rewards.append(events)
    if data_reset and self.model_is_fitted:
        N = len(self.online_action)
        predict_value = self.model.predict(np.array(self.model_action))
        model_action = np.argmax(predict_value, axis = 1)
        #if self.game_nr % 100 == 0:
        #    self.logger.info(f"actioninfo :{predict_value}, {self.online_action}")
        #self.logger.info(type(model_action), self.online_action, np.array(model_action) == np.array(self.online_action))
        acc =  round((np.array(model_action) == np.array(self.online_action)).sum() / N, 3)
        self.historic_data['offline_acc'].append(acc)
        rewards = np.array([reward_from_events(self, get_events(self.model_action[i], self.actions[model_action[i]], self.model_rewards[i])) for i in range(N)]).sum()
        self.historic_data['offline_rewards'].append(rewards)
        self.model_action = []
        self.online_action = []
        self.model_rewards = []
        self.logger.info(f"Round: {self.game_nr}, accuray info number of actions = {N}, accuracy = {acc}")

    #self.logger.info(f"Round {self.game_nr}: action {self_action}, reset {data_reset}, fitted {self.model_is_fitted}")
    if self.game_nr > 0 and self.game_nr % EVALUATION_PLOT == 0:
        fig, ax = plt.subplots(2, figsize=(7.2, 5.4), sharex=True)
        acc_list = self.historic_data['offline_acc']
        rewards_list = self.historic_data['offline_rewards']
        # accuracy.
        ax[0].plot(acc_list)
        ax[0].set_title('Model accuracy compared to rule-based agent')
        ax[0].set_ylabel('Accuracy')
        # rewards
        ax[1].plot(rewards_list)
        ax[1].set_title('Model rewards')
        ax[1].set_ylabel('Rewards')

        # Export the figure.
        fig.tight_layout()
        plt.savefig(f'./models/plot-acc.pdf')
        plt.close('all')


