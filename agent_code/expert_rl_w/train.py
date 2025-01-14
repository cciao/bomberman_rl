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
from .agent_setting import (SAVE_MODEL, TRAINING_ROUNDS,
                           BATCH_SIZE, TRANSITION_BUFFER_SIZE_MAX,
                           DECAY_GAMMA, EPSILON_END, EPSILON_DECAY,
                           N_STEP_LEARNING, N_STEP, PRIORITY_LEARNING, PRIORITY_RATIO,
                           EVALUATION_PLOT)

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
loop_old = 0
# Events
BOMBED_1_TO_2_CRATES = "BOMBED_1_TO_2_CRATES"
BOMBED_3_TO_5_CRATES = "BOMBED_3_TO_5_CRATES"
BOMBED_5_PLUS_CRATES = "BOMBED_5_PLUS_CRATES"
GET_IN_LOOP = "GET_IN_LOOP"
PLACEHOLDER_EVENT = "PLACEHOLDER"
ESCAPE = "ESCAPE"
NOT_ESCAPE = "NOT_ESCAPE"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
AWAY_FROM_CRATE = "AWAY_FROM_CRATE"
SURVIVED_STEP = "SURVIVED_STEP"
DESTROY_TARGET = "DESTROY_TARGET"
MISSED_TARGET = "MISSED_TARGET"
WAITED_NECESSARILY = "WAITED_NECESSARILY"
WAITED_UNNECESSARILY = "WAITED_UNNECESSARILY"
CLOSER_TO_PLAYERS = "CLOSER_TO_PLAYERS"
AWAY_FROM_PLAYERS = "AWAY_FROM_PLAYERS"



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_BUFFER_SIZE_MAX)
    self.n_step_buffer = deque(maxlen=N_STEP)
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
        if N_STEP_LEARNING:
            self.n_step_buffer.append(Transition(old_feature, self_action, next_feature, reward_from_events(self, new_events)))
            if len(self.n_step_buffer) >= N_STEP:
                reward_arr = np.array([self.n_step_buffer[i][-1] for i in range(N_STEP)])
                # Sum with the discount factor to get the accumulated rewards over N_STEP transitions.
                reward = ((DECAY_GAMMA) ** np.arange(N_STEP)).dot(reward_arr)
                self.transitions.append(
                    Transition(self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[-1][2], reward))
        else:
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
    #self.logger.debug(f'end of round {last_game_state is None}, {last_action is None} in final step')
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
        if N_STEP_LEARNING:
            self.n_step_buffer.append(Transition(last_feature, last_action, None, reward_from_events(self, new_events)))
            if len(self.n_step_buffer) >= N_STEP:
                reward_arr = np.array([self.n_step_buffer[i][-1] for i in range(N_STEP)])
                # Sum with the discount factor to get the accumulated rewards over N_STEP transitions.
                reward = ((DECAY_GAMMA) ** np.arange(N_STEP)).dot(reward_arr)
                self.transitions.append(
                    Transition(self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[-1][2], reward))
        else:
            self.transitions.append(Transition(last_feature, last_action, None, reward_from_events(self, new_events)))

    # train and update the model
    if len(self.transitions) > BATCH_SIZE:
        # mini_batch sampling
        batch = random.sample(self.transitions, BATCH_SIZE)
        # Initialization.
        X = [[] for i in range(len(self.actions))]  # Feature matrix for each action
        y = [[] for i in range(len(self.actions))]  # Target vector for each action
        residuals = [[] for i in range(len(self.actions))]
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

                # Prioritized experience replay.
                if PRIORITY_LEARNING and self.model_is_fitted:
                    # Calculate the residuals for the training instance.
                    X_tmp = X[action_idx][-1].reshape(1, -1)
                    target = y[action_idx][-1]
                    q_estimate = self.model.predict(X_tmp, action_idx=action_idx)[0]
                    res = (target - q_estimate) ** 2
                    residuals[action_idx].append(res)
        # update model
        if PRIORITY_LEARNING and self.model_is_fitted:
            # Initialization
            X_new = [[] for i in range(len(self.actions))]
            y_new = [[] for i in range(len(self.actions))]
            # For the training set of every action:
            for i in range(len(self.actions)):
                # Keep the specifed fraction of samples with the largest squared residuals.
                prio_size = int(len(residuals[i]) * PRIORITY_RATIO)
                idx = np.argpartition(residuals[i], -prio_size)[-prio_size:]
                X_new[i] = [X[i][j] for j in list(idx)]
                y_new[i] = [y[i][j] for j in list(idx)]
            # Update the training set.
            #self.logger.info(f'priority learning data: original: {[np.array(X[i]).shape for i in range(len(self.actions))]}, priority :{[np.array(X_new[i]).shape for i in range(len(self.actions))]}')
            X = X_new
            y = y_new

        self.logger.info(f'current round training data info, priority learning: {PRIORITY_LEARNING}, {[np.array(X[i]).shape for i in range(len(self.actions))]}')
        #self.logger.info(f'{np.array(X[0])}')
        self.model.partial_fit(X, y)
        self.model_is_fitted = True
        #if self.game_nr % 99 == 0:
        #    self.logger.info(f'training data print: {X}')
        #    self.logger.info(f'training data print: {y}')
        # update greedy epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
        self.logger.info(f"Training policy e-greedy:{self.epsilon}, n_step: {N_STEP_LEARNING}")

    evaluate_collection(self, new_events, True)
    evaluate_offline(self, last_feature, last_action, events, True)

    if not last_game_state["round"] % SAVE_MODEL:
        with open("./models/model-"+ str(last_game_state["round"]), "wb") as file:
            pickle.dump(self.model, file)

    if last_game_state["round"] == TRAINING_ROUNDS:
        with open("./models/model-"+ str(last_game_state["round"]), "wb") as file:
            pickle.dump(self.model, file)

    # evaluate from json.



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    global loop_old
    # Base rewards:
    aggressive_action = 0.3
    coin_action = 0.2
    crate_action = 0.05
    escape = 0.6
    bombing = 0.5
    waiting = 0.5
    passive = 0

    game_rewards = {
        # SPECIAL EVENTS
        ESCAPE: escape,
        NOT_ESCAPE: -escape,
        DESTROY_TARGET: bombing,
        MISSED_TARGET: -4 * escape,  # Prevent self-bomb
        BOMBED_1_TO_2_CRATES: 0.1,
        BOMBED_3_TO_5_CRATES: 0.3,
        BOMBED_5_PLUS_CRATES: 0.5,
        WAITED_NECESSARILY: waiting,
        WAITED_UNNECESSARILY: -waiting,
        CLOSER_TO_PLAYERS: aggressive_action,
        AWAY_FROM_PLAYERS: -aggressive_action,
        CLOSER_TO_COIN: coin_action,
        AWAY_FROM_COIN: -coin_action,
        CLOSER_TO_CRATE: crate_action,
        AWAY_FROM_CRATE: -crate_action,
        #GET_IN_LOOP: 0,
        GET_IN_LOOP : -0.025 * loop_old,
        SURVIVED_STEP: passive,

        # DEFAULT EVENTS
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
        e.CRATE_DESTROYED: 0.4,
        e.COIN_FOUND: 0,
        e.COIN_COLLECTED: 2,

        # kills
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -10,
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
    global loop_old
    events = copy.deepcopy(events_src)
    danger = old_feature[0] == 1

    if danger:
        # When in mortal danger, we only care about running away. 
        # Following the direction of escape will be rewarded and everything else will be punished.
        escape_direction = old_feature[2:4]
        if check_direction(escape_direction, self_action):
            events.append(ESCAPE)
        else:
            events.append(NOT_ESCAPE)

    else:
        # NOT DANGER
        # We have less pressure to make the right decisions without fatal risks.

        # Extracting information from the old game state.
        target_location = old_feature[1]
        players_location = old_feature[4:6]
        coins_location = old_feature[6:8]
        crates_location = old_feature[8:10]
        current_position = old_feature[10]
        loop_old = old_feature[11]

        # If the agent gets caught in a loop, he will be punished.
        if loop_old > 2:
            events.append(GET_IN_LOOP)

        # Analyze the explosion of different numbers of crates.
        if self_action == 'BOMB':
            if current_position == 1:
                events.append(BOMBED_1_TO_2_CRATES)
            elif current_position == 2:
                events.append(BOMBED_3_TO_5_CRATES)
            elif current_position == 3:
                events.append(BOMBED_5_PLUS_CRATES)

            # Reward bombing target.
            if target_location == 1:
                events.append(DESTROY_TARGET)
            else:
                events.append(MISSED_TARGET)

        elif self_action == 'WAIT':
            # Reward the agent if waiting is necessary.
            if (all(players_location == (0, 0)) and all(coins_location == (0, 0)) and all(crates_location == (0, 0)) and
                    target_location == 0):
                events.append(WAITED_NECESSARILY)
            else:
                events.append(WAITED_UNNECESSARILY)

        else:
            # If the current location should place a bomb and the agent does not place a bomb, the agent will be punished.
            if target_location == 1:
                events.append(MISSED_TARGET)

            # Reward or punish the actions of moving to players, coins and crates respectively.
            if not all(players_location == (0, 0)):
                if check_direction(players_location, self_action):
                    events.append(CLOSER_TO_PLAYERS)
                else:
                    events.append(AWAY_FROM_PLAYERS)
            if not all(coins_location == (0, 0)):
                if check_direction(coins_location, self_action):
                    events.append(CLOSER_TO_COIN)
                else:
                    events.append(AWAY_FROM_COIN)

            if not all(crates_location == (0, 0)):
                if check_direction(crates_location, self_action):
                    events.append(CLOSER_TO_CRATE)
                else:
                    events.append(AWAY_FROM_CRATE)
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

            with open("./models/evaluation-data", "wb") as file:
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


