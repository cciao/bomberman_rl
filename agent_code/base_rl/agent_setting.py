import events as e
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NUMBER_OF_ACTIONS = 6
NUMBER_OF_FEATURE = 12

# mini-batch
BATCH_SIZE = 512# Batch size should be <1% of the total experience buffer size
TRANSITION_BUFFER_SIZE_MAX = 1024  # Buffer must be a power of two to have a balanced tree

# Q-learning
STEP_ALPHA = 1
DECAY_GAMMA = 0.9

# epsilon greedy policy
EPSILON_START = 1  # 0.3 When using PER
EPSILON_END = 0.16
EPSILON_DECAY = 0.99999  # Diminishing Epsilon-Greedy

# models
SAVE_MODEL = 100
UPDATE_TARGET_MODEL = 10
TRAINING_ROUNDS=5000

#evaluation
EVALUATION_PLOT = 100

