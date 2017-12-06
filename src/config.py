# This file contains all the parameters associated with pirateAI.
# Many of the classes and files import from this.
# TODO: Make this a JSON or YAML?

# MODE = 'test'
MODE = 'train'
SHIP_NAME = 'Krusty Krab'

# Training (hyperopt)
NUM_PIRATES_PER_TRAIN = 3
MAX_TRAIN_TRIES = 3
NUM_PIRATES_PER_CULLING = 2
SALT_PER_WIN = 8
SALT_PER_LOSS = 1
STARTING_SALT = 100
TRAIN_PATIENCE = 3
LABEL_DICT = {0: 'W', 1: 'S', 2: 'A', 3: 'D'}

# Run duration
FULL_CYCLES = 100

# Ship
MAX_PIRATES_IN_SHIP = 20
MIN_PIRATES_IN_SHIP = 10

# Evaluation (marooning)
MAROON_CYCLES = 10
N_BEST_PIRATES = 10
BOUNTY = 1
MAX_ROUNDS = 3

# Connection information (for an external unity environment)
WINDOWS_IP = '192.168.2.3'
WINDOWS_PORT = 5008

# 0-1 percent of data to have in training set
TRAIN_TEST_SPLIT = 0.9

# Debug Tools
INPUT_DEBUG = False  # Blocking plot showing sample training image and histogram

# Logging
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

# Directories
import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.abspath(os.path.join(base_path, 'local', 'models'))
LOGS_DIR = os.path.abspath(os.path.join(base_path, 'local', 'logs'))
DATA_DIR = os.path.abspath(os.path.join(base_path, 'local', 'datasets'))
SHIP_DIR = os.path.abspath(os.path.join(base_path, 'local', 'ships'))

# Hyperparameter space to explore
# TODO: For now comment out the larger nets as they are unfair
from hyperopt import hp

SPACE = {'dataset'      : hp.choice('dataset', ['1203_d1.pickle',
                                                '1203_d2.pickle',
                                                '1203_d3.pickle',
                                                '1203_d4.pickle']),
         'dataset_size' : hp.choice('dataset_size', [2000, 3000, 4000]),
         'base'         : hp.choice('base', [
             'xception',
             'inception_res_v2',
             'res_net_50',
             'a3c',
             'tall_kernel',
             'a3c_sepconv',
             'minires',
             'simpleconv']),
         'head'         : hp.choice('head', ['fc']),
         'dim_reduction': hp.choice('dim_reduction', ['global_average',
                                                      'global_max',
                                                      'flatten']),
         'xception params'        : {
             'trainable'  : hp.choice('xception trainable', [True, False]),
             'pre_trained': hp.choice('xception pre_trained', [True, False]),
         },
         'inception_res_v2 params': {
             'trainable'  : hp.choice('iresv2 trainable', [True, False]),
             'pre_trained': hp.choice('iresv2 pre_trained', [True, False]),
         },
         'res_net_50 params'      : {
             'trainable'  : hp.choice('rn50 trainable', [True, False]),
             'pre_trained': hp.choice('rn50 pre_trained', [True, False]),
         },
         'fc params'    : {
             'dense_layers'      : hp.choice('dense_layers',
                                             [[64, 32], [256], [256, 64], [64], [32, 16], [256, 32]]),
             'dense_activations' : hp.choice('dense_activations', ['relu']),
             'dropout_percentage': hp.uniform('dropout_percentage', 0, 0.5),
         },

         'batch_size'   : hp.choice('batch_size', [16, 32]),
         'epochs'       : hp.choice('epochs', [25]),
         'optimizer'    : hp.choice('optimizer', ['rmsprop', 'sgd', 'adam', 'nadam']),
         'learning_rate': hp.choice('learning_rate', [0.0001, 0.00005]),
         'decay'        : hp.choice('decay', [0.0, 0.004, 0.0001]),
         'clipnorm'     : hp.choice('clipnorm', [0., 1.]),
         }
