import os
import sys
import cProfile

mod_path = os.path.abspath(os.path.join('..'))
sys.path.append(mod_path)
import src.config as config
from run import train

"""
This python file is used to profile training
"""

# Create a test ship and run training
test_ship_name = 'TestShip'

cProfile.run('train(ship_name=test_ship_name, full_cycles = 1, maroon_cycles = 1, max_pirates_in_ship = 2, '
             'min_pirates_in_ship = 2)')
