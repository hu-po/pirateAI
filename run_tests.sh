#!/usr/bin/env bash

# Run python tests in tests module
python -m unittest discover

# Individual tests
#python -m unittest tests.test_pirate
#python -m unittest tests.test_ship
#python -m unittest tests.test_island
#python -m unittest tests.test_model_chunks

# Might need context file in tests directory
## Context allows testing of modules without building solution
#import os
#import sys
#mod_path = os.path.abspath(os.path.join('..'))
#sys.path.append(mod_path)