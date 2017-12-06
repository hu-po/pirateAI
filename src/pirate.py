import os
import random
import logging
import numpy as np
import pickle
from uuid import uuid4
from keras.models import load_model
from src.dataset import image_analysis
import src.config as config

class Pirate(object):
    """
    Pirates are the agents on the island. When instantiated, pirates load a model
        into GPU memory.
    """

    def __init__(self, dna=None, name='Unborn', win=0, loss=0, saltyness=0, rank=None):
        """
        :param dna: (string) identifier uuid4 string for a pirate
        :param name: (string) the pirate's name
        :param win: (int) number of wins
        :param loss: (int) number of losses
        :param saltyness: (int) an estimate of a Pirate's performance (think ELO)
        :param rank: (int) rank of this pirate from their training batch
        :raises FileNotFoundError: Can't load pirate model
        :raises ValueError: no input given
        """
        self.log = logging.getLogger(__name__)
        self.dna = dna or str(uuid4())
        if name == 'Unborn':
            self.name = self._generate_name(rank=rank)
        else:
            self.name = name
        self.win = win
        self.loss = loss
        self.saltyness = saltyness
        # Model contains weights and graph
        self._model = self._load_model()

    def act(self, input, visualize=False):
        """
        Runs the pirate model on the given input (image, etc).
        :param input: input tensor, format matches model
        :param visualize: (bool) display incoming image and metadata
        :return: (int) action resulting from model.
        :raises ValueError: no input given
        """
        if input is None:
            raise ValueError("Please provide an input image to generate an action")
        if len(input.shape) == 3:
            input = np.expand_dims(input, axis=0)
        norm_input_image = input
        output = self._model.predict(norm_input_image)
        # Classification model outputs action probabilities
        action = np.argmax(output)
        if visualize or config.INPUT_DEBUG:  # This blocks the GIL to visualize
            image_analysis(image=input[0, :, :, :], label=action)
        return action

    def description(self):
        """
        Finds and returns the info in hyperparameter text file
        :return: (string) or None
        :raises FileNotFoundError: Can't find hyperparameter text file in path
        """
        for dirpath, _, files in os.walk(config.MODEL_DIR):
            if self.dna + '.pickle' in files:
                with open(os.path.join(dirpath, self.dna + '.pickle'), 'rb') as file:
                    data = pickle.load(file)
                assert isinstance(data, dict), 'Pirate description is corrupted'
                # Pirate description printed out to logger
                self.log.info('--- Pirate %s (dna: %s) ---' % (self.name, self.dna))
                model_summary = data.pop('model_summary', None)
                for line in model_summary:
                    self.log.info(line)
                for key, val in data.items():
                    self.log.info('%s : %s' % (str(key), str(val)))
                return data
        raise FileNotFoundError('Could not find description in path using given dna string')

    def _load_model(self):
        """
        Tries to find pirate model in the model path
        :return: (keras.model) or None
        :raises FileNotFoundError: Can't find model in path
        """
        for dirpath, _, files in os.walk(config.MODEL_DIR):
            if self.dna + '.h5' in files:
                return load_model(os.path.join(dirpath, self.dna + '.h5'))
        raise FileNotFoundError('Could not find model in path using given dna string')

    def _generate_name(self, rank=-1):
        """
        Generates a proper pirate name
        :param rank: (int) rank with respect to training batch
        :return:(string) name
        """
        name = ''
        # Titles are ordered based on rank
        titles = ['Salty ', 'Admiral ', 'Captain ', 'Don ', 'First Mate ', 'Gunmaster ',
                  'Sailor ', 'Deckhand ', 'Mc', 'Cookie ', 'Lil', '']
        if rank in range(len(titles)):
            name += titles[rank]
        # The real part of the name is chosen randomly
        real_names = ['Jack', 'Haddock', 'Blackbeard', 'Will', 'Long', 'Simon', 'Barbossa']
        name += random.choice(real_names)
        self.log.debug('The Pirate %s has been created' % name)
        return name

    def __eq__(self, pirate):
        """
        Compare pirates using their dna
        :param pirate: (Pirate) the other pirate
        :return: (bool) True if dna matches
        """
        return self.dna == pirate.dna
