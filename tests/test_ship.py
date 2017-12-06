import unittest
import os
import logging
from uuid import uuid4
from keras.models import Sequential
from keras.layers import Dense, Flatten
from src.pirate import Pirate
from src.ship import Ship

class TestShip(unittest.TestCase):
    def setUp(self):
        self.ship = Ship(ship_name='TestBoat')
        self.test_dnas = [str(uuid4()) for _ in range(10)]

        # Make a blank keras model for test pirates
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'local', 'models'))
        test_model = Sequential()
        test_model.add(Flatten(input_shape=(128, 128, 3)))
        test_model.add(Dense(1))
        test_model.compile(optimizer='rmsprop', loss='mse')

        # Save models and create pirates
        self.test_pirates = []
        for i, dna in enumerate(self.test_dnas):
            test_model.save(self.model_path + '/' + dna + '.h5')
            self.test_pirates.append(Pirate(dna=dna))
            if i > len(self.test_dnas) - 5:
                break  # Leave some models un-saved

        # Add pirates to the ship
        # Add some pirates
        for i in range(2):
            self.ship._add_pirate(dna=self.test_dnas[i])

    def tearDown(self):
        # Delete all the test models we created
        for i, pirate in enumerate(self.test_pirates):
            os.remove(self.model_path + '/' + pirate.dna + '.h5')
        self.ship.sink()
        del self.ship

    def test_get_set(self):
        with self.assertRaises(ValueError):
            self.ship._get_prop(dna=None, prop=['test'])  # dna should be string
            self.ship._get_prop(dna=self.test_dnas[0], prop=[None])
            self.ship._set_prop(dna=None, prop=[])  # dna should be string
            self.ship._set_prop(dna=self.test_dnas[0], prop=[])  # prop should be a dict
            self.ship._set_prop(dna=self.test_dnas[0], prop={1: 'a'})  # prop dict keys should be strings

        with self.assertLogs(level=logging.WARNING):
            # Column does not exist
            self.assertTrue(self.ship._set_prop(dna=self.test_dnas[0], prop={'test': 4}))
            err, _ = self.ship._get_prop(dna=self.test_dnas[0], prop=['test'])
            self.assertTrue(err)

        # Valid conditions
        self.assertFalse(self.ship._set_prop(dna=self.test_dnas[0], prop={'saltyness': 2, 'loss': 1}))
        self.assertFalse(self.ship._set_prop(dna=self.test_dnas[0], prop={'win': 'win + 1'}))
        err, _ = self.ship._get_prop(dna=self.test_dnas[0], prop=['win'])
        self.assertFalse(err)

    def test_add_remove(self):
        with self.assertRaises(ValueError):
            self.ship._add_pirate(dna=None)
            self.ship._walk_the_plank(dna=None)

    def test_create_pirate(self):
        with self.assertRaises(ValueError):
            self.ship.create_pirate(dna=None)

        with self.assertLogs(level=logging.WARNING):
            # Give it a pirate with no model
            err, _ = self.ship.create_pirate(dna=self.test_dnas[-2])
            self.assertTrue(err)
            # Give it a pirate not on the ship
            err, _ = self.ship.create_pirate(dna=str(uuid4()))
            self.assertTrue(err)

        err, _ = self.ship.create_pirate(dna=self.test_dnas[0])
        self.assertFalse(err)

    def test_get_best_pirates(self):
        # Should only return the 2 pirates in ship
        pirates = self.ship.get_best_pirates(n=3)
        self.assertEqual(len(pirates), 2)
        self.assertTrue(all(isinstance(pirate, Pirate) for pirate in pirates))


if __name__ == '__main__':
    unittest.main()
