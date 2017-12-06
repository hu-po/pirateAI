import os
import unittest
from uuid import uuid4
from keras.models import Sequential
from keras.layers import Dense, Flatten
from src.pirate import Pirate

class TestPirate(unittest.TestCase):

    def test_initialization(self):
        # No model will be found for a blank pirate
        with self.assertRaises(FileNotFoundError):
            blank_pirate = Pirate()

        # Make a blank keras model to try and load
        test_dna = str(uuid4())
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'local', 'models'))
        test_model = Sequential()
        test_model.add(Flatten(input_shape=(128, 128, 3)))
        test_model.add(Dense(1))
        test_model.save(model_path + '/' + test_dna + '.h5')
        test_pirate = Pirate(dna=test_dna)
        self.assertEqual(test_pirate.dna, test_dna)
        self.assertTrue(isinstance(test_pirate.name, str))

        # Delete the test model
        os.remove(model_path + '/' + test_dna + '.h5')

    def test_act(self):
        pass


if __name__ == '__main__':
    unittest.main()
