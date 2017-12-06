import unittest
from src.hyperopt_trainer import HyperoptTrainer

class TestModelChunks(unittest.TestCase):
    def setUp(self):
        self.trainer = HyperoptTrainer()
        self.hyperparams = {'dataset'      : 'test.pickle',
                            'dataset_size' : 80,
                            'head'         : 'fc',
                            'dim_reduction': 'flatten',
                            'fc params'    :
                                {
                                    'dense_layers'      : [32],
                                    'dense_activations' : 'relu',
                                    'dropout_percentage': 0.3,
                                },
                            'batch_size'   : 16,
                            'epochs'       : 1,
                            'optimizer'    : 'rmsprop',
                            'learning_rate': 0.0005,
                            'decay'        : 0.0001,
                            'clipnorm'     : 1.0
                            }

    def tearDown(self):
        del self.trainer, self.hyperparams

    def test_custom_model(self):
        pass  # Gets tested in functions below

    def test_model_chunk(self):
        pass  # Gets tested in functions below

    def test_a3c(self):
        self.hyperparams['base'] = 'a3c'
        self.trainer.model(self.hyperparams, test_mode=True)

    def test_a3c_sepconv(self):
        self.hyperparams['base'] = 'a3c_sepconv'
        self.trainer.model(self.hyperparams, test_mode=True)

    def test_simpleconv(self):
        self.hyperparams['base'] = 'simpleconv'
        self.trainer.model(self.hyperparams, test_mode=True)

    def test_minires(self):
        self.hyperparams['base'] = 'minires'
        self.trainer.model(self.hyperparams, test_mode=True)

    def test_tall_kernel(self):
        self.hyperparams['base'] = 'tall_kernel'
        self.trainer.model(self.hyperparams, test_mode=True)

    def test_inception_res_v2(self):
        self.hyperparams['base'] = 'inception_res_v2'
        self.hyperparams['inception_res_v2 params'] = \
            {
                'trainable'  : True,
                'pre_trained': True,
                'input_shape': (128, 128, 3)
            }
        self.trainer.model(self.hyperparams, test_mode=True)

    def test_res_net_50(self):
        self.hyperparams['base'] = 'res_net_50'
        self.hyperparams['res_net_50 params'] = \
            {
                'trainable'  : True,
                'pre_trained': True,
                'input_shape': (128, 128, 3)
            }
        self.trainer.model(self.hyperparams, test_mode=True)

    def test_xception(self):
        self.hyperparams['base'] = 'xception'
        self.hyperparams['xception params'] = \
            {
                'trainable'  : True,
                'pre_trained': True,
                'input_shape': (128, 128, 3)
            }
        self.trainer.model(self.hyperparams, test_mode=True)


if __name__ == '__main__':
    unittest.main()
