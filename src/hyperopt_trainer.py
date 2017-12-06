import logging
import os
import numpy as np
import time
from datetime import datetime
from uuid import uuid4
from PIL import Image
import random
import pickle
from collections import OrderedDict
import keras
from hyperopt import STATUS_OK
from hyperopt import fmin, hp, tpe
from keras import layers, Input, optimizers
from keras.models import Model
import src.config as config

from .model_chunks import custom_model
from .dataset import image_analysis

class HyperoptTrainer(object):
    """
    This class uses Hyperopt and Keras to generate pirates.
    """

    def __enter__(self):
        self.log = logging.getLogger(__name__)
        # Timing and record keeping
        self._start_time = time.time()
        self._results = {}
        self._max_eval = config.MAX_TRAIN_TRIES
        self._eval_idx = 0
        # Create directory for the logs and models
        folder_name = datetime.now().strftime('%Y%m%d')
        self._logs_dir = os.path.join(config.LOGS_DIR, folder_name)
        self._models_dir = os.path.join(config.MODEL_DIR, folder_name)
        os.makedirs(self._logs_dir, exist_ok=True)
        os.makedirs(self._models_dir, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def data_loader(self, dataset=None, shuffle=True, size=None):
        """
        Data providing function. Reads from local datasets folder
        :param dataset: (string) path to folder containing raw images and csv file with targets
        :param shuffle: (bool) shuffle training data
        :param size: (int) size of final dataset
        :return: x_train, y_train, x_test, y_test
        """
        self.log.info("Loading data")
        assert dataset is not None, "Please provide a dataset (folder name)"
        data_path = os.path.join(config.DATA_DIR, dataset)
        # Load dataset using pickle
        with open(data_path, 'rb') as file:
            image_paths, labels = pickle.load(file)
        one_hot_labels = []
        images = []
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            try:
                # One-hot encode the vectors
                one_hot = [0, 0, 0, 0]
                one_hot[label] = 1.0
                # Clean up image, normalize (mean to 0, std to 1)
                image = Image.open(path)
                image = np.asarray(image, dtype=np.float32) / 255
            except:  # If there is some issue reading in data, skip datapoint
                continue
            one_hot_labels.append(np.asarray(one_hot, dtype=np.float32))
            images.append(image)
        # Shuffle data before cutting it into test and src
        if shuffle:
            x = list(zip(images, one_hot_labels))
            random.shuffle(x)
            images, one_hot_labels = zip(*x)
        self.log.info("Separating data into test and src.")
        split_idx = int(config.TRAIN_TEST_SPLIT * len(one_hot_labels))
        train_input = images[:split_idx]
        train_target = one_hot_labels[:split_idx]
        test_input = images[split_idx:]
        test_target = one_hot_labels[split_idx:]
        if size:
            assert size < len(train_input), "Final dataset size too big, not enough data"
            train_input = train_input[:size]
            train_target = train_target[:size]
        self.log.info(" -- test : {}".format(len(test_target)))
        self.log.info(" -- src: {}".format(len(train_target)))
        # Convert to nparray before sending over
        return np.array(train_input), \
               np.array(train_target), \
               np.array(test_input), \
               np.array(test_target)

    def model(self, hyperparams, test_mode=False):
        """
        Builds and runs a model given a dictionary of hyperparameters
        :return: {dict}
            - loss: validation loss (to be minimized)
            - status: STATUS_OK (see hyperopt documentation)
        """
        run_doc = OrderedDict()  # Document important hyperparameters
        run_start_time = time.time()
        run_id = str(uuid4())
        # TODO: Not ideal: Loads from memory every time. Use generator?
        train_data, train_targets, test_data, test_targets = \
            self.data_loader(dataset=hyperparams['dataset'], size=hyperparams['dataset_size'])
        run_doc['dataset'] = hyperparams['dataset']
        run_doc['data_size'] = len(train_targets)
        # Visualization tools
        if config.INPUT_DEBUG:
            image_analysis(image=train_data[0, :, :, :], label=train_targets[0, :])
        # Input shape comes from image shape
        img_width = train_data[0].shape[0]
        img_height = train_data[0].shape[1]
        num_channels = train_data[0].shape[2]
        input_shape = (img_width, img_height, num_channels)
        run_doc['input_shape'] = '(%d, %d, %d)' % input_shape
        input_tensor = Input(shape=input_shape, dtype='float32', name='input_image')
        try:  # Model creation is in separate file
            x, run_doc = custom_model(input_tensor, params=hyperparams, run_doc=run_doc)
        except ValueError as e:
            if not test_mode:  # If not testing, ignore error causing models
                return {'loss': 100, 'status': STATUS_OK}
            else:
                raise e
        # Final layer classifies into 4 possible actions
        output = layers.Dense(4, activation='softmax')(x)
        # File names for the model and logs
        log_file = os.path.join(self._logs_dir, run_id)
        model_file = os.path.join(self._models_dir, run_id + '.h5')
        # Add some callbacks so we can track progress using Tensorboard
        callbacks = [keras.callbacks.EarlyStopping('val_loss', patience=config.TRAIN_PATIENCE, mode="min")]
        if not test_mode:  # Don't save models/logs if in testing mode
            callbacks += [keras.callbacks.TensorBoard(log_dir=log_file),
                          keras.callbacks.ModelCheckpoint(model_file, save_best_only=True)]
        # Choice of optimizer and optimization parameters
        if hyperparams['optimizer'] == 'sgd':
            optimizer = optimizers.SGD(lr=hyperparams["learning_rate"],
                                       decay=hyperparams["decay"],
                                       clipnorm=hyperparams["clipnorm"])
        elif hyperparams['optimizer'] == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=hyperparams["learning_rate"],
                                           decay=hyperparams["decay"],
                                           clipnorm=hyperparams["clipnorm"])
        elif hyperparams['optimizer'] == 'nadam':
            optimizer = optimizers.Nadam(lr=hyperparams["learning_rate"],
                                         schedule_decay=hyperparams["decay"],
                                         clipnorm=hyperparams["clipnorm"])
        elif hyperparams['optimizer'] == 'adam':
            optimizer = optimizers.Adam(lr=hyperparams["learning_rate"],
                                        decay=hyperparams["decay"],
                                        clipnorm=hyperparams["clipnorm"])
        # Save optimizer parameters to run doc
        run_doc['optimizer'] = hyperparams['optimizer']
        run_doc['opt_learning_rate'] = hyperparams["learning_rate"]
        run_doc['opt_decay'] = hyperparams["decay"]
        run_doc['opt_clipnorm'] = hyperparams["clipnorm"]
        # Create and compile the model
        model = Model(input_tensor, output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        # Print out model summary and store inside run documentation as list of strings
        model.summary()
        run_doc['model_summary'] = []
        model.summary(print_fn=(lambda a: run_doc['model_summary'].append(a)))
        # Fit the model to the datasets
        self.log.info("Fitting model (eval %d of %d) ..." % (self._eval_idx + 1, self._max_eval))
        self._eval_idx += 1
        model.fit(x=train_data, y=train_targets,
                  batch_size=hyperparams['batch_size'],
                  epochs=hyperparams['epochs'],
                  validation_data=(test_data, test_targets),
                  callbacks=callbacks,
                  verbose=1)
        val_loss, val_acc = model.evaluate(x=test_data, y=test_targets, verbose=2)
        self.log.info("     .... Completed!")
        self.log.info(" -- Evaluation time %ds" % (time.time() - run_start_time))
        self.log.info(" -- Total time %ds" % (time.time() - self._start_time))
        # Save training parameters to run doc
        run_doc['batch_size'] = hyperparams['batch_size']
        run_doc['epochs'] = hyperparams['epochs']
        run_doc['val_loss'] = val_loss
        run_doc['val_acc'] = val_acc
        # Results are used to pick best pirate
        self._results[run_id] = val_loss
        # Save run_doc to pickle file in model directory
        run_doc_file_name = run_id + '.pickle'
        if not test_mode:  # Don't save docs if in testing mode
            with open(os.path.join(self._models_dir, run_doc_file_name), 'wb') as f:
                pickle.dump(run_doc, f)
        self.log.info('Run Dictionary %s' % str(run_doc))
        # Delete the session to prevent GPU memory from getting full
        keras.backend.clear_session()
        # Optimizer minimizes validation loss
        return {'loss': val_loss, 'status': STATUS_OK}

    def run_hyperopt(self, max_eval, space):
        """
        Runs the hyperopt trainer
        :param max_eval: (int) max evaluations to carry out when running hyperopt
        :param space: {dict} }dictionary of hyperparameter space to explore
        :return: dictionary of best fit models by dna
        """
        # Reset run parameters
        self._max_eval = max_eval
        self._results = {}
        self._eval_idx = 0

        # Hyperopt is picky about the function handle
        def model_handle(params):
            return self.model(params)

        # Run the hyperparameter optimization
        _ = fmin(fn=model_handle, space=space, algo=tpe.suggest, max_evals=max_eval)
        return self._results
