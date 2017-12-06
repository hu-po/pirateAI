import logging
import os
import shutil
import sqlite3
from src.hyperopt_trainer import HyperoptTrainer
from src.pirate import Pirate
import src.config as config


def _check_dna(func):
    """
    Decorator makes sure dna is a string and not none
    :raises ValueError: if dna is not string
    """

    def wrapper(*args, **kwargs):
        dna = kwargs.get('dna', None)
        # TODO: Check if it is actually a uuid4, not just if its a string
        if not isinstance(dna, str):
            raise ValueError('dna must be a string UUID4')
        return func(*args, **kwargs)

    return wrapper


class Ship(object):
    """
      The Ship is where the Pirates live. It contains the interface to the sqlite database
         which is where scores and meta-data for each pirate is kept.
    """

    def __init__(self, ship_name='Boat'):
        """
        :param ship_name: (string) name this vessel!
        """
        self.log = logging.getLogger(__name__)
        self.name = ship_name
        # Database connection params
        self._c = None
        self._conn = None
        self._database_path = os.path.abspath(os.path.join(config.SHIP_DIR, self.name + '.db'))

    def __enter__(self):
        """
        Set off to sea! Connects to local sqlite db. Creates table if database does not yet exist
        """
        # Connect to the database, set up cursor
        self.log.debug('Starting up database at %s' % self._database_path)
        self._conn = sqlite3.connect(self._database_path)
        self._conn.row_factory = sqlite3.Row
        self._c = self._conn.cursor()
        # Create the table
        with self._conn:
            try:
                self._c.execute("""CREATE TABLE pirates (
                                dna TEXT,
                                name TEXT DEFAULT 'Unborn',
                                rank INTEGER DEFAULT 0,
                                win INTEGER DEFAULT 0,
                                loss INTEGER DEFAULT 0,
                                saltyness INTEGER DEFAULT 0
                                )""")
            except sqlite3.OperationalError:
                pass  # Table was already created
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    @_check_dna
    def _add_pirate(self, dna=None):
        """
        Adds a pirate to the ship
        :param dna: (string) identifier for the pirate
        :raises ValueError: if dna is not string
        """
        with self._conn:
            try:
                self._c.execute('INSERT INTO pirates(dna) VALUES (:dna)', {'dna': dna})
            except sqlite3.Error as e:
                self.log.warning('Could not add pirate to ship. Error: %s' % e)

    @_check_dna
    def _walk_the_plank(self, dna=None):
        """
        Removes a pirate from the ship
        :param dna: (string) identifier for the pirate
        :raises ValueError: if dna is not string
        """
        with self._conn:
            try:
                self._c.execute('DELETE FROM pirates WHERE dna=:dna', {'dna': dna})
            except sqlite3.Error as e:
                self.log.warning('Could not remove pirate from ship. Error: %s' % e)

    @_check_dna
    def _set_prop(self, dna=None, prop=None):
        """
        Updates properties of pirate on ship
        :param dna: (string) identifier for the pirate
        :param prop: {string:val,} name:value of the properties
        :return: (bool) error
        """
        # TODO: take a list of attributes to update
        if not isinstance(prop, dict) and not all(isinstance(p, str) for p in prop.keys()):
            raise ValueError('Must give a dictionary of properties with string keys to find')
        with self._conn:
            try:
                prop_str = ''
                for key, value in prop.items():
                    prop_str += key + ' = ' + str(value) + ' , '
                # TODO: This is unsafe SQL practices
                query = 'UPDATE pirates SET ' + prop_str[:-2] + 'WHERE dna = \'' + dna + '\''
                self._c.execute(query)
                return False
            except sqlite3.Error as e:
                self.log.warning('Could not set pirate properties. Error: %s' % e)
                return True

    @_check_dna
    def _get_prop(self, dna=None, prop=None):
        """
        Returns properties of pirate on ship
        :param dna: (string) identifier for the pirate
        :param prop: [string,] name(s) of the property
        :return: (bool), [val,] error, name:value of the properties
        """
        if not isinstance(prop, list) and not all(isinstance(p, str) for p in prop):
            raise ValueError('Must give a list of string properties to find')
        with self._conn:
            try:
                query = 'SELECT ' + ','.join(prop) + ' FROM pirates WHERE dna = \'' + dna + '\''
                self._c.execute(query)
                sql_row = [dict(a) for a in self._c.fetchall()]  # TODO: clean up b2b list comprehension
                return False, [row[key] for key, row in zip(prop, sql_row)]
            except (TypeError, sqlite3.Error) as e:
                self.log.warning('Could not get pirate properties. Error: %s' % e)
                return True, None

    @_check_dna
    def create_pirate(self, dna=None):
        """
        Creates a pirate on the ship. Watch out: this loads pirate model to memory.
        :param dna: (string) identifier for the pirate
        :return: (bool), (Pirate) error, the pirate object
        :raises ValueError: if dna is not string
        """
        with self._conn:
            try:
                self._c.execute('SELECT * FROM pirates WHERE dna=:dna', {'dna': dna})
                pirate_info = dict(self._c.fetchone())
            except (TypeError, sqlite3.Error) as e:
                self.log.warning('Could not find pirate in ship. Error: %s' % e)
                return True, None
        try:
            pirate = Pirate(dna=pirate_info.get('dna', None),
                            name=pirate_info.get('name', None),
                            rank=pirate_info.get('rank', None),
                            win=pirate_info.get('win', None),
                            loss=pirate_info.get('loss', None),
                            saltyness=pirate_info.get('saltyness', None))
            # Update the name for the pirate
            self._set_prop(dna=dna, prop={'name': '\'' + pirate.name + '\''})
        except FileNotFoundError:
            self.log.warning('Could not create pirate. Could not find model associated with it')
            return True, None
        return False, pirate

    def get_best_pirates(self, n=1):
        """
        The (up-to) N saltiest pirates on board the ship.
        :param n: (int) up to this number of pirates, less if not many pirates in db
        :return: [pirates+] list of pirates
        """
        with self._conn:
            self._c.execute('SELECT dna FROM pirates ORDER BY saltyness DESC LIMIT 50')
            sql_row = [dict(a) for a in self._c.fetchall()]
        pirates = []
        for i, d in enumerate(sql_row):
            if i >= n:
                break
            err, pirate = self.create_pirate(dna=d['dna'])
            if not err:  # Don't add pirates that throw an error on creation
                pirates.append(pirate)
        return pirates

    def marooning_update(self, winner, losers):
        """
        Updates the ship with the results from a marooning
        :param winners: [string,] list of string dnas for winning pirates
        :param losers: [string,] list of string dnas for losing pirates
        :return: (bool) error
        """
        # Update wins, losses, and saltyness for the winner and the losers
        if winner:  # Empty string is False in python
            # We can use the +1 formulation because we use string concatentation
            self._set_prop(dna=winner, prop={'win': 'win + 1'})
            self._set_prop(dna=winner, prop={'saltyness': 'saltyness + ' + str(config.SALT_PER_WIN)})
        for dna in losers:
            self._set_prop(dna=dna, prop={'loss': 'loss + 1'})
            self._set_prop(dna=dna, prop={'saltyness': 'saltyness - ' + str(config.SALT_PER_LOSS)})
        return True  # Not yet implemented

    def headcount(self):
        """
        How many pirates are on this ship?
        :return: (int) number of pirates on this ship (or 0 if error)
        """
        with self._conn:
            try:
                self._c.execute('SELECT count() FROM pirates')
                sql_row = [dict(a) for a in self._c.fetchall()]
                num_pirates = sql_row[0]['count()']
                self.log.info('There are currently %s pirates on the ship' % num_pirates)
                return num_pirates
            except (TypeError, sqlite3.Error) as e:
                self.log.warning('Failed to perform headcount ship. Error: %s' % e)
                return 0

    @_check_dna
    def delete_local_pirate_files(self, dna=None):
        """
        Deletes local files associated with a pirate (model, docs, logs).
        :param dna: (string) identifier for the pirate
        :raise FileNotFoundError: can't find the local files
        """
        removed = {'model': False, 'doc': False, 'log': False}
        for dirpath, _, files in os.walk(config.MODEL_DIR):
            if dna + '.h5' in files:
                os.remove(os.path.join(dirpath, dna + '.h5'))
                removed['model'] = True
            if dna + '.pickle' in files:
                os.remove(os.path.join(dirpath, dna + '.pickle'))
                removed['doc'] = True
        for dirpath, dirs, files in os.walk(config.LOGS_DIR):
            if dna in dirs:
                # Tensorboard logs are a folder
                shutil.rmtree(os.path.join(dirpath, dna))
                removed['log'] = True
        if not all(removed.values()):  # All of the files should be removed
            self.log.warning('When removing local files for %s, could not find %s' % (dna, removed))

    def less_pirates(self, n=config.NUM_PIRATES_PER_CULLING):
        """
        Removes the N pirates with lowest saltyness from the ship (and associated local files)
        :param n: (int) how many pirates to be removed
        """
        with self._conn:
            self._c.execute('SELECT dna FROM pirates ORDER BY saltyness ASC LIMIT ?', (str(n),))
            sql_row = [dict(a) for a in self._c.fetchall()]
        for d in sql_row:
            self.delete_local_pirate_files(dna=d['dna'])
            self._walk_the_plank(dna=d['dna'])

    def more_pirates(self, num_pirates=config.NUM_PIRATES_PER_TRAIN, max_tries=config.MAX_TRAIN_TRIES, space=config.SPACE):
        """
        Create pirates using hyperopt, adds them to the ship
        :param num_pirates: (int) number of pirates to generate
        :param max_tries: (int) max number of hyperopt runs before choosing best pirates
        """
        assert space, 'Please provide a hyperparameter space for creating pirate models'
        with HyperoptTrainer() as trainer:
            results = trainer.run_hyperopt(max_tries, space)
        # Sort results by highest validation accuracy
        top = sorted(results.items(), key=lambda e: e[1])
        self.log.info('Making %s more pirates' % num_pirates)
        for idx, (dna, _) in enumerate(top):
            if idx < num_pirates:  # Only add the best N pirates
                self._add_pirate(dna=dna)
                self._set_prop(dna=dna, prop={'rank': idx})
                self._set_prop(dna=dna, prop={'saltyness': config.STARTING_SALT})
            else:
                self.delete_local_pirate_files(dna=dna)  # Delete pirate model from memory