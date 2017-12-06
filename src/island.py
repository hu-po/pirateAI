import logging
import time
import random
import keras
from statistics import median
from unityagents import UnityEnvironment
from src.pirate import Pirate
import src.config as config


class Island(object):
    """
    The Island is where pirates are marooned, testing their saltyness.
        Holds the unity environment. Used as a context manager.
    """

    def __init__(self, host_ip=None, host_port=None, brain=None, file_name=None):
        """
        :param host_ip: (string) host ip, if not provided runs locally
        :param host_port: (string) host port, if not provided runs locally
        :param brain: (string) name of the external brain in unity environment
        :param file_name: (string) name of the unity environment executable
        """
        self.log = logging.getLogger(__name__)
        if not host_ip or not host_port:
            self.log.info('No host ip or port provided, running in local training mode')
            self._train_mode = True
        else:
            self.log.info('Running in external testing mode')
            self._train_mode = False
        self._host_ip = host_ip
        self._host_port = host_port
        self._brain_name = brain
        self.file_name = file_name

    def __enter__(self):
        # Connect to the Unity environment
        self.unity_env = UnityEnvironment(file_name=self.file_name,
                                          host_ip=self._host_ip,
                                          base_port=self._host_port)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Kill the Unity environment
        self.unity_env.close()
        del self.unity_env

    def maroon(self, ship=None, num_best_pirates=config.N_BEST_PIRATES):
        """
        Maroon some pirates. Figure out which one is truly saltiest.
        :param ship: (Ship) the ship is where pirates live
        :param num_best_pirates: number of pirates to select before choosing randomly
        :return:
        """
        assert ship, "No ship specified when marooning"
        # Randomly select 2 from N best pirates
        pirates = random.sample(ship.get_best_pirates(num_best_pirates), 2)
        # Run the marooning rounds
        self.log.info('Marooning the pirates %s' % ', '.join([pirate.name for pirate in pirates]))
        err, winners, losers = self._run_rounds(pirates=pirates)
        if not err:  # If no error occurred during the marooning
            # Update the ship accordingly
            ship.marooning_update(winners, losers)
        # Delete the session to prevent GPU memory from getting full
        keras.backend.clear_session()

    def _run_rounds(self, pirates=None, bounty=config.BOUNTY, max_rounds=config.MAX_ROUNDS):
        """
        Runs rounds between a list of pirates
        :param pirates: [pirates] list of N pirates to maroon
        :param bounty: (int) how many wins to be the winner
        :param max_rounds: (int) maximum number of rounds in one marooning
        :return: (bool),(string),[string,] error, winning pirate dna, losing pirates dna
        """
        if any([not isinstance(pirate, Pirate) for pirate in pirates]):
            raise ValueError('Some of the pirates you provided are not pirates')
        # tracking variables for the match
        score = [0] * len(pirates)
        round_idx = 0
        winner = False  # Is there a winning pirate?
        while round_idx < max_rounds:
            self.log.info("-------------- Round %s" % str(round_idx + 1))
            try:
                winner_idx, times = self._round(pirates)
                # times contains execution times for each step
                self.log.info("%d steps taken." % len(times))
                self.log.info("python execution time [median: %ds, max: %ds, min: %ds] "
                              % (median(times), max(times), min(times)))
                score[winner_idx] += 1
            except ValueError:
                self.log.warning('Bad values passed within a round, discarding results...')
            except TimeoutError:
                self.log.info('Round Complete! But no clear winner')
            round_idx += 1
            if any(score[i] >= bounty for i in score):
                winner = True
                break  # Break when a pirate reaches the max score
        if winner:
            winning_idx = score.index(max(score))
            self.log.info('Match complete! %s claims victory' % pirates[winning_idx].name)
            winning_pirate = pirates.pop(winning_idx)
            return False, winning_pirate.dna, [pirate.dna for pirate in pirates]
        else:
            self.log.info('Match complete! No pirate was able to demonstrate superior saltyness')
            return False, '', [pirate.dna for pirate in pirates]

    def _round(self, pirates=None, max_steps=10000):
        """
        Carries out a single round of pirate on pirate action
        :param pirates: [pirates] list of N pirates in the round
        :param max_steps: (int) maximum number of steps in round
        :return: (int),[int] index of winner, list of step execution times
        :raises TimeoutError: no done signal, max steps reached
        :raises ValueError: unity agents logic is having trouble
        """
        # Reset the environment
        env_info = self.unity_env.reset(train_mode=self._train_mode)
        # Time python code each step, interesting and a good sanity checker
        py_t0, py_t1 = None, None
        episode_times = []
        # Execute steps until environment sends done signal
        while True:
            if len(episode_times) > max_steps:
                raise TimeoutError('Unity environment never sent done signal, perhaps it disconnected?')
            # TODO: [0] index works because we only have one camera per pirate
            observation = env_info[self._brain_name].observations[0]
            agents_done = env_info[self._brain_name].local_done
            if all(agents_done):  # environment finished first
                raise TimeoutError('Neither pirate was able to find treasure')
            actions = []
            for i, pirate in enumerate(pirates):
                if agents_done[i]:
                    self.log.info("Round complete! %s got to the treasure first!" % pirate.name)
                    return i, episode_times
                # Get the action for each pirate based on its observation
                actions.append(pirate.act(observation[i, :, :, :]))
            if py_t0:  # timing
                episode_times.append(py_t1 - py_t0)  # timing
            py_t1 = time.time()  # timing
            env_info = self.unity_env.step(actions)  # Step in unity environment
            py_t0 = time.time()  # timing


if __name__ == '__main__':
    island = Island(brain='PirateBrain', file_name='local/unityenv/BootyFind')
