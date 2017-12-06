import logging
from src.ship import Ship
from src.island import Island
import src.config as config


# PLEASE GO TO config.py FILE TO CHANGE CONFIGS

def train(ship_name=config.SHIP_NAME,
          full_cycles=config.FULL_CYCLES,
          maroon_cycles=config.MAROON_CYCLES,
          max_pirates_in_ship=config.MAX_PIRATES_IN_SHIP,
          min_pirates_in_ship=config.MIN_PIRATES_IN_SHIP):
    """
    Train mode: runs a local UnityEnvironment, general loop is:
         - Generate pirates until you have some minimum amount in the ship
         - Run matches to rank pirates
         - Cull the worse performing pirates in the ship
    :param ship_name: (str) name of the ship
    :param full_cycles: (int) number of  full pirateAI cycles to run
    :param maroon_cycles: (int) number of maroonings to run for each hyperopt training
    :param max_pirates_in_ship: (int) max number of pirates in ship
    :param min_pirates_in_ship: (int) min number of pirates in ship
    """
    with Ship(ship_name=ship_name) as ship:
        for _ in range(full_cycles):
            pirates_in_ship = ship.headcount()
            while pirates_in_ship < max_pirates_in_ship:
                ship.more_pirates()  # Not enough hands on deck
                pirates_in_ship = ship.headcount()
            with Island(brain='PirateBrain', file_name='local/unityenv/Island') as island:
                for _ in range(maroon_cycles):
                    island.maroon(ship=ship)
            if pirates_in_ship > min_pirates_in_ship:
                ship.less_pirates()


def test(ship_name=config.SHIP_NAME,
         full_cycles=config.FULL_CYCLES,
         maroon_cycles=config.MAROON_CYCLES):
    """
    Test Mode: runs an external UnityEnvironment, general loop is:
        - Run matches to rank pirates
    :param ship_name: (str) name of the ship
    :param full_cycles: (int) number of  full pirateAI cycles to run
    :param maroon_cycles: (int) number of maroonings to run for each hyperopt training
    """
    with Ship(ship_name=ship_name) as ship:
        for _ in range(full_cycles):
            with Island(host_ip=config.WINDOWS_IP,
                        host_port=config.WINDOWS_PORT,
                        brain='PirateBrain') as island:
                for _ in range(maroon_cycles):
                    island.maroon(ship=ship)


if __name__ == '__main__':

    logger = logging.getLogger(__name__)

    if config.MODE == 'train':
        logger.info("Running in training mode")
        train()

    if config.MODE == 'test':
        logger.info("Running in Testing mode")
        test()
