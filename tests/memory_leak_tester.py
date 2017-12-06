import linecache
import os
import tracemalloc
import sys
mod_path = os.path.abspath(os.path.join('..'))
sys.path.append(mod_path)
import src.config as config
from run import train

"""
This python file is used to test for memory leaks using
tracemalloc while running training.
"""

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

# use tracemalloc to find memory leaks
tracemalloc.start()

test_ship_name = 'TestShip'

# Change some configs to only train a single model
config.NUM_PIRATES_PER_TRAIN = 1
config.MAX_TRAIN_TRIES = 1
train(ship_name=test_ship_name,
      full_cycles=1,
      maroon_cycles=1,
      max_pirates_in_ship=2,
      min_pirates_in_ship=2)

snapshot_before = tracemalloc.take_snapshot()
import keras
keras.backend.clear_session()
snapshot_after = tracemalloc.take_snapshot()
stats = snapshot_after.compare_to(snapshot_before, 'lineno')

print('TOP FOR AFTER SNAPSHOT --------------')
display_top(snapshot_after)

print('TOP FOR BEFORE SNAPSHOT --------------')
display_top(snapshot_before)

print("[ Top 10 ]")
for stat in stats[:10]:
    print(stat)

try:
    os.remove(os.path.join(config.SHIP_DIR, test_ship_name))
except FileNotFoundError:
    pass