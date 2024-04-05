
from subprocess import Popen
from concurrent.futures import ThreadPoolExecutor
import sys

num_trials = 10

prcs = []
max_prcs_count = 10

try:
    idx = 0
    for frequency in range (1,6,2):
        while(idx < num_trials):
            if len(prcs) < max_prcs_count:
                prcs.append(Popen(['python3', 'run_calibration_todorov.py', str(idx)], str(frequency)))
                idx += 1
            else:
                ret = prcs[0].wait()
                if type(ret) == int:
                    prcs.pop(0)

    for process in prcs:
        process.wait()

except KeyboardInterrupt:
    print("interrupted")
    for process in prcs:
        process.terminate()
