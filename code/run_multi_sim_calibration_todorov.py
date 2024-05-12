
from subprocess import Popen
from concurrent.futures import ThreadPoolExecutor
import sys

num_trials = 5

prcs = []
max_frequency = 10
max_amplitude = 21
max_prcs_count = 20
learnrates = [0.8, 0.4]

try:
    for learnrate in learnrates:
        for frequency in range(1,max_frequency,2):
            for amplitude in range(4,max_amplitude,4):
                idx = 0
                while(idx < num_trials):
                    if len(prcs) < max_prcs_count:
                        prcs.append(Popen(['python3', 'run_calibration_todorov.py', str(idx), str(learnrate), str(frequency), str(amplitude)]))
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
