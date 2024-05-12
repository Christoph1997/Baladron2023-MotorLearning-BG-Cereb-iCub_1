
from subprocess import Popen
from concurrent.futures import ThreadPoolExecutor
import sys

num_trials = 5

prcs = []
max_prcs_count = 15

learnrates = [0.8, 0.4, 0.2, 0.1, 0.01]

try:
    for learnrate in learnrates:
        for frequency in range(1,8,2):
            for amplitude in range(4,21,4):
                idx = 0
                while(idx < num_trials):
                    if len(prcs) < max_prcs_count:
                        prcs.append(Popen(['python3', 'run_reaching.py', str(idx), str(learnrate), str(frequency), str(amplitude)]))
                        idx += 1
                    else:
                        ret = prcs[0].wait()
                        if type(ret) == int:
                            prcs.pop(0)

                if len(prcs) >= max_prcs_count:
                    for process in prcs:
                        process.wait()

except KeyboardInterrupt:
    print("interrupted")
    for process in prcs:
        process.terminate()

learnrates = [0.01]

try:
    for learnrate in learnrates:
        for frequency in range(1,8,2):
            for amplitude in range(4,21,4):
                idx = 0
                while(idx < num_trials):
                    if len(prcs) < max_prcs_count:
                        prcs.append(Popen(['python3', 'run_reaching.py', str(idx), str(learnrate), str(frequency), str(amplitude)]))
                        idx += 1
                    else:
                        ret = prcs[0].wait()
                        if type(ret) == int:
                            prcs.pop(0)

                if len(prcs) >= max_prcs_count:
                    for process in prcs:
                        process.wait()

except KeyboardInterrupt:
    print("interrupted")
    for process in prcs:
        process.terminate()
