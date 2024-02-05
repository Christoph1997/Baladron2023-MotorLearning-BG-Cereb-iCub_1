
from subprocess import Popen
from concurrent.futures import ThreadPoolExecutor
import sys

num_trials = 5

prcs = []
#max_prcs_count = 5
max_concurrent_processes = 15

learnrates = [0.01,0.1,0.2,0.4,0.8]



def run_trial(args):
    idx, learnrate, frequency, amplitude = args
    return Popen(['python3', 'run_reaching.py', str(idx), str(learnrate), str(frequency), str(amplitude)])

try:
    with ThreadPoolExecutor(max_concurrent_processes) as executor:
        for learnrate in learnrates:
            for frequency in range(1, 7, 2):
                for amplitude in range(4, 20, 4):
                    args_list = [(idx, learnrate, frequency, amplitude) for idx in range(num_trials)]
                    processes = list(executor.map(run_trial, args_list))
                    for process in processes:
                        process.wait()

except KeyboardInterrupt:
    print("Interrupted")
    sys.exit(1)

"""
try:
    for learnrate in learnrates:
        for frequency in range(1,7,2):
            for amplitude in range(4,20,4):
                idx = 0
                while(idx < num_trials):
                    if len(prcs) < max_prcs_count:
                        prcs.append(Popen(['python3', 'run_reaching.py', str(idx), str(learnrate), str(frequency), str(amplitude)]))
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
"""
