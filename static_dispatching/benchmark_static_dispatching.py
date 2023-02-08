from os import listdir
from os.path import isfile, join

import numpy as np
from CPEnv import CompiledJssEnvCP

import pandas as pd


def FIFO(state, legal_actions):
    remaining_time = state[:, 0, 1]
    illegal_actions = np.invert(legal_actions)
    mask = illegal_actions * -1e8
    remaining_time += mask
    FIFO_action = np.argmax(remaining_time)
    return FIFO_action


def SPT(state, legal_actions):
    remaining_time = state[:, 0, 2]
    illegal_actions = np.invert(legal_actions)
    mask = illegal_actions * 1e8
    remaining_time += mask
    SPT_action = np.argmin(remaining_time)
    return SPT_action


def MTWR(state, legal_actions):
    remaining_time = np.sum(state[:, :, 2], axis=1)
    illegal_actions = np.invert(legal_actions)
    mask = illegal_actions * -1e8
    remaining_time += mask
    SPT_action = np.argmax(remaining_time)
    return SPT_action


def run_one_instance(instance: str, dispatching) -> float:
    env = CompiledJssEnvCP(instance)
    done = False
    state = env.reset()
    infos = {}
    while not done:
        real_state = np.array(state['interval_rep'], dtype=np.float32)
        legal_actions = np.array(state['action_mask'][:-1], dtype=bool)
        dispatcher_action = dispatching(real_state, legal_actions)
        #remaining_time = reshaped[:, 5]
        #illegal_actions = np.invert(legal_actions)
        #mask = illegal_actions * -1e8
        #remaining_time += mask
        #FIFO_action = np.argmax(remaining_time)
        assert legal_actions[dispatcher_action]
        state, reward, done, infos = env.step([dispatcher_action])
    env.reset()
    return infos['makespan']

if __name__ == '__main__':
    import time
    # list all files under the ../instances_run folder
    instances = [f'../instances_run/{f}' for f in listdir('../instances_run') if isfile(join('../instances_run', f)) and not f.startswith('.')]
    # sort instances
    instances.sort()
    # result dataframe
    result = []
    dispatchings_list = [FIFO, SPT, MTWR]
    for instance in instances:
        print(f'Running instance {instance}')
        for dispatching in dispatchings_list:
            start = time.time()
            makespan = run_one_instance(instance, dispatching)
            end = time.time()
            result.append((instance.split('/')[-1], dispatching.__name__, makespan, end-start))
        df = pd.DataFrame(result, columns=['instance', 'dispatching', 'makespan', 'time'])
        df.to_csv('results.csv')
