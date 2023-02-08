import argparse
import os
import random
import time

import json

import pandas as pd
from compiled_jss.CPEnv import CompiledJssEnvCP

from stable_baselines3.common.vec_env import VecEnvWrapper
from torch.distributions import Categorical

import wandb

import torch
import numpy as np

import ray

from MyVecEnv import WrapperRay

from Network import Agent


class VecPyTorch(VecEnvWrapper):

    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()


def make_env(seed, instance):
    def thunk():
        _env = CompiledJssEnvCP(instance)
        return _env

    return thunk


if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = str(0)
    parser = argparse.ArgumentParser(description='Solving')
    parser.add_argument('--gym-id', type=str, default="compiled_env:jss-v4",
                        help='the id of the gym environment')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the number of parallel worker')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--wandb-project-name', type=str, default="BenchmarkCPEnv",
                        help="the wandb's project name")
    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ray.init(log_to_driver=False, include_dashboard=False)

    with torch.inference_mode():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent = Agent()
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        agent.load_state_dict(checkpoint["model"])
        actor = agent.actor
        actor = torch.jit.script(actor)
        actor = actor.to(device, non_blocking=True)
        actor.eval()
    print(f'Using device {device}')

    # for each file in the 'instances_run' folder
    instances = []
    for file in sorted(os.listdir('instances_run')):
        if not file.startswith('.'):
            instances.append('instances_run/' + file)

    experiment_name = f"benchmark__{0}__{int(time.time())}"
    wandb.init(project=args.wandb_project_name + 'DEBUG', config=vars(args), name=experiment_name, save_code=True)

    all_datas = []
    with torch.inference_mode():
        # for each instance
        for instance in instances:
            print(f'Now solving instance {instance}')
            for iter_idx in range(10):
                random.seed(iter_idx)
                np.random.seed(iter_idx)
                torch.manual_seed(iter_idx)
                start_time = time.time()
                fn_env = [make_env(0, instance)
                          for i in range(args.num_workers * 4)]

                current_solution_cost = float('inf')
                current_solution = []
                ray_wrapper_env = WrapperRay(lambda n: fn_env[n](),
                                             args.num_workers, 4, device)
                envs = VecPyTorch(ray_wrapper_env, device)
                obs = envs.reset()
                total_episode = 0
                while total_episode < envs.num_envs:
                    logits = actor(obs['interval_rep'], obs['attention_interval_mask'], obs['job_resource_mask'],
                                   obs['action_mask'], obs['index_interval'], obs['start_end_tokens'])
                    # temperature vector
                    temperature = torch.arange(0.5, 2.0, step=(1.5 / (args.num_workers * 4)), device=device)
                    logits = logits / temperature[:, None]
                    probs = Categorical(logits=logits).probs
                    # random sample based on logits
                    actions = torch.multinomial(probs, probs.shape[1]).cpu().numpy()
                    obs, reward, done, infos = envs.step(actions)
                    total_episode += done.sum()
                    for env_idx, info in enumerate(infos):
                        if 'makespan' in info and int(info['makespan']) < current_solution_cost:
                            current_solution_cost = int(info['makespan'])
                            current_solution = json.loads(info['solution'])
                total_time = time.time() - start_time
                # write solution to file
                with open('solutions_found/' + instance.split('/')[-1] + '_' + str(iter_idx) + '.json', 'w') as f:
                    json.dump(current_solution, f)
                print(f'Instance {instance} - Iter {iter_idx} - Cost {current_solution_cost} - Time {total_time}')
                all_datas.append({'instance': instance.split('/')[-1],
                                  'cost': current_solution_cost,
                                  'time': total_time})

            df = pd.DataFrame(all_datas)
            df.to_csv('results.csv')
            wandb.save('results.csv')

    # log dataframe
    wandb.log({"results": wandb.Table(dataframe=df)})
    wandb.finish()