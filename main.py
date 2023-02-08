from collections import defaultdict

import collections
from statistics import mean

import multiprocessing as mp

import json
from typing import List

import ray
import torch
import torch.optim as optim
from compiled_jss.CPEnv import CompiledJssEnvCP

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

import argparse
from distutils.util import strtobool
import numpy as np
import gym
import time
import random
import os
import wandb
from stable_baselines3.common.vec_env import VecEnvWrapper

from MyVecEnv import WrapperRay
from Network import Agent
#from NonCompCPEnvSeq import NonCompiledJssEnvCPLazySeq


class Experience:

    def __init__(self, all_obs_actions_to_take, all_obs_actions_took, all_bad_path):
        self.all_obs_actions_to_take = all_obs_actions_to_take
        self.all_obs_actions_took = all_obs_actions_took
        self.all_bad_path = all_bad_path


@ray.remote(num_cpus=0)
def compute_trajectory(fn_env, history: List, solution_agent: List, make_span_agent: int, agent: Agent, all_actions: List, time_limit: int = 30) -> Tuple[Experience, float]:
    with torch.no_grad():
        env: CompiledJssEnvCP = fn_env()
        env.training_mode = False
        obs = env.reset()
        done = False
        obs_action_took = []
        for actions in history:
            next_mask = torch.from_numpy(obs['action_mask']).float().unsqueeze(0)
            next_interval_reps = torch.from_numpy(obs['interval_rep']).float().unsqueeze(0)
            next_job_resource_masks = torch.from_numpy(obs['job_resource_mask']).float().unsqueeze(0)
            next_attention_interval_masks = torch.from_numpy(obs['attention_interval_mask']).float().unsqueeze(0)
            next_start_end_tokens = torch.from_numpy(obs['start_end_tokens']).float().unsqueeze(0)
            next_index_intervals = torch.from_numpy(obs['index_interval']).float().unsqueeze(0)
            _, logprobas, _ = agent(next_interval_reps, next_attention_interval_masks,
                                                         next_job_resource_masks, next_mask, next_index_intervals, next_start_end_tokens)
            logprobas = logprobas.squeeze(0)
            for one_action in actions:
                obs_copy = {}
                for key in obs:
                    obs_copy[key] = np.copy(obs[key])
                obs_copy["job_resource_mask_selected"] = obs_copy["job_resource_mask"][int(one_action)]
                obs_action_took.append((obs_copy, int(one_action), logprobas[one_action]))
            obs, reward, done, info = env.step(np.array(actions, dtype=np.longlong))
            assert not done
        solution_found, cp_solution, make_span_cp = env.solve_using_cp(solution_agent, time_limit=time_limit)
        if make_span_cp < make_span_agent and solution_found:
            improvement = (make_span_agent - make_span_cp) / make_span_agent
        elif not solution_found:
            experience = Experience([], [], [])
            return experience, 0, make_span_agent, env.env_name
        else:
            improvement = 0
        obs_action_to_take = []
        i = 0
        while not done:
            mask = obs["action_mask"]
            legal_actions = np.argwhere(mask[:-1] == 1).flatten()
            #action_to_take = env.jobs_count
            action_to_takes = []
            for action in legal_actions:
                if cp_solution[action][len(env.partial_solution[action])] == env.current_timestamp:
                    action_to_takes.append(action)
            if len(action_to_takes) == 0:
                action_to_takes = [env.jobs_count]
            next_mask = torch.from_numpy(obs['action_mask']).float().unsqueeze(0)
            next_interval_reps = torch.from_numpy(obs['interval_rep']).float().unsqueeze(0)
            next_job_resource_masks = torch.from_numpy(obs['job_resource_mask']).float().unsqueeze(0)
            next_attention_interval_masks = torch.from_numpy(obs['attention_interval_mask']).float().unsqueeze(0)
            next_start_end_tokens = torch.from_numpy(obs['start_end_tokens']).float().unsqueeze(0)
            next_index_intervals = torch.from_numpy(obs['index_interval']).float().unsqueeze(0)
            _, logprobas, _ = agent(next_interval_reps, next_attention_interval_masks,
                                               next_job_resource_masks, next_mask, next_index_intervals, next_start_end_tokens)
            logprobas = logprobas.squeeze(0)
            for one_action in action_to_takes:
                obs_copy = {}
                for key in obs:
                    obs_copy[key] = np.copy(obs[key])
                obs_copy["job_resource_mask_selected"] = obs_copy["job_resource_mask"][int(one_action)]
                obs_action_to_take.append((obs_copy, int(one_action), logprobas[one_action]))
            obs, reward, done, info = env.step(np.array(action_to_takes, dtype=np.longlong))
            i += 1
        obs = env.reset()
        done = False
        bad_path = []
        for actions in history:
            obs, reward, done, info = env.step(np.array(actions, dtype=np.longlong))
            assert not done
        for x in range(len(history), len(all_actions)):
            actions = all_actions[x]
            next_mask = torch.from_numpy(obs['action_mask']).float().unsqueeze(0)
            next_interval_reps = torch.from_numpy(obs['interval_rep']).float().unsqueeze(0)
            next_job_resource_masks = torch.from_numpy(obs['job_resource_mask']).float().unsqueeze(0)
            next_attention_interval_masks = torch.from_numpy(obs['attention_interval_mask']).float().unsqueeze(0)
            next_start_end_tokens = torch.from_numpy(obs['start_end_tokens']).float().unsqueeze(0)
            next_index_intervals = torch.from_numpy(obs['index_interval']).float().unsqueeze(0)
            _, logprobas, _ = agent(next_interval_reps, next_attention_interval_masks,
                                               next_job_resource_masks, next_mask, next_index_intervals, next_start_end_tokens)
            logprobas = logprobas.squeeze(0)
            for one_action in actions:
                obs_copy = {}
                for key in obs:
                    obs_copy[key] = np.copy(obs[key])
                obs_copy["job_resource_mask_selected"] = obs_copy["job_resource_mask"][int(one_action)]
                bad_path.append((obs_copy, int(one_action), logprobas[one_action]))
            obs, reward, done, info = env.step(np.array(actions, dtype=np.longlong))
        experience = Experience(obs_action_to_take, obs_action_took, bad_path)
        return experience, improvement, make_span_cp, env.env_name


if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = str(0)
    parser = argparse.ArgumentParser(description='End-2-End Agent')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="JSSEnv:jss-v4",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed of the experiment')
    parser.add_argument('--cp-solving-time', type=int, default=60,
                        help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=False,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--wandb-project-name', type=str, default="followUpAttentionSeq",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=20,
                        help='the number of mini batch')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count(),
                        help='the number of parallel worker')
    parser.add_argument('--env-per-worker', type=int, default=1,
                        help='number of environment per worker')
    parser.add_argument('--max-grad-norm', type=float, default=None,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.3,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=20,
                        help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--target-kl', type=float, default=0.01,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")

    args = parser.parse_args()
    if not args.seed:
        args.seed = 0

    num_envs = args.num_workers * args.env_per_worker


    class VecPyTorch(VecEnvWrapper):

        def __init__(self, venv):
            super(VecPyTorch, self).__init__(venv)

        def reset(self):
            return self.venv.reset()

        def step_async(self, actions):
            actions = actions.cpu().numpy()
            self.venv.step_async(actions)

        def step_wait(self):
            obs, reward, done, info = self.venv.step_wait()
            reward = torch.from_numpy(reward).unsqueeze(dim=1)
            return obs, reward, done, info


    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args),
               name=experiment_name, save_code=True)

    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    def make_env(seed, instance):
        def thunk():
            env = CompiledJssEnvCP(instance)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk


    ray.init(num_cpus=args.num_workers, log_to_driver=False, include_dashboard=False, runtime_env={
                "env_vars": {
                    # force `ray` to not kill a proces on OOM but use SWAP instead
                    "RAY_DISABLE_MEMORY_MONITOR": "1",
                }
             })

    l = collections.deque(maxlen=100)

    instance_to_pick = [
        "instances_train/ta31",
        "instances_train/ta32",
        "instances_train/ta33",
        "instances_train/ta34",
    ]

    fn_env = [make_env(args.seed + i, instance_to_pick[(i // args.env_per_worker) % len(instance_to_pick)])
              for i in range(args.num_workers * args.env_per_worker)]

    ray_wrapper_env = WrapperRay(lambda n: fn_env[n](),
                                 args.num_workers, args.env_per_worker, device)

    envs = VecPyTorch(ray_wrapper_env)

    agent = Agent()
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    agent = agent.to(device, non_blocking=True)
    agent.train()


    if args.anneal_lr:
        lr = lambda f: f * args.learning_rate

    global_step = 0
    next_obs = envs.reset()
    next_mask = next_obs['action_mask'].clone()
    next_interval_reps = next_obs['interval_rep'].clone()
    next_job_resource_masks = next_obs['job_resource_mask'].clone()
    next_attention_interval_masks = next_obs['attention_interval_mask'].clone()
    next_start_end_tokens = next_obs['start_end_tokens'].clone()
    next_indexes = next_obs['index_interval'].clone()
    next_done = torch.zeros(num_envs).to(device, non_blocking=True)
    num_updates = 100
    best_makespan = defaultdict(lambda: float('inf'))
    best_solution = defaultdict(lambda: list)

    per_worker_episode = np.zeros((num_envs), dtype=int)
    total_nb_episodes = 0
    temporary_action_history = [[] for _ in range(num_envs)]
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    missed_step_per_env = [[] for _ in range(num_envs)]
    to_worker = []
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = lr(frac)
            optimizer.param_groups[0]['lr'] = lrnow
            writer.add_scalar("learning_rate", lrnow, global_step)

        min_len = float('inf')
        while len(to_worker) < num_envs:
            with torch.no_grad() as a:
                all_actions, logprobas, _ = agent(next_interval_reps, next_attention_interval_masks,
                                                             next_job_resource_masks, next_mask, next_indexes, next_start_end_tokens)
            next_obs, rs, ds, infos = envs.step(all_actions)
            for env_idx, info in enumerate(infos):
                action_took = json.loads(info['action_took'])
                global_step += len(action_took)
                assert len(action_took) >= 1
                temporary_action_history[env_idx].append(action_took)

            next_mask = next_obs['action_mask'].clone()
            next_interval_reps = next_obs['interval_rep'].clone()
            next_job_resource_masks = next_obs['job_resource_mask'].clone()
            next_attention_interval_masks = next_obs['attention_interval_mask'].clone()
            next_start_end_tokens = next_obs['start_end_tokens'].clone()
            next_indexes = next_obs['index_interval'].clone()
            next_done = torch.IntTensor(ds).to(device, non_blocking=True)

            for env_idx, info in enumerate(infos):
                if 'episode' in info.keys():
                    instance = 'global'
                    writer.add_scalar(f"{instance}/episode_reward", info['episode']['r'], global_step)
                    writer.add_scalar(f"{instance}/episode_length", info['episode']['l'], global_step)
                    if 'env_name' in info.keys():
                        instance = info['env_name']
                        writer.add_scalar(f"{instance}/episode_reward", info['episode']['r'], global_step)
                        writer.add_scalar(f"{instance}/episode_length", info['episode']['l'], global_step)
                if 'makespan' in info.keys():
                    instance = info['env_name']
                    makespan = int(info['makespan'])
                    l.append(makespan)
                    iter_best_makespan = min(best_makespan[instance], makespan)
                    if iter_best_makespan < best_makespan[instance]:
                        best_makespan[instance] = iter_best_makespan
                        best_solution[instance] = json.loads(info['solution'])
                    writer.add_scalar(f"{instance}/makespan", makespan, global_step)
                    writer.add_scalar(f"{instance}/best_makespan", best_makespan[instance], global_step)
                    writer.add_scalar("global/best_makespan", min(best_makespan.values()), global_step)
                    writer.add_scalar("optim_criteria", mean(l), global_step)
                    if len(temporary_action_history[env_idx]) > 0:
                        to_worker.append((fn_env[env_idx], list(temporary_action_history[env_idx]), json.loads(info['solution']), makespan))
                        min_len = min(min_len, len(temporary_action_history[env_idx]))
                    temporary_action_history[env_idx] = []
                    total_nb_episodes += 1
        selected_ob_idx = random.randint(1, min_len - 1)
        workers = []
        agent = agent.cpu()
        for worker in to_worker:
            fn_worker, worker_selected_history, worker_solution, worker_makespan = worker
            selected_history = list(worker_selected_history[:selected_ob_idx])
            workers.append(
                compute_trajectory.remote(fn_worker, selected_history, worker_solution, worker_makespan, agent,
                                          list(worker_selected_history),
                                          time_limit=args.cp_solving_time + num_updates))
        total_improvements = 0
        all_collected_experiences = []
        all_collected_improvements = []
        all_collected_makespans = []
        instance_worker = collections.defaultdict(list)
        instance_makespans = collections.defaultdict(list)
        instance_improvements = collections.defaultdict(list)
        for idx in range(0, len(workers), args.num_workers):
            output_iter = ray.get(workers[idx:idx + args.num_workers])
            output_iter = np.array(output_iter, dtype=object)
            experiences = np.array(output_iter[:, 0], dtype=Experience)
            improvements = np.array(output_iter[:, 1], dtype=float)
            makespan_cps = np.array(output_iter[:, 2], dtype=float)
            instance_names = np.array(output_iter[:, 3], dtype=str)
            total_improvements += improvements.sum()
            for experience, improvement, makespan, name in zip(experiences, improvements, makespan_cps, instance_names):
                if experience is not None:
                    all_collected_experiences.append(experience)
                    all_collected_improvements.append(improvement)
                    instance_worker[name].append(len(all_collected_makespans))
                    all_collected_makespans.append(makespan)
                    instance_makespans[name].append(makespan)
                    instance_improvements[name].append(improvement)
        instance_mean_makespan = {k: np.mean(v) for k, v in instance_makespans.items()}
        instance_std_makespan = {k: np.std(v) for k, v in instance_makespans.items()}
        all_collected_makespans_advantage = np.zeros_like(all_collected_makespans)
        all_collected_makespans = np.array(all_collected_makespans)
        for name, instance_idxes in instance_worker.items():
            all_collected_makespans_advantage[instance_idxes] = (all_collected_makespans[instance_idxes] -
                                                                 instance_mean_makespan[name]) / (
                                                                            instance_std_makespan[name] + 1e-8)
        instance_min_improvement = {k: np.min(v) for k, v in instance_improvements.items()}
        instance_max_improvement = {k: np.max(v) for k, v in instance_improvements.items()}
        all_collected_improvement_advantage = np.zeros_like(all_collected_improvements)
        all_collected_improvements = np.array(all_collected_improvements)
        for name, instance_idxes in instance_worker.items():
            all_collected_improvement_advantage[instance_idxes] = (all_collected_improvements[instance_idxes] -
                                                                   instance_min_improvement[name]) / ((
                                                                                                                  instance_max_improvement[
                                                                                                                      name] -
                                                                                                                  instance_min_improvement[
                                                                                                                      name]) + 1e-8)
        writer.add_scalar(f"avg_improvement_step", total_improvements / len(workers), global_step)
        workers = []
        agent = agent.to(device, non_blocking=True)
        all_interval_rep = []
        all_action_mask = []
        all_attention_interval_mask = []
        all_start_end_tokens = []
        all_resource_mask = []
        all_resource_mask_selected = []
        all_actions = []
        all_improvements = []
        all_log_probs = []
        all_indexes_interval = []
        for experience, improvement in zip(all_collected_experiences, all_collected_improvement_advantage):
            for train_obs, action_to_take, log_prob in experience.all_obs_actions_to_take:
                all_interval_rep.append(train_obs["interval_rep"])
                all_action_mask.append(train_obs["action_mask"])
                all_attention_interval_mask.append(train_obs["attention_interval_mask"])
                all_start_end_tokens.append(train_obs["start_end_tokens"])
                all_resource_mask.append(train_obs["job_resource_mask"])
                all_resource_mask_selected.append(train_obs["job_resource_mask_selected"])
                all_indexes_interval.append(train_obs["index_interval"])
                all_actions.append(action_to_take)
                all_improvements.append(improvement)
                all_log_probs.append(log_prob)
            for train_obs, action_to_take, log_prob in experience.all_bad_path:
                all_interval_rep.append(train_obs["interval_rep"])
                all_action_mask.append(train_obs["action_mask"])
                all_attention_interval_mask.append(train_obs["attention_interval_mask"])
                all_start_end_tokens.append(train_obs["start_end_tokens"])
                all_resource_mask.append(train_obs["job_resource_mask"])
                all_resource_mask_selected.append(train_obs["job_resource_mask_selected"])
                all_indexes_interval.append(train_obs["index_interval"])
                all_actions.append(action_to_take)
                all_improvements.append(-improvement)
                all_log_probs.append(log_prob)
        all_interval_rep = torch.FloatTensor(np.stack(all_interval_rep)).to(device, non_blocking=True)
        all_resource_mask = torch.BoolTensor(np.stack(all_resource_mask)).to(device, non_blocking=True)
        all_resource_mask_selected = torch.BoolTensor(np.stack(all_resource_mask_selected)).to(device,
                                                                                               non_blocking=True)
        all_action_mask = torch.FloatTensor(np.stack(all_action_mask)).to(device, non_blocking=True)
        all_attention_interval_mask = torch.FloatTensor(np.stack(all_attention_interval_mask)).to(device,
                                                                                                  non_blocking=True)
        all_start_end_tokens = torch.FloatTensor(np.stack(all_start_end_tokens)).to(device,
                                                                                                  non_blocking=True)
        all_actions = torch.LongTensor(np.stack(all_actions)).to(device, non_blocking=True)
        all_improvements = torch.FloatTensor(np.stack(all_improvements)).to(device, non_blocking=True)
        all_log_probs = torch.FloatTensor(np.stack(all_log_probs)).to(device, non_blocking=True)
        all_indexes_interval = torch.FloatTensor(np.stack(all_indexes_interval)).to(device, non_blocking=True)
        dataset = TensorDataset(all_interval_rep,
                                all_resource_mask,
                                all_resource_mask_selected,
                                all_action_mask,
                                all_improvements,
                                all_attention_interval_mask,
                                all_actions,
                                all_log_probs,
                                all_indexes_interval,
                                all_start_end_tokens)
        data_loader = DataLoader(dataset, batch_size=all_interval_rep.shape[0] // args.n_minibatch, shuffle=True)
        total_correct = 0
        total_examples = 0
        total_loss = 0
        iter_loss_idx = 0
        total_mean_kl = 0
        for iter_loss_idx in range(1, args.update_epochs + 1):
            mean_approx_kl = 0
            div_kl = 0
            for interval_rep, resource_mask, resource_mask_selected, action_mask, adv_improvement, attention_interval_mask, label, log_prob_action, indexes_inter, start_end_tokens in data_loader:
                optimizer.zero_grad(set_to_none=True)
                output, newlogproba, entropy = agent(interval_rep,
                                                                attention_interval_mask,
                                                                resource_mask,
                                                                action_mask,
                                                                indexes_inter,
                                                                start_end_tokens,
                                                                label)
                # Policy loss
                logratio = newlogproba - log_prob_action
                ratio = logratio.exp()

                # Policy loss
                pg_loss1 = -adv_improvement * ratio
                pg_loss2 = -adv_improvement * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = (torch.max(pg_loss1, pg_loss2)).mean()
                # pg_loss = (adv_makespan_ex * newlogproba).mean()
                pg_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    Y_hat = output.reshape((-1, output.shape[-1]))
                    preds = Y_hat.argmax(axis=1).type(label.dtype)
                    compare = (preds == label.reshape(-1)).type(torch.float32)
                    total_correct += compare.sum().cpu().item()
                    total_examples += compare.size(0)
                    total_loss += pg_loss.cpu().item()
                    # Stats
                    mean_approx_kl += ((ratio - 1) - logratio).mean()
                    div_kl += 1
            if args.kle_stop:
                if (mean_approx_kl / div_kl) > args.target_kl:
                    break
            total_mean_kl += mean_approx_kl / div_kl
        wandb.log({f"first_loss/cp_loss": total_loss / (len(data_loader) * 10),
                   f"first_loss/cp_acc": (total_correct * 100) / total_examples,
                   f"first_loss/iter_nb": iter_loss_idx,
                   f"first_loss/mean_approx_kl": total_mean_kl / iter_loss_idx}, step=global_step)
        del all_interval_rep
        del all_resource_mask
        del all_action_mask
        del all_actions
        del all_improvements
        del dataset
        del data_loader
        all_interval_rep = []
        all_action_mask = []
        all_attention_interval_mask = []
        all_start_end_tokens = []
        all_resource_mask = []
        all_resource_mask_selected = []
        all_actions = []
        all_advantage_makespan = []
        all_log_probs = []
        all_indexes_interval = []
        for experience, adv_makespan in zip(all_collected_experiences, all_collected_makespans_advantage):
            for train_obs, action_took, log_prob in experience.all_obs_actions_took:
                all_interval_rep.append(train_obs["interval_rep"])
                all_action_mask.append(train_obs["action_mask"])
                all_attention_interval_mask.append(train_obs["attention_interval_mask"])
                all_start_end_tokens.append(train_obs["start_end_tokens"])
                all_resource_mask.append(train_obs["job_resource_mask"])
                all_resource_mask_selected.append(train_obs["job_resource_mask_selected"])
                all_indexes_interval.append(train_obs["index_interval"])
                all_actions.append(action_took)
                all_advantage_makespan.append(adv_makespan)
                all_log_probs.append(log_prob)
        all_interval_rep = torch.FloatTensor(np.stack(all_interval_rep)).to(device, non_blocking=True)
        all_resource_mask = torch.BoolTensor(np.stack(all_resource_mask)).to(device, non_blocking=True)
        all_resource_mask_selected = torch.BoolTensor(np.stack(all_resource_mask_selected)).to(device,
                                                                                               non_blocking=True)
        all_action_mask = torch.FloatTensor(np.stack(all_action_mask)).to(device, non_blocking=True)
        all_attention_interval_mask = torch.FloatTensor(np.stack(all_attention_interval_mask)).to(device,
                                                                                                  non_blocking=True)
        all_start_end_tokens = torch.FloatTensor(np.stack(all_start_end_tokens)).to(device,
                                                                                                  non_blocking=True)
        all_actions = torch.LongTensor(np.stack(all_actions)).to(device, non_blocking=True)
        all_advantage_makespan = torch.FloatTensor(np.stack(all_advantage_makespan)).to(device, non_blocking=True)
        all_log_probs = torch.FloatTensor(np.stack(all_log_probs)).to(device, non_blocking=True)
        all_indexes_interval = torch.FloatTensor(np.stack(all_indexes_interval)).to(device, non_blocking=True)
        dataset = TensorDataset(all_interval_rep,
                                all_resource_mask,
                                all_resource_mask_selected,
                                all_action_mask,
                                all_advantage_makespan,
                                all_attention_interval_mask,
                                all_actions,
                                all_log_probs,
                                all_indexes_interval,
                                all_start_end_tokens)
        data_loader = DataLoader(dataset, batch_size=all_interval_rep.shape[0] // args.n_minibatch, shuffle=True)
        total_loss = 0
        iter_loss_idx = 0
        total_mean_kl = 0
        for iter_loss_idx in range(1, args.update_epochs + 1):
            mean_approx_kl = 0
            div_kl = 0
            for interval_rep, resource_mask, resource_mask_selected, action_mask, adv_makespan_ex, attention_interval_mask, label, log_prob_action, indexes_inter, start_end_tokens in data_loader:
                optimizer.zero_grad(set_to_none=True)
                _, newlogproba, entropy = agent(interval_rep,
                                                           attention_interval_mask,
                                                           resource_mask,
                                                           action_mask,
                                                           indexes_inter,
                                                           start_end_tokens,
                                                           label)
                # Policy loss
                logratio = newlogproba - log_prob_action
                ratio = logratio.exp()

                # Policy loss
                pg_loss1 = adv_makespan_ex * ratio
                pg_loss2 = adv_makespan_ex * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                # pg_loss = (adv_makespan_ex * newlogproba).mean()
                pg_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    total_loss += pg_loss.cpu().item()
                    # Stats
                    mean_approx_kl += ((ratio - 1) - logratio).mean()
                    div_kl += 1
            if args.kle_stop:
                if (mean_approx_kl / div_kl) > args.target_kl:
                    break
            total_mean_kl += mean_approx_kl / div_kl
        wandb.log({f"second_loss/cp_loss": total_loss / (len(data_loader) * 10),
                   f"second_loss/iter_nb": iter_loss_idx,
                   f"second_loss/mean_approx_kl": total_mean_kl / iter_loss_idx}, step=global_step)
        del all_interval_rep
        del all_resource_mask
        del all_action_mask
        del all_actions
        del all_advantage_makespan
        del dataset
        del data_loader

        to_worker = []

        checkpoint = {"model": agent.state_dict(),
                          "optimizer": optimizer.state_dict()}
        torch.save(checkpoint, 'checkpoint.pt')
        wandb.save('checkpoint.pt')
    envs.close()
    writer.close()
    ray.shutdown()