from typing import Tuple, Callable, Optional
from collections import OrderedDict

import gym
import torch
import numpy as np
import ray
from ray.rllib.env.base_env import BaseEnv, ASYNC_RESET_RETURN
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiEnvDict, EnvType, EnvID, MultiAgentDict
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from stable_baselines3.common.vec_env.util import obs_space_info, dict_to_obs

from MyDummyVecEnv import MyDummyVecEnv


@PublicAPI
class MyRemoteVectorEnv(BaseEnv):
    """Vector env that executes envs in remote workers.
    This provides dynamic batching of inference as observations are returned
    from the remote simulator actors. Both single and multi-agent child envs
    are supported, and envs can be stepped synchronously or async.
    You shouldn't need to instantiate this class directly. It's automatically
    inserted when you use the `remote_worker_envs` option for Trainers.
    """

    @property
    def observation_space(self):
        return self._observation_space

    def __init__(self, make_env: Callable[[int], EnvType], num_workers: int, env_per_worker: int, observation_space: Optional[gym.spaces.Space], device: torch.device):
        self.make_local_env = make_env
        self.num_workers = num_workers
        self.env_per_worker = env_per_worker
        self.num_envs = num_workers * env_per_worker
        self.poll_timeout = None

        self.actors = None  # lazy init
        self.pending = None  # lazy init

        self.observation_space = observation_space
        self.keys, shapes, dtypes = obs_space_info(self.observation_space)

        self.device = device

        self.buf_obs = OrderedDict(
            [(k, torch.zeros((self.num_envs,) + tuple(shapes[k]), dtype=torch.float, device=self.device)) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            self.buf_obs[key][env_idx * self.env_per_worker: (env_idx + 1) * self.env_per_worker] = torch.from_numpy(obs[key]).to(self.device,
                                                                                           non_blocking=True)

    def poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict,
                            MultiEnvDict, MultiEnvDict]:
        if self.actors is None:

            def make_remote_env(i):
                return _RemoteSingleAgentEnv.remote(self.make_local_env, i, self.env_per_worker)

            self.actors = [make_remote_env(i) for i in range(self.num_workers)]

        if self.pending is None:
            self.pending = {a.reset.remote(): a for a in self.actors}

        # each keyed by env_id in [0, num_remote_envs)
        ready = []

        # Wait for at least 1 env to be ready here
        while not ready:
            ready, _ = ray.wait(
                list(self.pending),
                num_returns=len(self.pending),
                timeout=self.poll_timeout)

        for obj_ref in ready:
            actor = self.pending.pop(obj_ref)
            env_id = self.actors.index(actor)
            ob, rew, done, info = ray.get(obj_ref)

            self._save_obs(env_id, ob)
            self.buf_rews[env_id * self.env_per_worker: (env_id + 1) * self.env_per_worker] = rew
            self.buf_dones[env_id * self.env_per_worker: (env_id + 1) * self.env_per_worker] = done
            self.buf_infos[env_id * self.env_per_worker: (env_id + 1) * self.env_per_worker] = info
        return (self._obs_from_buf(), self.buf_rews, self.buf_dones, self.buf_infos)

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, self.buf_obs)

    @PublicAPI
    def send_actions(self, action_list) -> None:
        for worker_id in range(self.num_workers):
            actions = action_list[worker_id * self.env_per_worker: (worker_id + 1) * self.env_per_worker]
            actor = self.actors[worker_id]
            obj_ref = actor.step.remote(actions)
            self.pending[obj_ref] = actor

    @PublicAPI
    def try_reset(self,
                  env_id: Optional[EnvID] = None) -> Optional[MultiAgentDict]:
        actor = self.actors[env_id]
        obj_ref = actor.reset.remote()
        self.pending[obj_ref] = actor
        return ASYNC_RESET_RETURN

    @PublicAPI
    def stop(self) -> None:
        if self.actors is not None:
            for actor in self.actors:
                actor.__ray_terminate__.remote()

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

@ray.remote(num_cpus=1)
class _RemoteSingleAgentEnv:
    """Wrapper class for making a gym env a remote actor."""

    def __init__(self, make_env, i, env_per_worker):
        self.env = MyDummyVecEnv([lambda: make_env((i * env_per_worker) + k) for k in range(env_per_worker)])

    def reset(self):
        return self.env.reset(), 0, False, {}

    def step(self, actions):
        return self.env.step(actions)
