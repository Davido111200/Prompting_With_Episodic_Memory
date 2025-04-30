import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.wrappers.clip_action import ClipAction
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

from typing import List, Callable, Optional, Union, Any, Type, Sequence
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

import sys
from src.data_utils import custom_load_dataset
from src.utils import *
from src.vec_env import PromptEnv
try:
    import dmc2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass




class DummyVecEnv1(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.
    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], num_processes):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((num_processes,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((num_processes,), dtype=bool)
        self.buf_rews = np.zeros((num_processes,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews, self.buf_dones, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                # self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.
        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.
        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                # self.buf_obs[key][env_idx] = obs
                self.buf_obs[key] = obs
            else:
                # self.buf_obs[key][env_idx] = obs[key]
                self.buf_obs[key] = obs[key]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        if device == -1:
            self.device = 'cpu'
        else:
            self.device = 'cuda:'+str(device)
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        # print(obs.shape, reward.shape, done.shape)
        return obs, reward, done, info

def make_env(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                    temperature, top_p, n_candidates, n_slots, epsilon, num_processes, i, gpu_id, evaluate, external_memory, external_memory_random, external_memory_extended, \
                    memory_verbalizer, memory_verbalizer_random, memory_verbalizer_extended,
                    ic_examples, ic_labels, model, tokenizer):
    def _thunk():
        env = PromptEnv(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                        temperature, top_p, n_candidates, n_slots, epsilon, \
                        num_processes, i, gpu_id, evaluate=evaluate, external_memory=external_memory, external_memory_random=external_memory_random,\
                        external_memory_extended=external_memory_extended, memory_verbalizer=memory_verbalizer, memory_verbalizer_random=memory_verbalizer_random,\
                        memory_verbalizer_extended=memory_verbalizer_extended,\
                        ic_examples=ic_examples, ic_labels=ic_labels, model=model, tokenizer=tokenizer)
        
        print('Finish Build Environment')
        
        return env

    return _thunk


def make_vec_envs(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                        temperature, top_p, n_candidates, n_slots, epsilon, \
                        num_processes, i, gpu_id, evaluate, external_memory, external_memory_random, external_memory_extended, \
                        memory_verbalizer, memory_verbalizer_random, memory_verbalizer_extended, ic_examples, ic_labels, model, tokenizer):
    # Maybe we should just load a part of the dataset here, so that we dont have to put all dataset to cuda device  


    envs = [
        make_env(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                    temperature, top_p, n_candidates, n_slots, epsilon, num_processes, i, gpu_id, evaluate=evaluate, \
                    external_memory=external_memory, external_memory_random=external_memory_random, external_memory_extended=external_memory_extended, \
                    memory_verbalizer=memory_verbalizer, memory_verbalizer_random=memory_verbalizer_random,\
                    memory_verbalizer_extended=memory_verbalizer_extended,
                    ic_examples=ic_examples, ic_labels=ic_labels, model=model, tokenizer=tokenizer)
    ]

    envs = DummyVecEnv1(envs, num_processes)
    envs = VecPyTorch(envs, gpu_id)

    return envs


def make_vec_envs_eval(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                    temperature, top_p, n_candidates, n_slots, epsilon, num_processes, i, gpu_id, evaluate, external_memory, external_memory_random,\
                    external_memory_extended, memory_verbalizer, memory_verbalizer_random, memory_verbalizer_extended, \
                    ic_examples, ic_labels, model, tokenizer):

    envs = [
        make_env_eval(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                    temperature, top_p, n_candidates, n_slots, epsilon, num_processes, i, gpu_id, evaluate=evaluate,\
                    external_memory=external_memory, external_memory_random=external_memory_random, external_memory_extended=external_memory_extended,\
                    memory_verbalizer=memory_verbalizer, memory_verbalizer_random=memory_verbalizer_random,\
                    memory_verbalizer_extended=memory_verbalizer_extended,\
                    ic_examples=ic_examples, ic_labels=ic_labels, model=model, tokenizer=tokenizer)
    ]

    envs = DummyVecEnv1(envs, num_processes)

    print("Finished Build Eval Environments")

    return envs



def make_env_eval(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                        temperature, top_p, n_candidates, n_slots, epsilon, num_processes, i, gpu_id, evaluate, external_memory, external_memory_random,\
                        external_memory_extended, memory_verbalizer, memory_verbalizer_random, memory_verbalizer_extended,\
                        ic_examples, ic_labels, model, tokenizer):
    def _thunk():
        # split the data base on number of gpus
        num_examples = int(len(valid_sentences)/args.num_actors) - 1
        
        print(f"Current actor {i} has examples start from {i*num_examples} to {(i+1)*num_examples}")

        valid_sentences_, valid_labels_ = valid_sentences[i*num_examples:(i+1)*num_examples], valid_labels[i*num_examples:(i+1)*num_examples]
        env = PromptEnv(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences_, valid_labels_, \
                        temperature, top_p, n_candidates, n_slots, epsilon, \
                        num_processes, i, gpu_id, evaluate=evaluate, external_memory=external_memory, external_memory_random=external_memory_random, \
                        external_memory_extended=external_memory_extended, memory_verbalizer=memory_verbalizer, memory_verbalizer_random=memory_verbalizer_random,\
                        memory_verbalizer_extended=memory_verbalizer_extended,\
                        ic_examples=ic_examples, ic_labels=ic_labels, model=model, tokenizer=tokenizer)
        
        print('Environment actor ', i, ' on gpu ', gpu_id, flush=True)
        
        return env

    return _thunk



def make_vec_envs_fseval(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                        temperature, top_p, n_candidates, n_slots, epsilon, num_processes, i, gpu_id, evaluate, external_memory, external_memory_random,\
                        external_memory_extended, memory_verbalizer, memory_verbalizer_random, memory_verbalizer_extended,\
                        ic_examples, ic_labels, model, tokenizer):

    envs = [
        make_env_fseval(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                        temperature, top_p, n_candidates, n_slots, epsilon, \
                        num_processes, i, gpu_id, evaluate=evaluate, external_memory=external_memory, external_memory_random=external_memory_random, \
                        external_memory_extended=external_memory_extended, memory_verbalizer=memory_verbalizer, memory_verbalizer_random=memory_verbalizer_random,\
                        memory_verbalizer_extended=memory_verbalizer_extended,\
                        ic_examples=ic_examples, ic_labels=ic_labels, model=model, tokenizer=tokenizer)
    ]

    envs = DummyVecEnv1(envs, num_processes)

    return envs



def make_env_fseval(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                    temperature, top_p, n_candidates, n_slots, epsilon, num_processes, i, gpu_id, evaluate, external_memory, external_memory_random,\
                    external_memory_extended, memory_verbalizer, memory_verbalizer_random, memory_verbalizer_extended, ic_examples, ic_labels, model, tokenizer):
    def _thunk():
        num_examples = len(valid_sentences)

        print(f"Few-shot env is on gpu {gpu_id} and actor {i} has {num_examples} examples")

        env = PromptEnv(params, args, reward_llm, tokenizer_llm, train_sentences, train_labels, valid_sentences, valid_labels, \
                        temperature, top_p, n_candidates, n_slots, epsilon, \
                        num_processes, i, gpu_id, evaluate=evaluate, external_memory=external_memory, external_memory_random=external_memory_random, \
                        external_memory_extended=external_memory_extended, memory_verbalizer=memory_verbalizer, memory_verbalizer_random=memory_verbalizer_random, \
                        memory_verbalizer_extended=memory_verbalizer_extended, ic_examples=ic_examples, ic_labels=ic_labels, model=model, tokenizer=tokenizer)
        
        print('Finish Build Few-shot Environment')
        
        return env
    return _thunk

def get_num_test(seed,
        params, 
        max_steps, 
        num_processes,
        gamma, 
        obs_size,
        i,
        gpu_id=0):
        
    _, _, all_test_sentences, all_test_labels = custom_load_dataset(params)
    num_examples = int(len(all_test_sentences)/params['num_actors']) - 1

    return num_examples