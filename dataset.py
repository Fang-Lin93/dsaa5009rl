import d4rl
import gym
import collections
import wrappers
from tqdm import tqdm
import numpy as np
from typing import Tuple, Optional
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


class D4RLDataset(object):

    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            """
            Here the dones signal is given if the observation does not change.
            Dones signal is used to separate trajectories
            """
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        self.observations = dataset['observations'].astype(np.float32)
        self.actions = dataset['actions'].astype(np.float32)
        self.rewards = dataset['rewards'].astype(np.float32)
        self.masks = 1.0 - dataset['terminals'].astype(np.float32)
        self.dones_float = dones_float.astype(np.float32)
        self.next_observations = dataset['next_observations'].astype(
            np.float32)
        self.size = len(dataset['observations'])

    def sample(self, batch_size: int) -> Batch:
        indices = self.sample_n_k(self.size, batch_size)
        return Batch(observations=self.observations[indices],
                     actions=self.actions[indices],
                     rewards=self.rewards[indices],
                     masks=self.masks[indices],
                     next_observations=self.next_observations[indices], )

    @staticmethod
    def sample_n_k(n, k):
        """Sample k distinct elements uniformly from range(n)"""
        """it is faster to get replace=False"""

        if not 0 <= k <= n:
            raise ValueError("Sample larger than population or is negative")
        if k == 0:
            return np.empty((0,), dtype=np.int64)
        elif 3 * k >= n:
            return np.random.choice(n, k, replace=False)
        else:
            result = np.random.choice(n, 2 * k)
            selected = set()
            selected_add = selected.add
            j = k
            for i in range(k):
                x = result[i]
                while x in selected:
                    x = result[i] = result[j]
                    j += 1
                    if j == 2 * k:
                        # This is slow, but it rarely happens.
                        result[k:] = np.random.choice(n, k)
                        j = k
                selected_add(x)
            return result[:k]


def compute_returns(traj):
    episode_return = 0
    for _, _, rew, _, _, _ in traj:
        episode_return += rew

    return episode_return


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]
    returns = []
    episode_return = 0

    for i in tqdm(range(len(observations)), desc='split to trajectories'):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        episode_return += rewards[i]
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])
            returns.append(episode_return)
            episode_return = 0
    return trajs, returns


def traj_return_normalize(dataset, scale=None):
    """
    iql_normalize: normalized reward <- reward /(max_return-min_return)* 1000.0
    seed https://github.com/ikostrikov/implicit_q_learning/blob/master/train_offline.py
    """
    trajs, returns = split_into_trajectories(dataset.observations, dataset.actions,
                                             dataset.rewards, dataset.masks,
                                             dataset.dones_float,
                                             dataset.next_observations)

    if scale is None:  # using the average trajectory length
        scale = int(np.mean([len(_) for _ in trajs]))
    assert scale > 0

    trajs.sort(key=compute_returns)
    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= scale

    return returns


def make_env_and_dataset(env_name: str,
                         seed: int = 0,
                         video_save_folder: str = None) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)  # -> np.float32

    if video_save_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_save_folder)

    # set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    # reward normalization  "iql_locomotion"
    traj_return_normalize(dataset, scale=1000.)

    return env, dataset


def make_env(env_name: str,
             seed: int,
             video_save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name)
    else:
        domain_name, task_name = env_name.split('-')
        env = wrappers.DMCEnv(domain_name=domain_name,
                              task_name=task_name,
                              task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if video_save_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_save_folder)

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == 'quadruped' else 0
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only,
                                      render_kwargs={
                                          'pixels': {
                                              'height': image_size,
                                              'width': image_size,
                                              'camera_id': camera_id
                                          }
                                      })
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
