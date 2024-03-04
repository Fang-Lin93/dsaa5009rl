import os

import numpy as np
import torch
import gym
from collections import deque
from tensorboardX import SummaryWriter
from dataset import make_env_and_dataset
from tqdm import trange
from rlagents import TorchBCLearner, JAXBCLearner
from eval import eval_agent, STATISTICS
from utils import prepare_output_dir, set_torch_seed
import argparse


parser = argparse.ArgumentParser(description='Offline Reinforcement Learning')
parser.add_argument('--env', default='hopper-medium-v2', help='name of the environment')
parser.add_argument('--agent', default='torchBC', help='name of the environment')
parser.add_argument('--create_video', action='store_true')
# buffer
parser.add_argument('--seed', default=520, type=int, help='seed')
parser.add_argument('--max_steps', default=100000, type=int, help='maximal number of gradient steps')
parser.add_argument('--eval_interval', default=5000, type=int,
                    help="evaluate the agent every 'eval_interval' gradient steps")
parser.add_argument('--log_interval', default=1000, type=int,
                    help="record the training statistics, such as loss every 'log_interval' gradient steps")
parser.add_argument('--num_eval_episodes', default=10, type=int,
                    help="number of evaluation episodes for each evaluation. Should be >= 10 for stability")
parser.add_argument('--batch_size', default=256, type=int, help='batch_size')


def run():
    args = parser.parse_args()
    set_torch_seed(args.seed)

    # create a saving directory
    save_dir = prepare_output_dir(suffix="Behavior-Cloning")
    with open(os.path.join(save_dir, f"seed_{args.seed}.txt"), "w") as f:
        print("\t".join(["steps"] + STATISTICS), file=f)
    summary_writer = SummaryWriter(os.path.join(save_dir, 'tensorboard', f'seed={args.seed}'))
    env, dataset = make_env_and_dataset(args.env, seed=args.seed)

    # create an agent
    obs_dim = len(env.observation_space.sample())
    act_dim = len(env.action_space.sample())

    if args.agent == 'torchBC':
        agent = TorchBCLearner(obs_dim=obs_dim,
                               act_dim=act_dim,
                               actor_lr=3e-4,
                               layer_norm=True,
                               hidden_dims=(256, 256, 256),
                               lr_decay_T=args.max_steps,
                               device=torch.device('cpu'))
    elif args.agent == 'jaxBC':
        agent = JAXBCLearner(seed=args.seed,
                             obs_dim=obs_dim,
                             act_dim=act_dim,
                             actor_lr=3e-4,
                             layer_norm=True,
                             hidden_dims=(256, 256, 256),
                             lr_decay_T=args.max_steps,)
    else:
        # TODO: initialize your own agent here
        raise NotImplementedError

    latest_mean_returns = deque(maxlen=5)  # track the performance of the latest 5 evaluation
    for i in trange(args.max_steps):

        # evaluation
        if i % args.eval_interval == 0:
            eval_res = eval_agent(i, agent, env, summary_writer, save_dir, args.seed, args.num_eval_episodes)
            latest_mean_returns.append(eval_res['mean'])
            print(f"Step={i}, Eval Mean={eval_res['mean']}")

        # training process
        batch = dataset.sample(args.batch_size)
        update_info = agent.update(batch)

        # record the training information
        if i % args.log_interval == 0:
            for k, v in update_info.items():
                summary_writer.add_scalar(f'training/{k}', v, i)
            summary_writer.flush()

    print(f"Final Mean Return={np.mean(latest_mean_returns)}")

    if args.create_video:
        print("Saving video...")
        # create the video
        env = gym.make("Hopper-v2")
        env = gym.wrappers.RecordVideo(env, save_dir)
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation)  # eval takes argmax from actor net
            observation, _, done, info = env.step(np.clip(action, -1, 1))
        env.close()

    return


if __name__ == '__main__':
    run()
