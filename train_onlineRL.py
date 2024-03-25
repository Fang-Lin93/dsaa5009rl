from envs import SuperTicTacToeEnv
import os
import datetime
import random
import numpy as np
from tqdm import tqdm, trange

from agents import torch_dqn
from agents.agent import Agent
from agents.jax_dqn import DQNLearner as JaxDQN
from agents.torch_dqn import DQNLearner as TorchDQN
from envs.super_tic_tac_toe import evaluate_tic_tac_toe
from datasets.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='Reinforcement Learning (SuperTicTacToe)')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--agent', default='jax_dqn', type=str, help='The agent')
parser.add_argument('--discount', default=0.95, type=float, help='The discount rate')
parser.add_argument('--max_steps', default=1_000_000, type=int, help='maximal number of episodes')
parser.add_argument('--buffer_size', default=1_000_000, type=int, help='capacity of the replay buffer')
parser.add_argument('--update_interval', default=100, type=int,
                    help="train the agent every 'eval_interval' action steps")
parser.add_argument('--log_interval', default=5000, type=int,
                    help="log the statistics every 'eval_interval' action steps")
parser.add_argument('--eval_interval', default=10000, type=int,
                    help="evaluate the agent every 'eval_interval' action steps")
parser.add_argument('--num_eval', default=100, type=int,
                    help="number of evaluation episodes for each evaluation.")
parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
parser.add_argument('--device', default=-1, type=int, help='-1 means cpu')


class RandomAgent(Agent):
    name = 'random_agent'

    def __init__(self):
        pass

    def sample_actions(self, observations, legal_actions) -> int:
        return np.random.choice(legal_actions.reshape(-1).nonzero()[0])

    def update_state(self, state: dict) -> None:
        pass


def run():
    # set random seeds
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create environemtns
    env = SuperTicTacToeEnv(random_place_prob=0.)
    eval_env = SuperTicTacToeEnv(random_place_prob=0.)

    # create replay buffer with board symmetry augmentations
    replay_buffer = ReplayBuffer(obs_shape=(6, 6, 2),
                                 capacity=args.buffer_size,
                                 rotation_expand_k=3)  # rotation symmetries

    # define the opponent agent for testing, here I only have a random agent
    ts_agent = RandomAgent()

    # define Double DQN agent for learning

    Learner = {'torch_dqn': TorchDQN,
               'jax_dqn': JaxDQN}[args.agent]

    agent = Learner(seed=0,
                    hidden_dims=(64, 64, 64, 1),
                    obs_shape=(6, 6, 2),
                    act_dim=36,
                    lr_decay_steps=args.max_steps // args.update_interval,
                    greedy_max=1,
                    greedy_min=0.1,
                    greedy_decay_steps=args.max_steps // args.update_interval,
                    discount=args.discount,
                    device='cpu' if args.device < 0 else f'cuda:{args.device}')

    agent.training = True
    # define process recorder, here I name the save directory with the datetime for easy tracking
    time_str = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_dir = os.path.join('results', 'tensorboard', agent.name, time_str)
    print(f"results are saved in {save_dir}")
    summary_writer = SummaryWriter(save_dir)

    # process bar
    num_steps = 0
    bar = tqdm(total=args.max_steps,
               desc=f'Train Agent={agent.name}',
               smoothing=0.01,
               colour='GREEN')

    best_win_rate = 0.

    # main train/test process
    while num_steps < args.max_steps:
        board, la, player_id, done = env.reset()

        # record transitions for saving to replay buffer
        obs = [None, None]
        actions = [None, None]
        next_obs = [None, None]
        obs[player_id] = board[:, :, [player_id, 1 - player_id]]

        # each game process
        while not done:
            action = agent.sample_actions(observations=board[:, :, [player_id, 1 - player_id]],
                                          legal_actions=la)

            actions[player_id] = action

            # in super-TTT game: rewards = +1 for winner, -1 for loser, 0 for tie or continuing
            (board, la, player_id, done), rewards = env.step(action)
            next_obs[player_id] = board[:, :, [player_id, 1 - player_id]]

            if done:
                # add opponent's transition
                replay_buffer.insert(observation=obs[player_id],
                                     action=actions[player_id],
                                     reward=rewards[player_id],
                                     next_observation=board[:, :, [player_id, 1 - player_id]],
                                     mask=0)

                # add final hand's transition
                replay_buffer.insert(observation=obs[1 - player_id],
                                     action=actions[1 - player_id],
                                     reward=rewards[1 - player_id],
                                     next_observation=board[:, :, [1 - player_id, player_id]],
                                     mask=0)

            else:
                if obs[player_id] is not None and next_obs[player_id] is not None:
                    replay_buffer.insert(observation=obs[player_id],
                                         action=actions[player_id],
                                         reward=rewards[player_id],
                                         next_observation=next_obs[player_id],
                                         mask=1)  # continue the game

                obs[player_id] = next_obs[player_id]

            num_steps += 1
            if num_steps < args.max_steps:
                bar.update(1)

            if num_steps % args.update_interval == 0:
                batch = replay_buffer.sample(batch_size=args.batch_size)
                if batch:
                    update_info = agent.update(batch)
                    # record the training information
                    if num_steps % args.log_interval == 0:
                        for k, v in update_info.items():
                            summary_writer.add_scalar(f'training/{k}', v, num_steps)
                        summary_writer.flush()

            if num_steps % args.eval_interval == 0:
                agent.training = False
                # test against itself
                _, _, ties, _ = evaluate_tic_tac_toe([agent, agent], eval_env, num_episodes=args.num_eval)
                eval_info = {
                    'self_tie_rate': np.mean(ties)
                }
                # test against the opponent agent
                wins, returns, _, lens = evaluate_tic_tac_toe([agent, ts_agent], eval_env, num_episodes=args.num_eval)
                eval_info['mean_return'] = returns.mean(axis=0)[0]
                eval_info['return_std'] = returns.std(axis=0)[0]
                eval_info['winning_rate'] = wins.count(0)/len(wins)
                eval_info['episode_lens'] = lens.mean()

                for k, v in eval_info.items():
                    summary_writer.add_scalar(f'eval/{k}', v, num_steps)
                summary_writer.flush()
                agent.training = True

                if eval_info['winning_rate'] >= best_win_rate:
                    agent.save_ckpt(ckpt_folder=os.path.join(os.getcwd(), 'ckpt'),
                                    prefix=f'best_{time_str}')
                    best_win_rate = eval_info['winning_rate']

    agent.save_ckpt(ckpt_folder=os.path.join(os.getcwd(), 'ckpt'),
                    prefix=f'finished_{time_str}')


if __name__ == '__main__':
    run()
