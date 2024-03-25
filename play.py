from envs import SuperTicTacToeEnv
from agents.torch_dqn import DQNLearner as TorchDQN
from agents.jax_dqn import DQNLearner as JaxDQN

env = SuperTicTacToeEnv()


# use Torch DQN
agent = TorchDQN(seed=0,
                 hidden_dims=(64, 64, 64, 1),
                 obs_shape=(6, 6, 2),
                 act_dim=36,
                 lr_decay_steps=100000,
                 greedy_max=1,
                 greedy_min=0.1,
                 greedy_decay_steps=100000,
                 device='cpu')

agent.load_ckpt(ckpt_folder='ckpt',
                prefix=f'finished_20240324-161543')


# use Jax DQN
agent = JaxDQN(seed=0,
               hidden_dims=(64, 64, 64, 1),
               obs_shape=(6, 6, 2),
               act_dim=36,
               lr_decay_steps=100000,
               greedy_max=1,
               greedy_min=0.1,
               greedy_decay_steps=100000,
               device='cpu')

agent.load_ckpt(ckpt_folder='ckpt',
                prefix=f'finished_20240324-195941')

env.play_agent(agent)
