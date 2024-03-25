
import random
import numpy as np
from agents.agent import Agent
from tqdm import trange


class SuperTicTacToeEnv(object):
    """
    The game board is of size 6x6.
    The first player who get 4 in a row/column/across the diagonal is the winner.
    The state is of shape (6, 6, 2), with 0 means the current player and 1 means the opponent.
    The action is an integer between 0 and 35.
    """

    def __init__(self, random_place_prob=0.):
        self.winner = -1
        self.player_id = 0  # current player id
        self.board = np.zeros((6, 6, 2))
        self.is_game_over = False

        self.random_place_prob = random_place_prob
        self.perturbations = [(-1, -1), (-1, 1), (1, -1), (1, 1),
                              (0, -1), (0, 1), (-1, 0), (1, 0)]

    def reset(self, rand_start=True):
        self.winner = -1
        self.player_id = random.randint(0, 1) if rand_start else 0
        self.board = np.zeros((6, 6, 2))
        self.is_game_over = False
        return self.get_state()

    def jump(self, board: np.ndarray, player_id: int):
        self.winner = -1
        self.player_id = player_id
        self.board = board
        self.is_game_over = False

    def step(self, action):
        row, col = int(action // 6), int(action % 6)
        occupied = self.board.sum(axis=-1).reshape(-1)
        assert occupied[action] == 0, f"The position ({row}, {col}) is occupied !"

        if np.random.random() > self.random_place_prob:
            self.board[row, col, self.player_id] += 1  # selection accepted
        else:
            dr, dc = self.perturbations[np.random.choice(range(8))]
            row += dr
            col += dc
            # row += 1 - int(np.random.random() < 0.5) * 2
            # col += 1 - int(np.random.random() < 0.5) * 2  # neighborhood perturbation
            if 0 <= row <= 5 and 0 <= col <= 5:  # inside the board
                action_id = row * 6 + col
                if occupied[action_id] == 0:  # not occupied
                    self.board[row, col, self.player_id] += 1

        rewards = [0, 0]

        if self.check_winner():
            self.winner = self.player_id
            self.is_game_over = True
            rewards[self.winner] = 1
            rewards[1 - self.winner] = -1
            # method1: winner -> 1, loser -> -1, tie -> 0
            # method2: winner -> 10, loser -> 0, tie -> 0
            ...

        if self.board.sum(axis=-1).all():  # no place to take actions
            self.is_game_over = True

        self.player_id = 1 - self.player_id
        return self.get_state(), rewards

    def get_legal_actions(self) -> list:
        vec_board = self.board.reshape(-1, 2)
        return [i_ for i_ in range(36) if not any(vec_board[i_, :])]

    def get_state(self) -> tuple[np.ndarray, np.ndarray, int, bool]:
        """
        :return: board, legal_actions(1=legal, 0=occupied), player_id, done
        """
        assert self.board.max() <= 1
        return self.board, 1-self.board.sum(axis=-1).reshape(-1), self.player_id, self.is_game_over

    def check_winner(self) -> bool:
        """
        check whether the current player is the winner
        """
        board = self.board[:, :, self.player_id]

        for i in range(6):  # i-th row
            # row or column
            if any(sum(board[i, j:j+4]) >= 4 for j in range(3)) or any(sum(board[j:j+4, i]) >= 4 for j in range(3)):
                return True
            # diagonally
            if i < 3:
                if any(sum(board[i+j, k+j] for j in range(4)) >= 4 for k in range(3)):  # done-side
                    return True

                if any(sum(board[i+j, 5-j-k] for j in range(4)) >= 4 for k in range(3)):  # up-side
                    return True
        return False

    @staticmethod
    def get_winner_pattern():
        patterns = []
        for i in range(6):  # i-th row
            # row or column
            for j in range(3):
                board = np.zeros((6, 6))
                board[i, j:j + 4] = 1
                patterns.append(board)

                board = np.zeros((6, 6))
                board[j:j + 4, i] = 1
                patterns.append(board)

        for i in range(3):
            for k in range(3):
                board_done = np.zeros((6, 6))
                board_up = np.zeros((6, 6))
                for j in range(4):
                    board_done[i + j, k + j] = 1
                    board_up[i + j, 5 - j - k] = 1
                patterns.append(board_done)
                patterns.append(board_up)
        return patterns

    def play_agent(self, agent: Agent):
        if hasattr(agent, 'training'):
            agent.training = False

        board, la, player_id, done = self.reset()
        print("Human player id = 0 ('O'), AI player id = 1  ('X')")
        print("New game:")
        self.render(show_act_idx=True)
        while not done:
            if player_id == 0:  # human player
                action = int(input('Please enter your action index:'))
                while not 0 <= action <= 35 or la[action] == 0:
                    action = int(input(f'Please enter your action index: (position {action} is occupied!)'))
            else:
                action = agent.sample_actions(observations=board[:, :, [player_id, 1 - player_id]],
                                              legal_actions=la)
            (board, la, player_id, done), rewards = self.step(action)
            print(f"Player{1-player_id} choose action={action}:")
            self.render(show_act_idx=True)

        if self.winner > -1:
            print(f"Player {self.winner} wins!")

    def render(self, board=None, mark: int = 0, show_act_idx: bool = False):
        if board is None:
            board = self.board

        if board.ndim < 3:
            board = np.concatenate([board[:, :, np.newaxis], np.zeros((6, 6, 1))], axis=-1)
            if mark != 0:
                board = board[:, :, [1, 0]]

        board_str = ''
        for row in range(6):
            board_str += '|'
            for col in range(6):
                if board[row, col, 0] > 0:
                    board_str += '  O '
                elif board[row, col, 1] > 0:
                    board_str += '  X '
                elif show_act_idx:
                    board_str += f'[{row*6+col}]'.rjust(4)
                else:
                    board_str += '  _ '
            board_str += '|\n'

        print(board_str)

    def self_check(self, num_check_runs: int = 10000):

        # before check: set the environment to be deterministic
        __random_place_prob = self.random_place_prob
        __winner_patters = self.get_winner_pattern()
        self.random_place_prob = 0

        for _ in trange(num_check_runs):
            board, _, _, done = self.reset()
            while not done:
                a = np.random.choice(self.get_legal_actions())
                (board, _, _, done), _ = self.step(a)

            if self.winner == -1:
                assert not self.get_legal_actions(), "No legal actions should be available"
            else:
                assert any((board[:, :, self.winner] * __winner_patters).sum(axis=-1).sum(axis=-1) >= 4), "winner error"

        print("check passed!")
        self.random_place_prob = __random_place_prob


def evaluate_tic_tac_toe(agents, env: SuperTicTacToeEnv, num_episodes: int) -> (
        tuple)[list, np.ndarray, np.ndarray, np.ndarray]:
    assert len(agents) == 2
    wins, returns, lengths, ties, info = [], [], [], [], {}
    for _ in range(num_episodes):
        board, la, player_id, done = env.reset()
        rewards = [0, 0]
        episode_len = 0

        while not done:
            observations = board[:, :, [player_id, 1 - player_id]]
            action = agents[player_id].sample_actions(observations, la)
            (board, la, player_id, done), rewards = env.step(action)
            episode_len += 1
        # episodic statistics from wrappers/EpisodeMonitor
        wins.append(env.winner)
        ties.append(int(env.winner < 0))
        returns.append(rewards)
        lengths.append(episode_len)

    returns, ties, lengths = np.array(returns), np.array(ties), np.array(lengths)
    return wins, returns, ties, lengths


if __name__ == '__main__':
    # from agents import RandomAgent

    self = SuperTicTacToeEnv()
    winners = []
    # agent = RandomAgent()

    # evaluate_tic_tac_toe([RandomAgent(), RandomAgent()], self, num_episodes=100)

    for t_ in trange(1000):
        obs, la, _, done = self.reset()
        while not done:
            a = np.random.choice(self.get_legal_actions())
            (obs, la, _, done), _ = self.step(a)
            # la = self.get_legal_actions()
            # done = state['done']
        # if self.winner == -1:
        #     break

        winners.append(self.winner)
