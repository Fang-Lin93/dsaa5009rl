
import gym
from agents.agent import Agent, Batch, InfoDict


class RandomAgent(Agent):
    name = 'random agent without training'

    def __init__(self, action_space: gym.spaces.box.Box):
        self.action_space = action_space

    def update(self, batch: Batch) -> InfoDict:
        pass

    def sample_actions(self, observations=None):
        return self.action_space.sample()
