## DSAA5009: Offline Reinforcement Learining project.

In this project, you need to train an agent to learn to do continuous control on the 
[D4RL](https://github.com/digital-brain-sh/d4rl) dataset, which is a standard dataset for offline 
reinforcement learning. There are many datasets in D4RL. Here we focus on the Mujoco locomotion task
[hopper](https://www.gymlibrary.dev/environments/mujoco/hopper/), in which you are required to use the offline dataset 
[hopper-medium-v2](https://github.com/Farama-Foundation/d4rl/wiki/Tasks) (1M samples from a policy trained to 
approximately 1/3 the performance of the expert) to train the agent to control the one-leg entity to hop forward:

![](hopper.gif)

Compared to online reinforcement learning tasks, offline reinforcement learning tasks present a greater challenge due
to the agent's reliance solely on available data, precluding its interaction with the environment to obtain performance 
evaluations and modify its actions. While online reinforcement learning algorithms can be employed to train
the agent, caution must be exercised when the agent takes actions that lack feedback from the offline dataset. This is 
because in the context of offline reinforcement learning, emphasizing actions with spurious high-yield actions results 
in the accumulation of errors and, ultimately, the failure of the performance. In the hopper task, 
such a failure would cause the one-leg to fall down.

Prior to commencing the project, you need to prepare the Mujoco environment and install the D4RL datasets. 
To establish the Mujoco task environment, you may refer to the [guidelines](https://ivanvoid.github.io/voidlog.github.io/2022/05/27/d4rl_installation.html)
or just simply run:
```
pip install -r requirements.txt
```
To use the D4RL dataset, you need to setup the D4RL environment by 
```
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .
```


You can refer to [tutorial.ipynb](offline_tutorial.ipynb) for a detailed illustration and code demo.

Have fun!

