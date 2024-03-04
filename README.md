## DSAA5009: Offline Reinforcement Learining project.

In this project, you need to train an agent to learn to do continuous control based on the Offline Reinforcement Learning standard dataset
[D4RL](https://github.com/digital-brain-sh/d4rl) dataset. You are required to trian your agent for the MUJOCO locamotion task [hopper](https://www.gymlibrary.dev/environments/mujoco/hopper/) And in the offline RL setting, you are required to use the offline dataset 
[hopper-medium-v2](https://github.com/Farama-Foundation/d4rl/wiki/Tasks) to train an agent to learn to hops that move in the forward (right) direction.

(insert hopper.gif)

Unlike online reinforcement learning tasks, offline reinforcement learning tasks are challenging because the agent you train relies solely on the available data and is unable to interact with the environment to obtain performance evaluations to modify its actions.So, when training an agent, you can use familiar online reinforcement learning algorithms, but be mindful of the steps you need to take when the agent takes actions for which the offline dataset does not provide feedback. This is because in the offline reinforcement learning task, the **unknown risk action** taken by the agent will be easily recognised as a spurious high-yield action, leading to the accumulation of errors and ultimately to the failure of the intelligence. In the hopper task, this failure means that the one-legg will fall down and thus not be able to get up again.

Before begin your work, you need to first prepare the MUJOCO environment and install the D4RL datasets. For setting up the MUJOCO task environment, you can refer to the following [tutorial](https://ivanvoid.github.io/voidlog.github.io/2022/05/27/d4rl_installation.html) or just simply run:
```
pip install -r requirements.txt
```
To use the D4RL dataset, you need to setup the D4RL environment by 
```
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .
```
If there are any missing packages, you can check ```requirements.txt``` or 
simply run
```
pip install -r requirements.txt
```

You can refer to [tutorial.ipynb](tutorial.ipynb) for a detailed illustration and code demo.

