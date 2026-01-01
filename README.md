# Rainbow DQN - Breakout

This repository contains a clean, research-friendly implementation of **Rainbow DQN** (minus Categorical DQN) for the Atari Breakout environment using PyTorch and Gymnasium.

## Features implemented
- **Double DQN**: Reduces overestimation bias.
- **Dueling Networks**: Separate value and advantage streams.
- **Prioritized Experience Replay (PER)**: Learns more from important transitions.
- **Multi-step Learning (N-Step)**: Faster reward propagation.
- **Noisy Networks**: Learnable exploration (replaces epsilon-greedy).

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training
Start training the agent. Checkpoints are saved to `checkpoints/`.
```bash
python main.py train
```
Resume from a specific checkpoint:
```bash
python main.py train --resume checkpoints/2023-10-27T10-00-00/dqn_breakout_100000.chkpt
```
**Note:** The replay buffer is only saved on `KeyboardInterrupt` (Ctrl+C) or if explicitly requested, to save time during regular checkpoints.

### Visualization
Watch the agent play:
```bash
python main.py visualize
```
Or specify a checkpoint:
```bash
python main.py visualize checkpoints/2023-10-27T10-00-00/dqn_breakout_1000000.chkpt
```

### Plotting
Plot training metrics (Rewards, Loss, Q-values):
```bash
python main.py plot
```
Plots are saved as `training_plots.png` in the log directory.

## Project Structure
- `src/model.py`: DQN Architecture (Dueling, Noisy).
- `src/agent.py`: Agent logic (action selection, learning).
- `src/buffer.py`: Prioritized Replay Buffer with N-step returns.
- `src/wrappers.py`: Atari wrappers (NoopReset, FrameStack, etc.).
- `src/config.py`: Hyperparameters.
- `src/train.py`: Training loop.
