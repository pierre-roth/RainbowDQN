import torch

class Config:
    # Environment
    ENV_ID = "BreakoutNoFrameskip-v4"
    REWARD_CLIPPING = False
    
    # Compute Device
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # N-Step Learning
    N_STEP = 3
    
    # Training Hyperparameters
    TOTAL_TIMESTEPS = 10_000_000
    LEARNING_RATE = 0.00005
    BUFFER_SIZE = 1_000_000      # Number of transitions to store (Optimized storage)
    BATCH_SIZE = 32
    GAMMA = 0.99                 # Discount factor
    TARGET_NETWORK_UPDATE_FREQ = 32000  # Update every 32000 steps (standard for Rainbow/DQN)
    
    # Priority Replay Buffer (PER)
    PER_ALPHA = 0.5
    PER_BETA_START = 0.4
    PER_BETA_FRAMES = 10_000_000 # Anneal beta over total steps
    
    # Noisy Networks (Exploration)
    # 0.5 is standard for NoisyNet-DQN
    STD_INIT = 0.5 
    
    # Training Loop
    LEARNING_STARTS = 80_000     # Steps before training starts
    TRAIN_FREQUENCY = 4          # Train every n steps
    
    # Logging
    LOG_INTERVAL = 1000          # Steps between logging
    SAVE_INTERVAL = 100_000      # Steps between saving model
    EVAL_INTERVAL = 50_000       # Steps between evaluation
    EVAL_EPISODES = 3            # Number of episodes to run for evaluation
