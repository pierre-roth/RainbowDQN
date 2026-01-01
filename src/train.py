import torch
import random
import numpy as np
import datetime
from pathlib import Path
from tqdm import tqdm
from .config import Config
from .wrappers import make_env
from .agent import DQNAgent
from .buffer import PrioritizedReplayBuffer
from .utils import MetricLogger

def evaluate(agent: DQNAgent, step: int, logger: MetricLogger) -> None:
    """Run evaluation episodes."""
    # Use episodic_life=False to ensure natural game flow (full episodes)
    # and clip_rewards=False to measure real game score
    eval_env = make_env(Config.ENV_ID, render_mode=None, episodic_life=False, clip_rewards=False)
    rewards = []
    
    try:
        for _ in range(Config.EVAL_EPISODES):
            obs, info = eval_env.reset()
            done = False
            total_reward = 0
            
            # Track lives to handle dead-lock when waiting for fire
            current_lives = info.get("lives", 5)
            
            while not done:
                # Check if we lost a life and need to fire to resume
                new_lives = info.get("lives", current_lives)
                
                if new_lives < current_lives:
                    action = 1 # FIRE
                    current_lives = new_lives
                else:
                    action = agent.select_action(obs, training=False)
                
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                # Full game over logic
                if terminated or truncated:
                    # In no-episodic-life mode, terminated usually means GAME OVER (lives=0)
                    # But just to be safe and consistent with visualize:
                    if info.get("lives", 0) == 0:
                        done = True
                    else:
                        # Should not theoretically happen if terminated=True means game over in raw env,
                        # but if it does, reset to continue
                        obs, info = eval_env.reset()
                        current_lives = info.get("lives", 5)
                
                total_reward += reward
            rewards.append(total_reward)
            
        avg_reward = sum(rewards) / len(rewards)
        print(f"\nEvaluation at step {step}: Avg Reward {avg_reward:.2f}")
        logger.log_eval(step, avg_reward)
        
    finally:
        eval_env.close()

def train(resume_path: str = None, seed: int = 42) -> None:
    
    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize Environment
    # headless training (render_mode=None) for speed
    # clip_rewards=False to learn to maximize game score (distinguish brick values)
    # UNLESS explicitly enabled in Config (e.g. for Rainbow setup alignment)
    env = make_env(Config.ENV_ID, render_mode=None, episodic_life=True, clip_rewards=Config.REWARD_CLIPPING)
    
    # Init seed for env
    env.reset(seed=seed)
    env.action_space.seed(seed)
    
    # Initialize Agent
    agent = DQNAgent(
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n
    )
    
    # Setup directories
    start_step = 0
    if resume_path:
        resume_path = Path(resume_path)
        print(f"Resuming from checkpoint: {resume_path}")
        start_step = agent.load(resume_path)
        print(f"Resuming training from step {start_step}")
        
        # Reuse the directory of the checkpoint
        save_dir = resume_path.parent
    else:
        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Buffer (PER)
    buffer = PrioritizedReplayBuffer(
        capacity=Config.BUFFER_SIZE,
        observation_shape=env.observation_space.shape
    )
    
    # Load buffer if resuming
    if resume_path:
        buffer_path = resume_path.with_suffix(".buffer.npz")
        # Try generic buffer path if specific one fails
        if not buffer_path.exists():
             buffer_path = save_dir / "buffer.npz"
             
        if buffer_path.exists():
            print(f"Loading replay buffer from {buffer_path}...")
            buffer.load(buffer_path)
        else:
            print(f"Warning: No buffer file found at {buffer_path}. Starting with empty buffer.")
    
    # Logger
    logger = MetricLogger(save_dir, resume=(resume_path is not None))
    
    # Training Loop
    print(f"Starting training on {Config.DEVICE}...")
    
    obs, info = env.reset(seed=seed)
    
    episode = 0
    if resume_path:
        episode = logger.get_last_episode()
    
    pbar = tqdm(total=Config.TOTAL_TIMESTEPS, initial=start_step, desc="Training")
    
    try:
        for step in range(start_step, Config.TOTAL_TIMESTEPS):
            
            # Select Action
            action = agent.select_action(obs)
            
            # Step Env
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Add to Buffer
            buffer.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
            
            # Logging step metrics
            loss, q = None, None
            
            # Learn
            if step > Config.LEARNING_STARTS and step % Config.TRAIN_FREQUENCY == 0:
                # Buffer guard - only skip learning, not the whole loop
                if len(buffer) >= Config.BATCH_SIZE:
                    beta = min(1.0, Config.PER_BETA_START + step * (1.0 - Config.PER_BETA_START) / Config.PER_BETA_FRAMES)
                    batch, weights, tree_indices = buffer.sample(Config.BATCH_SIZE, beta)
                    loss, q, td_errors = agent.learn(batch, weights)
                    buffer.update_priorities(tree_indices, td_errors)
                
            # Update Target Network
            # With Soft Updates (TAU < 1.0), we run this every step (FREQ=1)
            # If TAU == 1.0 (Hard), we run every FREQ steps.
            if step % Config.TARGET_NETWORK_UPDATE_FREQ == 0:
                 agent.sync_target_network()
            
            # Log step info
            if loss is not None:
                logger.log_step(reward, loss, q)
            else:
                logger.log_step(reward, None, None)
                
            # Handle Episode End
            if done:
                # Only log full episode (all lives lost)
                # 'lives' in info is populated by EpisodicLifeEnv
                lives = info.get("lives", 0)
                if lives == 0:
                    logger.log_episode()
                    episode += 1
                
                obs, info = env.reset()
                
                # Log to console periodically
                if lives == 0 and episode % 10 == 0:
                    metrics = {
                        "ep": episode,
                        "rew": f"{logger.ep_rewards[-1]:.1f}" if logger.ep_rewards else "0.0"
                    }
                    pbar.set_postfix(metrics)
                    logger.record(episode, step) 
                    
            # Evaluation
            if step > Config.LEARNING_STARTS and step % Config.EVAL_INTERVAL == 0:
                 evaluate(agent, step, logger)
                    
            # Save Model
            if step > 0 and step % Config.SAVE_INTERVAL == 0:
                save_path = save_dir / f"dqn_breakout_{step}.chkpt"
                agent.save(save_path, step)
                pbar.write(f"Saved checkpoint to {save_path}")

            pbar.update(1)
            
    except KeyboardInterrupt:
        pbar.close()
        print("\nTraining interrupted! Saving emergency checkpoint...")
        save_path = save_dir / "dqn_breakout_interrupted.chkpt"
        agent.save(save_path, step)
        print(f"Saved emergency checkpoint to {save_path}")
        
        # Save emergency buffer
        buffer_save_path = save_path.with_suffix(".buffer.npz")
        print("Saving emergency buffer (this may take a while)...")
        buffer.save(buffer_save_path)
        print(f"Saved emergency buffer to {buffer_save_path}")
        
    pbar.close()
    print("Training Complete or Interrupted!")
    env.close()

def visualize(checkpoint_path: str = None) -> None:
    # Use episodic_life=False for visualization (play full game naturally?) 
    # Actually, previous training=False implied episodic_life=False.
    env = make_env(Config.ENV_ID, render_mode="human", episodic_life=False, clip_rewards=False)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    
    if checkpoint_path is None:
        # Try to find the latest checkpoint
        checkpoints_root = Path("checkpoints")
        if checkpoints_root.exists():
            # Find all .chkpt files recursively
            all_checkpoints = list(checkpoints_root.glob("**/*.chkpt"))
            if all_checkpoints:
                # Sort by modification time (latest first) or by step count in filename
                # Filename format: dqn_breakout_<step>.chkpt
                # Let's try to extract step count if possible, else mtime
                def get_step(p):
                    try:
                        return int(p.stem.split('_')[-1])
                    except ValueError:
                        return 0
                
                # Sort by step count descending, then mtime
                latest_checkpoint = max(all_checkpoints, key=lambda p: (get_step(p), p.stat().st_mtime))
                checkpoint_path = latest_checkpoint
                print(f"No checkpoint provided. Auto-found latest: {checkpoint_path}")
            else:
                print("No checkpoints found in 'checkpoints/'. Using random agent.")
        else:
            print("'checkpoints/' directory not found. Using random agent.")
    else:
        print(f"Loading checkpoint: {checkpoint_path}")

    if checkpoint_path:
        agent.load(checkpoint_path)
    
    obs, info = env.reset()
    current_lives = info.get("lives", 5)
    total_reward = 0
    try:
        while True:
            # Render is handled by make_env with render_mode='human'
            
            # Check for life loss to auto-fire
            new_lives = info.get("lives", current_lives)
            if new_lives < current_lives:
                action = 1 # FIRE
                current_lives = new_lives
            else:
                action = agent.select_action(obs, training=False)
                
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            obs = next_obs
            
            if terminated or truncated:
                print(f"Game Over! Score: {total_reward}")
                total_reward = 0
                obs, info = env.reset()
                current_lives = info.get("lives", 5)
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        env.close()
