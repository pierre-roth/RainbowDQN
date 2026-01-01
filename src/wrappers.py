import gymnasium as gym
import ale_py
import numpy as np

class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    """
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            return self.reset(**kwargs)

        return obs, info

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # We lost a life, so mark as terminated for the agent to boost learning
            terminated = True
        self.lives = lives
        
        # Ensure lives are in info for logging
        info['lives'] = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from lost life state
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                 obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        info['lives'] = self.lives
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame (frameskipping)
    and return the max between the two last frames (to deal with flickering).
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Update buffer with latest frames
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            
            total_reward += reward
            if terminated or truncated:
                # If we terminate early, populate buffer with the final observation
                if i < self._skip - 2:
                     self._obs_buffer[0] = obs
                     self._obs_buffer[1] = obs
                elif i < self._skip - 1:
                     self._obs_buffer[1] = obs
                break
        
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer = np.zeros((2,)+self.env.observation_space.shape, dtype=np.uint8)
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs, info

class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)

def make_env(env_id, render_mode=None, episodic_life=True, clip_rewards=True):
    env = gym.make(env_id, render_mode=render_mode)
    
    # Standard Atari Wrappers
    # Note: NoopResetEnv removed as using FireResetEnv which is more robust for Breakout
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
        
    # Resize and Grayscale
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    
    # Frame Stacking
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env
