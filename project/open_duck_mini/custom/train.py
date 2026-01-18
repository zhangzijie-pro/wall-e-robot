import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from duck_rl_env import DuckRLEnv, DuckRLEnvCfg  # 您的env

def make_env():
    def _init():
        env = DuckRLEnv(DuckRLEnvCfg())
        return env
    return _init

if __name__ == "__main__":
    num_envs = 4  # 子进程并行（根据GPU调整）
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    model = PPO(
        "MlpPolicy",  # 对于向量观测；用"CnnPolicy"如果有图像
        env,
        verbose=1,
        tensorboard_log="./duck_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
    )
    
    # 评估回调
    eval_callback = EvalCallback(env, best_model_save_path="./logs/", log_path="./logs/", eval_freq=10000)
    
    model.learn(total_timesteps=1_000_000, callback=eval_callback)  # 训练1M步
    model.save("duck_ppo_model")
    env.close()