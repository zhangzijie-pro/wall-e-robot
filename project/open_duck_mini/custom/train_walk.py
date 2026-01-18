import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from duck_walk_env import DuckWalkEnv, DuckWalkEnvCfg

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="flat_terrain", help="Task: flat_terrain or rough_terrain_backlash")
parser.add_argument("--num_timesteps", type=int, default=300000000, help="Total timesteps")
args = parser.parse_args()

def make_env():
    cfg = DuckWalkEnvCfg()
    cfg.task_name = args.task  # 动态设置任务（影响地形）
    return lambda: DuckWalkEnv(cfg)

if __name__ == "__main__":
    num_envs = 8  # 根据GPU调整
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=3e-4, batch_size=64, n_steps=2048,  # 超参数：从原仓库推断，调整以稳定
        tensorboard_log="./duck_logs/",
    )
    
    eval_callback = EvalCallback(env, eval_freq=10000, log_path="./duck_logs/")
    model.learn(total_timesteps=args.num_timesteps, callback=eval_callback)
    model.save("duck_walk_ppo")
    env.close()