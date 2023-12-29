import time
import gym
import wandb
import numpy as np
from stable_baselines3 import SAC
import os
import sys
from pathlib import Path
import random
from stable_baselines3.common.vec_env import DummyVecEnv
import models.denseMlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from trackingEnv import TrackingEnv


def make_env(
             optimal_distance: float = 0.75,
             ts: float = 0.05,
             mass_noise: bool = True,
             output_delay: bool = True,
             sensor_noise: bool = True,
             start_noise: bool = True,
             move_target: bool = True,
             asymmetric_actor_buffer_length: int = 15):

    def _init() -> gym.Env:
        env = TrackingEnv(
            optimal_distance=optimal_distance,
            Ts=ts,
            sensor_noise=sensor_noise,
            mass_noise=mass_noise,
            start_noise=start_noise,
            output_delay=output_delay,
            move_target=move_target,
            asymmetric_actor_buffer_length=asymmetric_actor_buffer_length
        )
        env.seed(random.randint(1, 10000))
        return env

    return _init



if __name__ == "__main__":

    settings = {'num_cpu': 8,
                'policy': 'DenseMlpPolicy',
                'n_timesteps': 10000,
                'WandB': False,
                'WandB_project': '<WandB_project>',
                'WandB_entity': '<WandB_entity>',
                'WandB_API_key': '<WandB_API_key>',
                'render': False,
                'eval_episodes': 5,
                'eval_mode': False,
                'optimal_distance': 0.75,
                'Ts': 0.05,
                'sensor_noise': True,
                'mass_noise': True,
                'start_noise': True,
                'output_delay': True,
                'move_target': True,
                'asymmetric_actor_buffer_length': 15
                }

    if settings['eval_mode']:
        eval_env = DummyVecEnv([make_env(optimal_distance=settings['optimal_distance'],
                                         ts=settings['Ts'],
                                         sensor_noise=settings['sensor_noise'],
                                         mass_noise=settings['mass_noise'],
                                         start_noise=settings['start_noise'],
                                         output_delay=settings['output_delay'],
                                         move_target=settings['move_target'],
                                         asymmetric_actor_buffer_length=settings['asymmetric_actor_buffer_length'])])

        # MODEL TO TEST
        model_ID = 1672672721
        model_NUMBER = 0
        model = SAC.load(os.path.join("experiments/SAC_{}".format(model_ID), "SAC_{}".format(model_NUMBER)), env=eval_env)

        # Eval
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=settings['eval_episodes'],
                                                  render=settings['render'])
        print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

    else:
        t = int(time.time())

        # Path for Models
        pathname = os.path.dirname(sys.argv[0])
        abs_path = os.path.abspath(pathname)
        current_path = Path(os.path.join(abs_path, "experiments", "SAC_{}".format(t)))
        current_path.mkdir(parents=True, exist_ok=True)

        if settings['WandB']:
            wandb.login(key=settings['WandB_API_key'])
            wandb.init(project=settings['WandB_project'], entity=settings['WandB_entity'],
                       name="SAC_{}".format(t), config=settings)

        # Training VecEnv
        vec_env = DummyVecEnv([make_env(optimal_distance=settings['optimal_distance'],
                                        ts=settings['Ts'],
                                        sensor_noise=settings['sensor_noise'],
                                        mass_noise=settings['mass_noise'],
                                        start_noise=settings['start_noise'],
                                        output_delay=settings['output_delay'],
                                        move_target=settings['move_target'],
                                        asymmetric_actor_buffer_length=settings['asymmetric_actor_buffer_length']) for i in range(settings['num_cpu'])])

        # Create Model for Training
        model = SAC(settings['policy'], vec_env, verbose=1)

        # We create a separate environment for evaluation
        eval_env = DummyVecEnv([make_env(optimal_distance=settings['optimal_distance'],
                                         ts=settings['Ts'],
                                         sensor_noise=settings['sensor_noise'],
                                         mass_noise=settings['mass_noise'],
                                         start_noise=settings['start_noise'],
                                         output_delay=settings['output_delay'],
                                         move_target=settings['move_target'],
                                         asymmetric_actor_buffer_length=settings['asymmetric_actor_buffer_length'])])

        # Save Best Models
        best_episodes = np.full((10,), -100.0)

        # RL Training
        while True:
            model.learn(settings['n_timesteps'])

            # Eval
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=settings['eval_episodes'], render=settings['render'])
            print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

            if settings['WandB']:
                wandb.log({'test': mean_reward})

            worst_model = np.argmin(best_episodes)
            if mean_reward > best_episodes[worst_model]:
                best_episodes[worst_model] = mean_reward
                model.save(os.path.join(current_path, "SAC_{}".format(worst_model)))
                np.savetxt(os.path.join(current_path, "models_score.csv"), best_episodes, delimiter=",")

