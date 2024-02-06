# Exploring Deep Reinforcement Learning for Robust Target Tracking using Micro Aerial Vehicles

This repository is the official implementation of [Exploring Deep Reinforcement Learning for Robust Target Tracking using Micro Aerial Vehicles](https://ieeexplore.ieee.org/abstract/document/10407017). 

---

## Abstract

<div style="text-align: justify"> 
The capability to autonomously track a non-cooperative target is a key technological 
requirement for micro aerial vehicles.
In this paper, we propose an output feedback control scheme based on deep reinforcement
learning for controlling a micro 
aerial vehicle to persistently track a flying target while maintaining visual contact.
The proposed method leverages relative position data for control, relaxing
the assumption of having access to full state information which is typical of related
approaches in literature.
Moreover, we exploit classical robustness indicators in the learning process through
domain randomization to increase the robustness of the learned policy. 
Experimental results validate the proposed approach for target tracking,
demonstrating high performance and robustness with respect to mass
mismatches and control delays. The resulting nonlinear controller
significantly outperforms a standard model-based design in numerous off-nominal
scenarios.
</div>

<figure>
  <figcaption align = "center"><b>The target tracking task. The tracker (blue) follows the target (red) while maintaining attitude alignment.</b></figcaption>
  <img
  src="images/overview_ok_omega.png"
  alt="Actor Net">
</figure>

## Configuration

The code has been tested with Python 3.8 in a Windows host machine. You can install the requirements with:

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install stable-baselines3==1.5.0
```

**NB**: torch must be installed depending on your pc configuration (CPU or GPU).


---

## Usage

Inside `train.py` there is a list of training parameters that can be configured:
```
'num_cpu': 8,
'policy': 'DenseMlpPolicy',
'n_timesteps': 10000,
'WandB': False,
'WandB_project': '<your_wandb_project>',
'WandB_entity': '<your_wandb_entity>',
'WandB_API_key': '<your_wandb_api_key>',
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
```
where:
- **num_cpu** is the number of parallel training environments (Suggested: PC cores number - 2).
- **policy** stable-baseline policy. DenseMlpPolicy is out asymmetric implementation of SAC.
- **n_timesteps** Timesteps of training before a validation test.
- **WandB** boolean if you want to use Wandb to monitor the training.
- **WandB_project - WandB_entity - WandB_API_key** Wandb configuration parameters.
- **render** enable rendering of the experiments. (Use only for test)
- **eval_episodes** number of episodes for the validation phase.
- **eval_mode** enables evaluation mode (Leave it false if you want to start a new training).
- **optimal_distance** Optimal distance from the target.
- **Ts** Time step of the RL training and the dynamic model.
- **sensor_noise - mass_noise - start_noise - output_delay** flags to enable the randomization and noise.
- **move_target** flag to enable the target movements.
- **asymmetric_actor_buffer_length** length of the Actor observation buffer.

Once configured the hyperparameters, just run `train.py` to start the training.

## Citing

If you use this code in a scientific context, please cite the following:

> 

BibTeX details:

```bibtex
@INPROCEEDINGS{10407017,
  author={Dionigi, Alberto and Leomanni, Mirko and Saviolo, Alessandro and Loianno, Giuseppe and Costante, Gabriele},
  booktitle={2023 21st International Conference on Advanced Robotics (ICAR)}, 
  title={Exploring Deep Reinforcement Learning for Robust Target Tracking Using Micro Aerial Vehicles}, 
  year={2023},
  volume={},
  number={},
  pages={506-513},
  keywords={Deep learning;Visualization;Target tracking;Reinforcement learning;Robustness;Delays;Output feedback},
  doi={10.1109/ICAR58858.2023.10407017}
}
```
