# Weights & Biases Integration for tokenHSI

This document explains how to use the Weights & Biases (wandb) integration for tracking training metrics in tokenHSI.

## Installation

Make sure wandb is installed:
```bash
pip install wandb>=0.15.0
```

## Configuration

### Basic Setup

To enable wandb logging, add the following to your training configuration file:

```yaml
use_wandb: True

wandb:
  project: "tokenHSI"
  entity: "your-wandb-username"  # Optional
  name: "experiment-name"         # Optional, defaults to config name
  tags: ["amp", "humanoid"]       # Optional tags
  notes: "Training notes"         # Optional notes
  mode: "online"                  # online, offline, or disabled
```

### Configuration Options

- **use_wandb**: Set to `True` to enable wandb logging
- **project**: The wandb project name
- **entity**: Your wandb username or team name (optional)
- **name**: Name for this specific run (optional)
- **tags**: List of tags to organize runs
- **notes**: Notes about the experiment
- **mode**: 
  - `"online"`: Sync data in real-time
  - `"offline"`: Save locally, sync later with `wandb sync`
  - `"disabled"`: Turn off wandb

## Tracked Metrics

The integration automatically tracks all training metrics:

### Performance Metrics
- `performance/total_fps`: Total frames per second
- `performance/step_fps`: Step frames per second
- `performance/update_time`: Update time per epoch
- `performance/play_time`: Play time per epoch

### Loss Metrics
- `losses/a_loss`: Actor loss
- `losses/c_loss`: Critic loss
- `losses/disc_loss`: Discriminator loss (AMP)
- `losses/bounds_loss`: Bounds loss
- `losses/entropy`: Entropy

### Training Information
- `info/epochs`: Current epoch
- `info/last_lr`: Learning rate
- `info/lr_mul`: Learning rate multiplier
- `info/clip_frac`: Clipping fraction
- `info/kl`: KL divergence

### AMP-Specific Metrics
- `info/disc_agent_acc`: Discriminator accuracy on agent samples
- `info/disc_demo_acc`: Discriminator accuracy on demonstrations
- `info/disc_agent_logit`: Agent logits
- `info/disc_demo_logit`: Demo logits
- `info/disc_grad_penalty`: Gradient penalty
- `info/disc_reward_mean`: Mean discriminator reward
- `info/disc_reward_std`: Std of discriminator rewards

### Rewards and Episode Lengths
- `rewards/mean`: Mean episode reward
- `rewards/{i}`: Reward for each value dimension
- `rewards/{task_name}`: Task-specific rewards (multi-task)
- `episode_lengths/mean`: Mean episode length

### Evaluation Metrics
- `eval/{exp_name}/success_rate`: Success rate
- `eval/{exp_name}/success_precision`: Success precision
- `eval/{exp_name}/num_trials`: Number of trials
- `eval/{exp_name}/success_trials`: Successful trials
- `eval/{exp_name}/fail_trials`: Failed trials

## Usage Examples

### Example 1: Basic Training with wandb

```bash
python run.py --task HumanoidTraj --cfg_env humanoid_traj.yaml --cfg_train amp_humanoid_traj.yaml
```

Make sure your `amp_humanoid_traj.yaml` includes:
```yaml
use_wandb: True
wandb:
  project: "tokenHSI-humanoid"
```

### Example 2: Offline Mode

For environments without internet access:

```yaml
use_wandb: True
wandb:
  project: "tokenHSI"
  mode: "offline"
```

Later sync the data:
```bash
wandb sync wandb/offline-run-*
```

### Example 3: Organizing Experiments

Use tags and groups to organize related experiments:

```yaml
use_wandb: True
wandb:
  project: "tokenHSI"
  group: "ablation-study"
  tags: ["ablation", "disc_coef", "v1.0"]
  notes: "Testing different discriminator coefficients"
```

## Model Artifacts

The integration automatically saves model checkpoints to wandb:
- Regular checkpoints (based on `save_freq`)
- Final model checkpoint

Models are saved with metadata including epoch and frame numbers.

## Viewing Results

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your project
3. View runs, compare metrics, and create reports

## Disabling wandb

To disable wandb without changing config files:
```bash
export WANDB_MODE=disabled
python run.py ...
```

Or in the config:
```yaml
use_wandb: False
```

## Troubleshooting

### Authentication
If not logged in, run:
```bash
wandb login
```

### Offline Sync Issues
If offline runs fail to sync:
```bash
wandb sync --sync-all
```

### Memory Issues
For large experiments, consider logging less frequently or using offline mode. 