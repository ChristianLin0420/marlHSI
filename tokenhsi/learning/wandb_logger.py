import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional, List


class WandbLogger:
    """Weights & Biases logger for tokenHSI training metrics."""
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Initialize WandB logger.
        
        Args:
            config: Training configuration dictionary
            enabled: Whether to enable wandb logging
        """
        self.enabled = enabled and config.get('use_wandb', False)
        self.config = config
        
        if self.enabled:
            # Extract wandb specific config
            wandb_config = config.get('wandb', {})

            print(f"wandb_config: {wandb_config}")
            
            # Initialize wandb
            self.run = wandb.init(
                project=wandb_config.get('project', 'tokenHSI'),
                entity=wandb_config.get('entity', None),
                name=wandb_config.get('name', config.get('name', 'unnamed_run')),
                config=config,
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
                mode=wandb_config.get('mode', 'online'),  # online, offline, or disabled
                group=wandb_config.get('group', None),
                job_type=wandb_config.get('job_type', 'train'),
                reinit=wandb_config.get('reinit', True),
            )
            
            # Define custom metrics for better visualization
            self._define_metrics()
    
    def _define_metrics(self):
        """Define custom metrics and their relationships."""
        # Performance metrics
        wandb.define_metric("performance/total_fps", summary="mean")
        wandb.define_metric("performance/step_fps", summary="mean")
        wandb.define_metric("performance/update_time", summary="mean")
        wandb.define_metric("performance/play_time", summary="mean")
        
        # Loss metrics
        wandb.define_metric("losses/a_loss", summary="min")
        wandb.define_metric("losses/c_loss", summary="min")
        wandb.define_metric("losses/disc_loss", summary="min")
        wandb.define_metric("losses/bounds_loss", summary="min")
        wandb.define_metric("losses/entropy", summary="mean")
        
        # Reward metrics
        wandb.define_metric("rewards/mean", summary="max")
        wandb.define_metric("episode_lengths/mean", summary="mean")
        
        # Success metrics
        wandb.define_metric("success/rate", summary="max")
        wandb.define_metric("success/precision", summary="max")
        
        # Set step metric
        wandb.define_metric("frame", step_metric="frame")
        wandb.define_metric("epoch", step_metric="epoch")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, 
                   commit: bool = True):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (frame or epoch)
            commit: Whether to commit the log immediately
        """
        if not self.enabled:
            return
        
        # Flatten nested dictionaries
        flat_metrics = self._flatten_dict(metrics)
        
        # Convert tensors to scalars
        processed_metrics = {}
        for key, value in flat_metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    processed_metrics[key] = value.item()
                else:
                    # Log tensor statistics
                    processed_metrics[f"{key}/mean"] = value.mean().item()
                    processed_metrics[f"{key}/std"] = value.std().item()
                    processed_metrics[f"{key}/min"] = value.min().item()
                    processed_metrics[f"{key}/max"] = value.max().item()
            elif isinstance(value, (int, float, np.number)):
                processed_metrics[key] = value
            elif isinstance(value, np.ndarray) and value.size == 1:
                processed_metrics[key] = value.item()
        
        wandb.log(processed_metrics, step=step, commit=commit)
    
    def log_training_metrics(self, train_info: Dict[str, Any], frame: int, 
                           epoch: int, mean_rewards: Optional[torch.Tensor] = None,
                           mean_lengths: Optional[float] = None,
                           multi_task_rewards: Optional[Dict[str, torch.Tensor]] = None):
        """
        Log comprehensive training metrics.
        
        Args:
            train_info: Training information dictionary
            frame: Current frame number
            epoch: Current epoch number
            mean_rewards: Mean episode rewards
            mean_lengths: Mean episode lengths
            multi_task_rewards: Multi-task specific rewards
        """
        if not self.enabled:
            return
        
        metrics = {
            "frame": frame,
            "epoch": epoch,
        }
        
        # Performance metrics
        if 'total_time' in train_info:
            metrics["performance/total_time"] = train_info['total_time']
        if 'play_time' in train_info:
            metrics["performance/play_time"] = train_info['play_time']
        if 'update_time' in train_info:
            metrics["performance/update_time"] = train_info['update_time']
        
        # Loss metrics
        loss_keys = ['actor_loss', 'critic_loss', 'disc_loss', 'b_loss', 'entropy']
        for key in loss_keys:
            if key in train_info:
                value = train_info[key]
                if isinstance(value, list):
                    metrics[f"losses/{key}"] = torch.mean(torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in value])).item()
                elif isinstance(value, torch.Tensor):
                    metrics[f"losses/{key}"] = value.mean().item()
                else:
                    metrics[f"losses/{key}"] = value
        
        # Info metrics
        info_keys = ['kl', 'last_lr', 'lr_mul', 'actor_clip_frac', 
                     'disc_agent_acc', 'disc_demo_acc', 'disc_agent_logit', 
                     'disc_demo_logit', 'disc_grad_penalty', 'disc_logit_loss']
        for key in info_keys:
            if key in train_info:
                value = train_info[key]
                if isinstance(value, list):
                    metrics[f"info/{key}"] = torch.mean(torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in value])).item()
                elif isinstance(value, torch.Tensor):
                    metrics[f"info/{key}"] = value.mean().item()
                else:
                    metrics[f"info/{key}"] = value
        
        # Discriminator rewards
        if 'disc_rewards' in train_info:
            disc_rewards = train_info['disc_rewards']
            if isinstance(disc_rewards, torch.Tensor):
                metrics["info/disc_reward_mean"] = disc_rewards.mean().item()
                metrics["info/disc_reward_std"] = disc_rewards.std().item()
        
        # Episode rewards and lengths
        if mean_rewards is not None:
            if isinstance(mean_rewards, torch.Tensor):
                for i in range(mean_rewards.shape[0]):
                    metrics[f"rewards/{i}"] = mean_rewards[i].item()
                metrics["rewards/mean"] = mean_rewards.mean().item()
            else:
                metrics["rewards/mean"] = mean_rewards
        
        if mean_lengths is not None:
            metrics["episode_lengths/mean"] = mean_lengths
        
        # Multi-task rewards
        if multi_task_rewards is not None:
            for task_name, task_rewards in multi_task_rewards.items():
                if isinstance(task_rewards, torch.Tensor) and task_rewards.numel() > 0:
                    metrics[f"rewards/{task_name}"] = task_rewards.mean().item()
        
        self.log_metrics(metrics, step=frame)
    
    def log_evaluation_metrics(self, eval_results: Dict[str, Any], frame: int):
        """
        Log evaluation metrics.
        
        Args:
            eval_results: Evaluation results dictionary
            frame: Current frame number
        """
        if not self.enabled:
            return
        
        metrics = {"frame": frame}
        
        for exp_name, results in eval_results.items():
            prefix = f"eval/{exp_name}"
            
            if 'success_rate' in results:
                metrics[f"{prefix}/success_rate"] = results['success_rate']
            if 'success_precision' in results:
                metrics[f"{prefix}/success_precision"] = results['success_precision']
            if 'num_trials' in results:
                metrics[f"{prefix}/num_trials"] = results['num_trials']
            if 'success_trials' in results:
                metrics[f"{prefix}/success_trials"] = results['success_trials']
            if 'fail_trials' in results:
                metrics[f"{prefix}/fail_trials"] = results['fail_trials']
            if 'fail_trials_because_terminate' in results:
                metrics[f"{prefix}/fail_terminate"] = results['fail_trials_because_terminate']
        
        self.log_metrics(metrics, step=frame)
    
    def log_video(self, video: np.ndarray, caption: str = "episode", 
                  fps: int = 30, step: Optional[int] = None):
        """
        Log video to wandb.
        
        Args:
            video: Video array of shape (T, H, W, C)
            caption: Caption for the video
            fps: Frames per second
            step: Step number
        """
        if not self.enabled:
            return
        
        wandb.log({
            f"videos/{caption}": wandb.Video(video, fps=fps, format="mp4")
        }, step=step)
    
    def log_histogram(self, values: torch.Tensor, name: str, 
                     step: Optional[int] = None):
        """
        Log histogram to wandb.
        
        Args:
            values: Tensor of values
            name: Name of the histogram
            step: Step number
        """
        if not self.enabled:
            return
        
        wandb.log({
            f"histograms/{name}": wandb.Histogram(values.cpu().numpy())
        }, step=step)
    
    def save_model(self, model_path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save model artifact to wandb.
        
        Args:
            model_path: Path to the model file
            metadata: Optional metadata for the model
        """
        if not self.enabled:
            return
        
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            metadata=metadata or {}
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish the wandb run."""
        if self.enabled and hasattr(self, 'run'):
            self.run.finish()
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', 
                     sep: str = '/') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items) 