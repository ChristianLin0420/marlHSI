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
            
            # Get task information for dynamic naming
            # Try different ways to get the task name - prioritize args.task
            task_name = 'unknown'
            
            # Try to get task name from args first (most reliable)
            if 'args' in config and hasattr(config['args'], 'task'):
                task_name = config['args'].task
            elif hasattr(config.get('args', {}), 'task'):
                task_name = config.get('args', {}).task
            elif 'name' in config:
                task_name = config['name']
            elif 'params' in config and 'config' in config['params'] and 'name' in config['params']['config']:
                task_name = config['params']['config']['name']
            
            print(f"Detected task_name: {task_name}")
            
            # Generate dynamic project name if not specified
            project_name = wandb_config.get('project')
            if project_name is None:
                project_name = self._generate_project_name(task_name)
                print(f"Generated project_name: {project_name}")
            
            # Generate dynamic run name if not specified
            run_name = wandb_config.get('name')
            if run_name is None:
                run_name = self._generate_run_name(task_name, config)
                print(f"Generated run_name: {run_name}")
            
            # Generate dynamic group name if not specified
            group_name = wandb_config.get('group')
            if group_name is None:
                group_name = self._generate_group_name(task_name)
                print(f"Generated group_name: {group_name}")
            
            # Update tags with task-specific information
            tags = list(wandb_config.get('tags', []))
            task_tags = self._generate_task_tags(task_name)
            tags.extend(tag for tag in task_tags if tag not in tags)
            print(f"Final tags: {tags}")
            
            # Initialize wandb
            self.run = wandb.init(
                project=project_name,
                entity=wandb_config.get('entity', None),
                name=run_name,
                config=config,
                tags=tags,
                notes=wandb_config.get('notes', ''),
                mode=wandb_config.get('mode', 'online'),  # online, offline, or disabled
                group=group_name,
                job_type=wandb_config.get('job_type', 'train'),
                reinit=True,  # Always reinit to avoid conflicts
            )
            
            print(f"Wandb run initialized with name: {self.run.name}")
            
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
        
        # Don't define step metrics to avoid conflicts with existing logging
        # wandb.define_metric("frame", step_metric="frame")
        # wandb.define_metric("epoch", step_metric="epoch")
    
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
        
        # Log without explicit step to avoid conflicts
        try:
            if step is not None:
                wandb.log(processed_metrics, step=step, commit=commit)
            else:
                wandb.log(processed_metrics, commit=commit)
        except Exception as e:
            # If there's a step conflict, log without step
            print(f"Wandb step conflict, logging without step: {e}")
            wandb.log(processed_metrics, commit=commit)
    
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
            "training/frame": frame,
            "training/epoch": epoch,
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
        
        # Log without explicit step to avoid conflicts
        self.log_metrics(metrics, step=None)
    
    def log_evaluation_metrics(self, eval_results: Dict[str, Any], frame: int):
        """
        Log evaluation metrics.
        
        Args:
            eval_results: Evaluation results dictionary
            frame: Current frame number
        """
        if not self.enabled:
            return
        
        metrics = {"evaluation/frame": frame}
        
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
        
        # Log without explicit step to avoid conflicts
        self.log_metrics(metrics, step=None)
    
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
        
        video_data = {f"videos/{caption}": wandb.Video(video, fps=fps, format="mp4")}
        # Log without explicit step to avoid conflicts
        if step is not None:
            video_data["video/frame"] = step
        wandb.log(video_data)
    
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
        
        hist_data = {f"histograms/{name}": wandb.Histogram(values.cpu().numpy())}
        # Log without explicit step to avoid conflicts
        if step is not None:
            hist_data["histogram/frame"] = step
        wandb.log(hist_data)
    
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
    
    def _generate_project_name(self, task_name: str) -> str:
        """Generate project name based on task."""
        task_to_project = {
            'HumanoidTraj': 'tokenHSI-trajectory',
            'HumanoidSit': 'tokenHSI-sitting', 
            'HumanoidCarry': 'tokenHSI-carrying',
            'HumanoidClimb': 'tokenHSI-climbing',
            'HumanoidTrajSitCarryClimb': 'tokenHSI-multi-task',
            'HumanoidCompSitCarry': 'tokenHSI-skill-composition',
            'HumanoidCompClimbCarry': 'tokenHSI-skill-composition',
            'HumanoidCompTrajCarry': 'tokenHSI-skill-composition',
            'HumanoidAdaptCarryBox2Objs': 'tokenHSI-adaptation',
            'HumanoidAdaptTrajGround2Terrain': 'tokenHSI-adaptation',
            'HumanoidAdaptCarryGround2Terrain': 'tokenHSI-adaptation',
            'HumanoidLongTerm4BasicSkills': 'tokenHSI-longterm',
        }
        return task_to_project.get(task_name, f'tokenHSI-{task_name.lower()}')
    
    def _generate_run_name(self, task_name: str, config: Dict[str, Any]) -> str:
        """Generate run name based on algorithm and configuration."""
        
        # Get algorithm name as the primary component - try different ways to access it
        algo_name = 'basic'
        if 'params' in config and 'algo' in config['params'] and isinstance(config['params']['algo'], dict):
            algo_name = config['params']['algo'].get('name', 'unknown')
        elif isinstance(config.get('algo'), dict):
            algo_name = config.get('algo', {}).get('name', 'unknown')
        elif hasattr(config.get('algo'), 'name'):
            algo_name = getattr(config.get('algo'), 'name', 'unknown')
        elif 'algo' in config and isinstance(config['algo'], str):
            algo_name = config['algo']
        
        # Start with algorithm name as the base
        base_name = algo_name.lower()
        
        # Add network architecture details
        network_name = ''
        try:
            if 'params' in config and 'network' in config['params'] and isinstance(config['params']['network'], dict):
                network_name = config['params']['network'].get('name', '')
            elif isinstance(config.get('network'), dict):
                network_name = config.get('network', {}).get('name', '')
            elif hasattr(config.get('network'), 'name'):
                network_name = getattr(config.get('network'), 'name', '')
            elif 'network' in config and isinstance(config['network'], str):
                network_name = config['network']
        except (AttributeError, TypeError):
            # If network is an object without get method, skip it
            network_name = ''
        
        # Add architecture modifiers based on network name
        if 'transformer' in network_name.lower():
            base_name += '-transformer'
        
        # Add task type modifiers for multi-task or specialized tasks
        if 'multi_task' in network_name.lower():
            base_name += '-multi-task'
        elif 'comp' in network_name.lower():
            base_name += '-composition'
        elif 'adapt' in network_name.lower():
            base_name += '-adaptation'
        elif 'longterm' in network_name.lower():
            base_name += '-longterm'
        else:
            # For single task scenarios, add a simplified task identifier
            if task_name.lower() != 'humanoid':
                # Extract core task type from task name
                task_identifier = ''
                if 'traj' in task_name.lower():
                    task_identifier = 'traj'
                elif 'sit' in task_name.lower():
                    task_identifier = 'sit'
                elif 'carry' in task_name.lower():
                    task_identifier = 'carry'
                elif 'climb' in task_name.lower():
                    task_identifier = 'climb'
                
                if task_identifier:
                    base_name += f'-{task_identifier}'
        
        return base_name
    
    def _generate_group_name(self, task_name: str) -> str:
        """Generate group name based on task type."""
        if any(skill in task_name for skill in ['Traj', 'Sit', 'Carry', 'Climb']):
            if 'Comp' in task_name:
                return 'skill_composition'
            elif 'Adapt' in task_name:
                return 'adaptation'
            elif 'LongTerm' in task_name:
                return 'longterm'
            elif any(multi in task_name for multi in ['TrajSitCarryClimb']):
                return 'multi_task'
            else:
                return 'basic_skills'
        return 'other'
    
    def _generate_task_tags(self, task_name: str) -> List[str]:
        """Generate task-specific tags."""
        tags = []
        
        # Add task type tags
        if 'Traj' in task_name:
            tags.append('trajectory')
        if 'Sit' in task_name:
            tags.append('sitting')
        if 'Carry' in task_name:
            tags.append('carrying')
        if 'Climb' in task_name:
            tags.append('climbing')
        
        # Add method tags
        if 'Comp' in task_name:
            tags.append('composition')
        if 'Adapt' in task_name:
            tags.append('adaptation')
        if 'LongTerm' in task_name:
            tags.append('longterm')
        
        # Add skill complexity
        if any(multi in task_name for multi in ['TrajSitCarryClimb']):
            tags.append('multi_task')
        elif any(skill in task_name for skill in ['Traj', 'Sit', 'Carry', 'Climb']):
            tags.append('basic_skill')
        
        return tags 