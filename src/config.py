"""
Configuration management utilities for emotion recognition project.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    type: str
    num_classes: int = 6
    dropout_rate: float = 0.3
    max_length: int = 128
    model_name: Optional[str] = None

@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1

@dataclass
class DataConfig:
    """Data configuration dataclass."""
    max_length: int = 128
    text_column: str = "text"
    label_column: str = "label"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_sizes: Dict[str, int] = field(default_factory=lambda: {
        'train': 32, 'validation': 64, 'test': 64
    })

@dataclass
class PreprocessingConfig:
    """Preprocessing configuration dataclass."""
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = False
    remove_extra_whitespace: bool = True
    remove_stopwords: bool = False
    expand_contractions: bool = True
    emoji_handling: str = "convert"  # remove, convert, keep
    add_special_tokens: bool = True

@dataclass
class EvaluationConfig:
    """Evaluation configuration dataclass."""
    metrics: list = field(default_factory=lambda: [
        "accuracy", "f1_macro", "f1_weighted", 
        "precision_macro", "recall_macro", "roc_auc"
    ])
    plot_confusion_matrix: bool = True
    plot_training_history: bool = True
    plot_embeddings: bool = True
    save_predictions: bool = True

@dataclass
class PathsConfig:
    """Paths configuration dataclass."""
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "outputs"
    log_dir: str = "logs"
    cache_dir: str = "cache"

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
    emotions: list = field(default_factory=lambda: [
        "sadness", "joy", "love", "anger", "fear", "surprise"
    ])
    device: Dict[str, Any] = field(default_factory=lambda: {
        "auto_detect": True, "force_cpu": False
    })
    logging: Dict[str, str] = field(default_factory=lambda: {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    })

class ConfigManager:
    """Manages loading and validating configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_cache = {}
    
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration as OmegaConf DictConfig
        """
        config_path = Path(config_path)
        
        # Check cache first
        if str(config_path) in self.config_cache:
            logger.debug(f"Loading config from cache: {config_path}")
            return self.config_cache[str(config_path)]
        
        # Load from file
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert to OmegaConf for better handling
            config = OmegaConf.create(config_dict)
            
            # Cache the config
            self.config_cache[str(config_path)] = config
            
            logger.info(f"Loaded configuration: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise
    
    def load_model_config(self, model_type: str) -> DictConfig:
        """
        Load configuration for specific model type.
        
        Args:
            model_type: Type of model (distilbert, twitter_roberta, bilstm, ensemble)
            
        Returns:
            Model configuration
        """
        config_file = f"{model_type}_config.yaml"
        config_path = self.config_dir / config_file
        
        return self.load_config(config_path)
    
    def merge_configs(self, base_config: DictConfig, 
                     override_config: Union[DictConfig, Dict]) -> DictConfig:
        """
        Merge two configurations with override taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        if isinstance(override_config, dict):
            override_config = OmegaConf.create(override_config)
        
        merged = OmegaConf.merge(base_config, override_config)
        return merged
    
    def validate_config(self, config: DictConfig) -> bool:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        required_sections = ['model', 'training', 'data']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model config
        if 'type' not in config.model:
            raise ValueError("Model type not specified")
        
        if 'num_classes' not in config.model:
            raise ValueError("Number of classes not specified")
        
        # Validate training config
        if config.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if config.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if config.training.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        # Validate data splits
        total_split = config.data.train_split + config.data.val_split + config.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError("Data splits must sum to 1.0")
        
        logger.info("Configuration validation passed")
        return True
    
    def create_experiment_config(self, config: DictConfig) -> ExperimentConfig:
        """
        Create structured experiment config from dictionary config.
        
        Args:
            config: Dictionary configuration
            
        Returns:
            Structured experiment configuration
        """
        # Create structured config objects
        model_config = ModelConfig(**config.model)
        training_config = TrainingConfig(**config.training)
        data_config = DataConfig(**config.data)
        preprocessing_config = PreprocessingConfig(**config.preprocessing)
        evaluation_config = EvaluationConfig(**config.evaluation)
        paths_config = PathsConfig(**config.paths)
        
        experiment_config = ExperimentConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            preprocessing=preprocessing_config,
            evaluation=evaluation_config,
            paths=paths_config,
            emotions=config.emotions,
            device=config.device,
            logging=config.logging
        )
        
        return experiment_config
    
    def save_config(self, config: Union[DictConfig, Dict], 
                   output_path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_yaml(config)
        else:
            config_dict = config
        
        with open(output_path, 'w') as f:
            if isinstance(config_dict, str):
                f.write(config_dict)
            else:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration: {output_path}")
    
    def get_available_configs(self) -> Dict[str, Path]:
        """
        Get list of available configuration files.
        
        Returns:
            Dictionary mapping config names to file paths
        """
        configs = {}
        
        if self.config_dir.exists():
            for config_file in self.config_dir.glob("*.yaml"):
                config_name = config_file.stem.replace("_config", "")
                configs[config_name] = config_file
        
        return configs
    
    def create_run_config(self, base_config_name: str, 
                         overrides: Dict[str, Any] = None,
                         experiment_name: str = None) -> DictConfig:
        """
        Create configuration for a specific run with overrides.
        
        Args:
            base_config_name: Name of base configuration
            overrides: Dictionary of configuration overrides
            experiment_name: Name for the experiment
            
        Returns:
            Run-specific configuration
        """
        # Load base configuration
        base_config = self.load_model_config(base_config_name)
        
        # Apply overrides if provided
        if overrides:
            base_config = self.merge_configs(base_config, overrides)
        
        # Add experiment metadata
        if experiment_name:
            base_config.experiment = {
                'name': experiment_name,
                'base_config': base_config_name,
                'timestamp': None  # Will be set during training
            }
        
        # Validate final configuration
        self.validate_config(base_config)
        
        return base_config

def setup_logging(config: Union[DictConfig, Dict]):
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration containing logging settings
    """
    logging_config = config.get('logging', {})
    
    level = logging_config.get('level', 'INFO')
    format_str = logging_config.get('format', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str
    )
    
    # Set specific logger levels
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    logger.info(f"Logging configured with level: {level}")

def get_device_config(config: Union[DictConfig, Dict]) -> str:
    """
    Get device configuration.
    
    Args:
        config: Configuration containing device settings
        
    Returns:
        Device string (cuda, mps, or cpu)
    """
    import torch
    
    device_config = config.get('device', {})
    
    if device_config.get('force_cpu', False):
        return 'cpu'
    
    if device_config.get('auto_detect', True):
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    return 'cpu'
