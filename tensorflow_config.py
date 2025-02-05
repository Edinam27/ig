import logging
import tensorflow as tf
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import psutil
import platform
import yaml

class TensorFlowConfigManager:
    """Manages TensorFlow configuration with robust CPU/GPU handling."""
    
    def __init__(self, memory_limit: Optional[float] = None, config_path: str = 'config/tf_config.yaml'):
        self.logger = logging.getLogger(__name__)
        self.memory_limit = memory_limit
        self.config_path = Path(config_path)
        self._setup_logging()
        self._load_config()
        
    def _setup_logging(self) -> None:
        """Configure logging for TensorFlow operations."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler('logs/tensorflow.log')
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)

    def _load_config(self) -> None:
        """Load TensorFlow configuration from YAML file."""
        if not self.config_path.exists():
            self._create_default_config()
            
        try:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = self._get_default_config()

    def _create_default_config(self) -> None:
        """Create default configuration file."""
        self.config_path.parent.mkdir(exist_ok=True)
        config = self._get_default_config()
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f)
        except Exception as e:
            self.logger.error(f"Failed to create default config: {e}")

    def _get_default_config(self) -> Dict:
        """Get default TensorFlow configuration."""
        return {
            'gpu': {
                'memory_growth': True,
                'memory_limit': 0.7,
                'allow_growth': True
            },
            'cpu': {
                'num_threads': psutil.cpu_count(logical=False)
            },
            'mixed_precision': False,
            'eager_execution': True
        }

    def configure_tensorflow(self) -> None:
        """Configure TensorFlow with automatic GPU/CPU fallback."""
        try:
            # Clear any existing configurations
            tf.keras.backend.clear_session()
            
            # Enable eager execution first
            if self.config['eager_execution']:
                tf.compat.v1.enable_eager_execution()
            
            # Try GPU configuration first
            if self._configure_gpu():
                self.logger.info("GPU configuration successful")
            else:
                self._configure_cpu()
                self.logger.info("CPU configuration successful")
                
            # Additional optimizations
            self._configure_mixed_precision()
            
        except Exception as e:
            self.logger.error(f"TensorFlow configuration failed: {e}")
            self._configure_cpu()  # Fallback to CPU
            raise RuntimeError(f"TensorFlow configuration error: {e}")

    def _configure_gpu(self) -> bool:
        """Configure GPU settings for TensorFlow."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return False
                
            for gpu in gpus:
                # Enable memory growth
                if self.config['gpu']['memory_growth']:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit
                memory_limit = self.memory_limit or self.config['gpu']['memory_limit']
                if memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=int(memory_limit * 1024)
                        )]
                    )
            
            return True
            
        except Exception as e:
            self.logger.warning(f"GPU configuration failed: {e}")
            return False

    def _configure_cpu(self) -> None:
        """Configure TensorFlow for CPU-only operation."""
        try:
            # Hide GPU devices
            tf.config.set_visible_devices([], 'GPU')
            
            # Configure threading
            num_threads = self.config['cpu']['num_threads']
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)
            
        except Exception as e:
            self.logger.error(f"CPU configuration failed: {e}")
            raise

    def _configure_mixed_precision(self) -> None:
        """Configure mixed precision training if enabled."""
        if self.config['mixed_precision']:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
            except Exception as e:
                self.logger.warning(f"Mixed precision configuration failed: {e}")

    def verify_configuration(self) -> Tuple[bool, str]:
        """Verify TensorFlow configuration with a simple operation test."""
        try:
            # Test basic operations
            with tf.device('/CPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
                c = tf.matmul(a, b)
            
            # Test model creation
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(1, input_shape=(1,))
            ])
            
            return True, "Configuration verified successfully"
        except Exception as e:
            return False, f"Configuration verification failed: {str(e)}"

    def get_system_info(self) -> Dict:
        """Get system and TensorFlow configuration information."""
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'tensorflow_version': tf.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024 ** 3),
            'cuda_available': tf.test.is_built_with_cuda(),
            'gpu_devices': [str(gpu) for gpu in tf.config.list_physical_devices('GPU')],
            'eager_execution': tf.executing_eagerly()
        }