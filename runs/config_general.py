from typing import Optional
from tap import Tap
from source.types import BinaryChoices

SEED_MAX_VALUE = 2**30


# In tap, argument help strings are added as a comment on the same line as its corresponding argument.
class ArgsGeneralDefaults(Tap):
    # fmt: off
    root: str  # root dir of project
    dataset_root: Optional[str] = None  # Optional custom path to dataset dir (will be set to root + "/data" if None)
    model_root: Optional[str] = None  # Optional custom path to model dir (will be set to root + "/models" if None)
    
    project_name: str  # Name of the project. Is used as the folder name for model saving.
    code_dir_to_log: Optional[str] = None # Optional custom path to dir which contains all code to be logged to wandb. (Default (None) is the root dir)
    
    download: BinaryChoices = 0  # Download dataset if not found in root

    # device: Literal["cuda"] = "cuda"  # Device, currently only supports cuda
    num_workers: int = 10  # Number of workers for dataloader

    # Distributed Setup
    omp_num_threads: int = 1  # Number of threads
    dist_backend = "nccl"  # Backend for distributed training
    dist_url: str = "env://"  # URL for distributed training

    rank: int = 0  # AKA global rank, will be overwriten during runtime
    gpu: int = 0  # AKA local rank, will be overwriten during runtime
    world_size: int = 1  # Number of GPUs per node, will be overwritten during runtime

    cudnn_benchmark: BinaryChoices = 0  # Use cudnn benchmark. May lead to improved training speed, but can be of the cost of reproducibility.

    seed: Optional[int] = None  # Optional seed for random number generator. If None, the script will overwrite this value with a randomly generated seed, that is stored in the wandb config.

    wandb_api_key: Optional[str] = None  # Optional wandb API key. This is used to log the model to wandb.

    # fmt: on

    def __init__(self, *args, **kwargs):
        super().__init__(underscores_to_dashes=True, *args, **kwargs)


# These keys are usually set during runtime and are not relevant for use in subsequent training processes.
# Hence, they are not saved to the config file during training.
CONFIG_KEYS_IGNORE_WHEN_SAVING = [
    "root",
    "dataset_root",
    "model_root",
    "code_dir_to_log",
    "download",
    "num_workers",
    "omp_num_threads",
    "dist_backend",
    "dist_url",
    "rank",
    "gpu",
    "world_size",
    "cudnn_benchmark",
    "wandb_api_key",
]
