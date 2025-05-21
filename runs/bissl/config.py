from typing import Optional, Literal

from source.types import (
    DownstreamDatasets,
    DownstreamTaskTypes,
    PretrainDatasets,
    OptimizerChoices,
    FloatNonNegative,
    FloatBetween0and1,
    BinaryChoices,
    HinvSolverTypes,
)

from runs.config_general import ArgsGeneralDefaults
from runs.fine_tune.config import ArgsFineTuningDefaults


class ArgsBiSSLDefaults(ArgsGeneralDefaults):
    # fmt: off
    pretrained_model_backbone: str  # Name of .pth file with pretrained model backbone weights, e.g. "Pretext_simclr_arch-resnet50_backbone_id-f04gipeq.pth"
    pretrained_model_head: str  # Name of .pth file with pretrained model head weights, e.g. "Pretext_simclr_arch-resnet50_head_id-f04gipeq.json"
    pretrained_model_config: str # Name of .json file with pretrained model config, e.g. "Pretext_simclr_arch-resnet50_config_id-f04gipeq.json"
    
    project_name: Literal["BiSSL"] = "BiSSL"  # Name of the project. Is used in naming for model saving and data logging.

    epochs: int = 500  # Number of training stage alternations (here called epochs) to train the downstream task
    downstream_task: DownstreamTaskTypes  # Type of downstream task.

    # Downstream Data
    d_dataset: DownstreamDatasets  # Downstream dataset name.
    d_batch_size: int  # Downstream batch size

    # Downstream Training
    d_lr: FloatNonNegative  # Upper level (downstream) learning rate
    d_wd: FloatNonNegative  # Upper level (downstream) weight decay
    d_momentum: FloatBetween0and1 = ArgsFineTuningDefaults.momentum  # Upper level (downstream) momentum
    d_optimizer: OptimizerChoices = ArgsFineTuningDefaults.optimizer # Upper level (downstream) optimizer
    d_beta1: FloatBetween0and1 = ArgsFineTuningDefaults.beta1  # Upper level (downstream) beta1 for adam optimizers
    d_beta2: FloatBetween0and1 = ArgsFineTuningDefaults.beta2  # Upper level (downstream) beta2 for adam optimizers

    # Downstream Linear Warmup Hyperparameters
    d_linear_warmup_epochs: int = 20  # Number of epochs for the linear downstream head warmup
    d_linear_warmup_lr: Optional[FloatNonNegative] = None  # Downstream linear warmup learning rate. Defaults to the downstream learning rate d_lr.
    d_linear_warmup_wd: Optional[FloatNonNegative] = None  # Downstream linear warmup weight decay. Defaults to the downstream weight decay d_wd.
    d_linear_warmup_momentum: Optional[FloatBetween0and1] = None  # Downstream linear warmup momentum. Defaults to the downstream momentum d_momentum.

    # Remaining optionals are set equal to the corresponding hyperparameters used for pretraining
    # Pretext Data
    p_batch_size: Optional[int] = None  # Lower level (pretext) batch size. Defaults to the batch size used for pretraining.
    p_dataset: Optional[PretrainDatasets] = None  # Lower level (pretext) dataset name. Defaults to the dataset used for pretraining.
    p_img_size: Optional[int] = None  # Lower level (pretext) image size. Defaults to the image size used for pretraining.
    p_img_crop_min_ratio: Optional[float] = None  # Lower level (pretext) image crop min ratio. Defaults to the crop min ratio used for pretraining.

    # Pretext Training
    p_lr: FloatNonNegative  # Lower level (pretext) learning rate.

    p_wd: Optional[FloatNonNegative] = None  # Lower level (pretext) weight decay. Defaults to the weight decay used for pretraining.
    p_momentum: Optional[FloatBetween0and1] = None  # Lower level (pretext) momentum. Defaults to the momentum used for pretraining.
    p_optimizer: Optional[OptimizerChoices] = None  # Lower level (pretext) optimizer. Defaults to the optimizer used for pretraining.
    p_beta1: Optional[FloatBetween0and1] = None  # Lower level (pretext) beta1 for adam optimizers. Defaults to the beta1 used for pretraining.
    p_beta2: Optional[FloatBetween0and1] = None  # Lower level (pretext) beta2 for adam optimizers. Defaults to the beta2 used for pretraining.

    save_model: BinaryChoices = 1  # Whether to save the (lower-level) pretext model backbone during training

    # BiSSL Hyperparameters
    lower_num_iter: int = 20  # Number of lower level (pretext) training iterations.
    upper_num_iter: int = 8  # Number of upper level (downstream) training iterations.
    # Cg Solver Args
    hinv_solver: HinvSolverTypes = "cg"  # Hessian inverse solver type. "cg" uses the conjugate gradient method.
    cg_lam: float =  0.001  # Lambda parameter for the cg solver and lower level regularization scaling.
    cg_lam_dampening: float = 10.0  # Lambda dampening parameter for the cg solver.
    cg_iter_num: int = 5  # Number of cg iterations.
    cg_verbose: BinaryChoices = 0  # Whether to print cg solver verbose output.

    # fmt: on
