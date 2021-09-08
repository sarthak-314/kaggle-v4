from pytorch_lightning.callbacks import (
    BackboneFinetuning, EarlyStopping, LearningRateMonitor, ModelCheckpoint
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pl_bolts.callbacks import (
    TrainingDataMonitor, ModuleDataMonitor, BatchGradientVerificationCallback
)
from pl_bolts.callbacks.printing import PrintTableMetricsCallback
import os 


def backbone_finetuning(unfreeze_epoch=10, backbone_initial_ratio_lr=0.1, lr_multiplier=1.5): 
    # NOTE: Backbone should be at model.backbone
    
    # Scheduler for increasing backbone learning rate (epoch > unfreeze_epoch)
    multiplier = lambda epoch: lr_multiplier
    backbone_finetuning = BackboneFinetuning(
        unfreeze_backbone_at_epoch = unfreeze_epoch, # Backbone freezed till this epoch
        lambda_func = multiplier,
        backbone_initial_ratio_lr = backbone_initial_ratio_lr, # initial backbone lr = current lr * initial backbone ratio
        verbose = True, # Display the backbone and model learning rates
        should_align = True, # Align the learning rates
    )
    return backbone_finetuning

def early_stopping(patience=5, monitor='val/acc', mode='max'): 
    print(f'EarlyStopping: Will wait for {patience} epochs for the {monitor} to improve and then stop training')
    early_stopping = EarlyStopping(
        patience = patience, 
        monitor = monitor, 
        mode = mode, 
    )        
    return early_stopping

def lr_monitor(logging_interval='step'): 
    print(f'LearningRateMonitor: Will log learning rate for learning rate schedulers every {logging_interval} during training')
    return LearningRateMonitor(
        logging_interval = logging_interval, 
        log_momentum = True, # for moment based optimizers
    )

def model_checkpoint(checkpoint_dir='./', filename='epoch_{epoch:02d}-loss_{val/loss:.4f}-acc_{val/acc:.4f}', save_top_k=3, monitor='val/acc', mode='max'):
    checkpoint_dir = str(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'Save top {save_top_k} models at {checkpoint_dir} with name {filename} if score increases')
    model_checkpoint = ModelCheckpoint(
        dirpath = checkpoint_dir,  
        filename = filename, 
        save_top_k = save_top_k, 
        save_last = True, 
        monitor = monitor, 
        mode = mode, 
    )
    return model_checkpoint

def print_table_metrics(): 
    print('PrintTableMetricsCallback: Will print table metrics at every epoch ending')
    return PrintTableMetricsCallback()

#TODO: Learn more about model pruning and it's callback
#TODO: Learn more about stochastic weight averaging and it's parameters

def training_data_monitor(log_every_n_steps=25): 
    print('Create histogram for each batch input in training step')
    return TrainingDataMonitor(
        log_every_n_steps = log_every_n_steps
    )
    
def module_data_monitor(): 
    print(f'ModuleDataMonitor: Histograms of data passing through a .forward pass')
    return ModuleDataMonitor()

def batch_grad_verification():     
    print(f'BatchGradientVerificationCallback: Lightweight verification callback')
    return BatchGradientVerificationCallback()

# LOGGERS: Tensorboard & Wandb
def tb_logger(tb_dir): 
    return TensorBoardLogger(str(tb_dir))

def wandb_logger(wandb_run, wandb_dir, project): 
    return WandbLogger(name=wandb_run, save_dir=str(wandb_dir), project=project)