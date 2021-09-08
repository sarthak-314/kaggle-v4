import tensorflow as tf
try: 
    import tensorflow_hub as hub
except: 
    print('Tensorflow Hub not availible')
try: 
    import tensorflow_addons as tfa 
except: 
    print('Tensorflow addons not availible')

from termcolor import colored 
import datetime
import os

AUTO = { 'num_parallel_calls': tf.data.AUTOTUNE }
##### Startup Functions #####
def tf_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy

def mixed_precision(): 
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

def jit(): 
    tf.config.optimizer.set_jit(True)


##### Important Functions #####
def save_weights(model, filepath):
    filepath = str(filepath)
    print('Saving model weights at', colored(filepath, 'blue'))
    model.save_weights(filepath=filepath, options=get_save_locally())

def get_gcs_path(dataset_name): 
    from kaggle_datasets import KaggleDatasets
    return KaggleDatasets().get_gcs_path(dataset_name)
    
def get_save_locally(): 
    return tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

def get_load_locally(): 
    return tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')


##### Factory Functions #####
def tf_lr_scheduler_factory(lr_scheduler_kwargs): 
    if isinstance(lr_scheduler_kwargs, float): 
        print(colored('Using constant learning rate', 'yellow'))
        return lr_scheduler_kwargs
    lr_scheduler = tfa.optimizers.ExponentialCyclicalLearningRate(
        initial_learning_rate=lr_scheduler_kwargs['init_lr'], 
        maximal_learning_rate=lr_scheduler_kwargs['max_lr'], 
        step_size=lr_scheduler_kwargs['step_size'], 
        gamma=lr_scheduler_kwargs['gamma'], 
    )
    return lr_scheduler

def tf_optimizer_factory(optimizer_kwargs, lr_scheduler): 
    optimizer_name = optimizer_kwargs['name']
    if optimizer_name == 'AdamW': 
        optimizer = tfa.optimizers.AdamW(
            weight_decay=optimizer_kwargs['weight_decay'],
            learning_rate=lr_scheduler,  
            amsgrad=False, 
            clipnorm=optimizer_kwargs['max_grad_norm'], 
        )
    elif optimizer_name == 'Adagrad': 
        optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=lr_scheduler, 
        )
        print('Skipping weight decay for Adagrad')
    if optimizer_kwargs['use_lookahead']: 
        print(colored('Using Lookahead', 'red'))
        optimizer = tfa.optimizers.Lookahead(optimizer)
    if optimizer_kwargs['use_swa']: 
        print(colored('Using SWA', 'red'))
        optimizer = tfa.optimizers.SWA(optimizer)
    return optimizer


# CALLBACKS 
MONITOR = 'val_loss'
MODE = 'min'
VERBOSE = 2
COMMON_KWARGS = {
    'monitor': MONITOR, 
    'mode': MODE, 
    'verbose': VERBOSE, 
}

def tb(tb_dir, train_steps): 
    start_profile_batch = train_steps+10
    stop_profile_batch = start_profile_batch + 100
    profile_range = f"{start_profile_batch},{stop_profile_batch}"
    log_path = tb_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_path, histogram_freq=1, update_freq=20,
        profile_batch=profile_range, 
    )
    return tensorboard_callback
    
def checkpoint(checkpoint_dir, kwargs=COMMON_KWARGS):
    checkpoint_filepath = 'checkpoint.h5'
    if checkpoint_dir is not None: 
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_filepath = checkpoint_dir / checkpoint_filepath
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_best_only=True, 
        **kwargs, 
    )

def early_stop(patience=3, kwargs=COMMON_KWARGS):
    return tf.keras.callbacks.EarlyStopping(
        patience=patience, 
        restore_best_weights=True, 
        **kwargs,
    )

def reduce_lr_on_plateau(patience, kwargs=COMMON_KWARGS): 
    return tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=patience,
        min_delta=0.0001,
        min_lr=0,
        **kwargs, 
    )

def time_stop(max_train_hours):
    import tensorflow_addons as tfa 
    return tfa.callbacks.TimeStopping(
        seconds=max_train_hours*3600, 
        verbose=1, 
    )
    
def tqdm_bar(): 
    import tensorflow_addons as tfa
    return tfa.callbacks.TQDMProgressBar()

def terminate_on_nan(): 
    return tf.keras.callbacks.TerminateOnNaN()

def tensorboard_callback(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir)
    )

def wandb_callback():
    from wandb.keras import WandbCallback
    return WandbCallback()

def make_callbacks_list(model, callbacks): 
    return tf.keras.callbacks.CallbackList(
        callbacks, 
        add_progbar = True, 
        model = model,
        add_history=True, 
    )


