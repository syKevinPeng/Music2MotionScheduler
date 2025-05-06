import os, sys
sys.path.append("/fs/nexus-projects/PhysicsFall/")
from Music2MotionScheduler.src.data_processing.dataset import SchdulerDataModule
from Music2MotionScheduler.src.models import baseline
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from pathlib import Path
from Music2MotionScheduler.src.configs.defaults import get_cfg_defaults

# Load configuration
cfg = get_cfg_defaults()
config_file = Path(".")/"configs"/"baseline0.yaml"
if not config_file.exists():
    raise FileNotFoundError(f"Config file {config_file} does not exist")
cfg.merge_from_file(config_file)
cfg.freeze()


saving_dir = Path("/fs/nexus-projects/PhysicsFall/Music2MotionScheduler/ckpt")


# Initialize Wandb logger
wandb_logger = WandbLogger(project=cfg.WANDB.PROJECT,
                            name = cfg.WANDB.NAME,
                            mode = cfg.WANDB.MODE,
                            notes = cfg.WANDB.NOTES,
                            config = cfg)

# Initialize dataset and model
data_module = SchdulerDataModule(dataset_path = cfg.DATASET.PATH, batch_size=cfg.TRAIN.BATCH_SIZE, label_format=cfg.DATASET.LABEL_FORMAT)
batch_size, train_dummy_data, val_dummy_data=data_module.setup()
# seq_length, input_dim = train_dummy_data[0].shape
# print(f'seq_length: {seq_length}, input_dim: {input_dim}')
# print(f'num labels: {len(data_module.dataset.label_converter.label_list)}')
# exit()
model = baseline.Baseline(
    input_dim=cfg.MODEL.INPUT_DIM,
    num_labels=cfg.MODEL.NUM_LABELS,
    seq_length=cfg.MODEL.SEQ_LENGTH,
    d_model = cfg.MODEL.DIM_MODEL,
    nhead = cfg.MODEL.NHEAD,
    num_layers = cfg.MODEL.NUM_LAYERS,

)


exp_id = wandb_logger.experiment.id
# Define checkpoint callback
checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    monitor='train_loss',
    dirpath=str(Path(cfg.WANDB.SAVE_DIR)/exp_id),
    filename='music_scheduler-{epoch:02d}-{train_loss:.2f}',
    mode='min',
    every_n_epochs=30,  
    save_top_k = -1,
)

# Initialize trainer
trainer = pl.Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    max_epochs=cfg.TRAIN.MAX_EPOCHS,
)

# Train the model
trainer.fit(model, datamodule=data_module)