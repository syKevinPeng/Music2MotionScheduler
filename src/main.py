import os, sys
sys.path.append("/fs/nexus-projects/PhysicsFall/")
from Music2MotionScheduler.src.data_processing.dataset import SchdulerDataModule
from Music2MotionScheduler.src.models import baseline
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from pathlib import Path

data_path = Path("/fs/nexus-projects/PhysicsFall/data/Motorica_beats/sliced_audio_features417")
label_path = Path("/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset/all_label.pkl")
saving_dir = Path("/fs/nexus-projects/PhysicsFall/Music2MotionScheduler/ckpt")

max_epochs = 50

# Initialize Wandb logger
wandb_logger = WandbLogger(project='Music2MotionScheduler',
                            name = "test",
                            mode = "disabled",
                            notes = "debugging")

# Initialize dataset and model
data_module = SchdulerDataModule(dataset_path = data_path, label_path = label_path)
batch_size, train_dummy_data, val_dummy_data=data_module.setup()
seq_length, input_dim = train_dummy_data[0].shape
model = baseline.Baseline(
    input_dim=input_dim,
    num_labels=len(data_module.dataset.label_converter.label_list),
    seq_length=seq_length
)


exp_id = wandb_logger.experiment.id
# Define checkpoint callback
checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    monitor='train_loss',
    dirpath=str(saving_dir/exp_id),
    filename='music_scheduler-{epoch:02d}-{train_loss:.2f}',
    mode='min',
    every_n_epochs=30,  
    save_top_k = -1,
)

# Initialize trainer
trainer = pl.Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    max_epochs=max_epochs
)

# Train the model
trainer.fit(model, datamodule=data_module)