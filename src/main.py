import os, sys
sys.path.append("/ihchomes/peng2000/editdance/")
from Music2MotionScheduler.src.data_processing.dataset import SchdulerDataModule
from Music2MotionScheduler.src.models import baseline
from Music2MotionScheduler.src.models import seq2seq
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from pathlib import Path
from Music2MotionScheduler.src.configs.defaults import get_cfg_defaults

# Load configuration
cfg = get_cfg_defaults()
config_file = Path(".")/"configs"/"seq2seq0.yaml"
if not config_file.exists():
    raise FileNotFoundError(f"Config file {config_file} does not exist")
cfg.set_new_allowed(True)
cfg.merge_from_file(config_file)
cfg.freeze()


# Initialize Wandb logger
wandb_logger = WandbLogger(project=cfg.WANDB.PROJECT,
                            name = cfg.WANDB.NAME,
                            mode = cfg.WANDB.MODE,
                            notes = cfg.WANDB.NOTES,
                            config = cfg)

# add config to wandb
wandb_logger.experiment.config.update(cfg)

# Initialize dataset and model
data_module = SchdulerDataModule(dataset_path = cfg.DATASET.PATH, batch_size=cfg.TRAIN.BATCH_SIZE, label_format=cfg.DATASET.LABEL_FORMAT, class_list_path=cfg.DATASET.LABEL_LIST_PATH)
label_size = data_module.get_label_length()
batch_size, train_dummy_data, val_dummy_data=data_module.setup()
seq_length, input_dim = train_dummy_data[0].shape
label = train_dummy_data[1]

print(f'label shape: {label.shape}')
print(f'seq_length: {seq_length}, input_dim: {input_dim}')
print(f'num labels: {len(data_module.dataset.label_converter.label_list)}')

if cfg.MODEL.NAME == "baseline":
    model = baseline.Baseline(
        input_dim=cfg.MODEL.INPUT_DIM,
        num_labels=cfg.MODEL.NUM_LABELS,
        seq_length=cfg.MODEL.SEQ_LENGTH,
        d_model = cfg.MODEL.DIM_MODEL,
        nhead = cfg.MODEL.NHEAD,
        num_layers = cfg.MODEL.NUM_LAYERS,

    )
elif cfg.MODEL.NAME == "seq2seq":
    model = seq2seq.Seq2SeqDanceGenerator(
        music_feature_dim = cfg.MODEL.INPUT_DIM,
        dance_label_vocab_size = cfg.MODEL.NUM_LABELS,
        embed_dim = cfg.MODEL.EMBED_DIM,
        hidden_dim = cfg.MODEL.HIDDEN_DIM,
        n_layers = cfg.MODEL.NUM_LAYERS,
        dropout = cfg.MODEL.DROPOUT,
        learning_rate = cfg.MODEL.LEARNING_RATE,
        max_output_len = cfg.MODEL.MAX_OUTPUT_LEN,
        teacher_forcing_ratio = cfg.MODEL.TEACHER_FORCING_RATIO,
    )
else:
    raise ValueError(f"Model {cfg.MODEL.NAME} not supported")


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