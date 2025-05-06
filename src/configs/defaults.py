from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "baseline"
_C.MODEL.SEQ_LENGTH = 1000
_C.MODEL.INPUT_DIM = 417
_C.MODEL.NUM_LABELS = 232
_C.MODEL.DIM_MODEL = 256
_C.MODEL.NHEAD = 8
_C.MODEL.NUM_LAYERS = 8


_C.WANDB = CN()
_C.WANDB.PROJECT = "Music2MotionScheduler"
_C.WANDB.NAME = "baseline0"
_C.WANDB.MODE = "online"
_C.WANDB.NOTES = "baseline model"
_C.WANDB.SAVE_DIR = "/fs/nexus-projects/PhysicsFall/Music2MotionScheduler/ckpt"

_C.DATASET = CN()
_C.DATASET.PATH = "/fs/nexus-projects/PhysicsFall/editable_dance_project/data/motorica_beats"
_C.DATASET.LABEL_FORMAT = "number"  # "one_hot" or "number"

_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCHS = 500
_C.TRAIN.BATCH_SIZE = 32



def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values.
    """
    return _C.clone()