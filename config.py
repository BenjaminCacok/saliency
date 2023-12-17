from easydict import EasyDict

cfg = EasyDict()

cfg.DATA = EasyDict()
cfg.DATA.RESOLUTION = (352, 352)
cfg.DATA.SALICON_ROOT = "C:/Users/18336/Documents/SJTUTIS/ours/type3"
cfg.DATA.SALICON_TRAIN = "./dataset/type3_train.csv"
cfg.DATA.SALICON_VAL = "./dataset/type3_val.csv"

# Train
cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 2

cfg.SOLVER = EasyDict()
cfg.SOLVER.LR = 1e-4
cfg.SOLVER.MIN_LR = 1e-8
cfg.SOLVER.MAX_EPOCH = 30
