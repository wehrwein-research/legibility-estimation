import sys, json
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import SiameseDeciderOld, BinaryFinetuner, BinaryFinetunerDataset, get_train_and_val_loader 
sys.path.append('../../../data')
sys.path.append('../siamese_experiments')
sys.path.append('../../')
sys.path.append('../../binary')
sys.path.append("../../pairwise")

PROJ_ROOT = str(Path(*Path.cwd().parts[:Path().cwd().parts.index('legibility-estimation')+1]))
SAVE_DIR = PROJ_ROOT + '/MODEL/bordercut/weights/'
N=1

def get_config(args):
    cfg = args[1]
    cfg = json.load(open(cfg))
    return cfg

def main():
    we_logging = False if len(sys.argv) < 3 else sys.argv[2] != 'no'
    print('We logging?:', we_logging)
    num = '' if len(sys.argv) < 4 else sys.argv[3] 
    cfg = get_config(sys.argv)
    weight_path = SAVE_DIR + Path(cfg["name"]).stem+num + ".ckpt"
    prefix = cfg["scrape_dir"]
    ckpt = cfg["trained_ckpt"]

    logger = pl.loggers.wandb.WandbLogger(name=Path(cfg["name"]).stem+num, 
                                          project="mix-models") if we_logging else False
    checkpoint = ModelCheckpoint(
        dirpath=SAVE_DIR,
        monitor='val_acc',
        mode='max',
        filename=Path(cfg["name"]).stem + num
    )

    trained_model = SiameseDeciderOld.load_from_checkpoint(ckpt)
    model = BinaryFinetuner(trained_model, prefix=prefix, train_path=cfg['train_path'], val_path=cfg['val_path'], 
            concat=cfg['concat'], lr=cfg['lr'], 
            batch_size=cfg['batch_size'], weight_decay=cfg['weight_decay'],  blur_kernal_size=cfg['blur_kernal_size'], 
            colorshift=cfg['colorshift'])

    if torch.cuda.is_available():
        model.cuda()

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=4, verbose=False, mode="min")
    callbacks = [checkpoint, early_stop_callback] if cfg['early_stopping'] else [checkpoint]
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=cfg["num_epochs"],
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=2
    )
    trainer.fit(model)

if __name__ == "__main__":
    for i in range(N):
        main()
