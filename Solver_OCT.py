gpus = "0,1"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import warnings
warnings.filterwarnings("ignore")

import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

import timm
import torch
import torch.nn as nn
import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from utils.loss.smooth import LabelSmoothingLoss
from utils.mixup import mixup_data, mixup_criterion
pl.seed_everything(0)


class Model(pl.LightningModule):
    def __init__(self, **args):
        super(Model, self).__init__()
        for k, v in args.items():
            setattr(self, k, v)
        self.args = args
        self.model = timm.create_model(self.model_name, pretrained = True, in_chans = 1, num_classes = self.num_classes, drop_rate = self.drop_rate)
        self.criterion = LabelSmoothingLoss(classes = self.num_classes, smoothing = self.smoothing)
        self.save_hyperparameters()

    class Data(Dataset):
        def __init__(self, df, trans, **args):
            self.df = df
            self.trans = trans
            for k, v in args.items():
                setattr(self, k, v)
        
        def __getitem__(self, idx):
            image = np.array(Image.open(self.df.loc[idx, "oct_file"]))
            label = np.array(self.df.loc[idx, "label"])

            if self.trans is not None:
                image = self.trans(image = image)["image"]
            return image, label

        def __len__(self):
            return len(self.df)

    def prepare_data(self):
        img_files = sorted(glob.glob("./data/train/images/*/*_crop.jpg"))
        oct_files = sorted(glob.glob("./data/train/images/*/*/*_crop.png"))

        labels = pd.read_csv("./data/train/train.csv")
        labels["label"] = labels.non + 2 * labels.early + 3 * labels.mid_advanced - 1
        labels["uid"] = labels.pop("data")

        df_img = pd.DataFrame({"img_file": img_files})
        df_img["uid"] = df_img.img_file.apply(lambda x: int(os.path.basename(os.path.dirname(x))))
        df_oct = pd.DataFrame({"oct_file": oct_files})
        df_oct["uid"] = df_oct.oct_file.apply(lambda x: int(os.path.basename(os.path.dirname(x))))
        df_oct = df_oct.iloc[::5]

        df = labels.merge(df_img, on = "uid", how = "outer").merge(df_oct, on = "uid", how = "outer")
        df = df.reset_index(drop = True)

        split = GroupKFold(5)
        train_idx, valid_idx = list(split.split(df, groups = df.uid))[self.fold]
        self.df_train = df.loc[train_idx].reset_index(drop = True) if self.fold != -1 else df.reset_index(drop = True)
        self.df_valid = df.loc[valid_idx].reset_index(drop = True)
        self.ds_train = self.Data(self.df_train, self.trans_train, **self.args)
        self.ds_valid = self.Data(self.df_valid, self.trans_valid, **self.args)

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.batch_size, shuffle = True, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, self.batch_size, num_workers = 4)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate, weight_decay = 2e-5)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.learning_rate, steps_per_epoch = int(len(self.train_dataloader())), epochs = self.num_epochs, anneal_strategy = "linear", final_div_factor = 30,), 'name': 'learning_rate', 'interval':'step', 'frequency': 1}
        return [optimizer], [lr_scheduler]

    def on_fit_start(self):
        metric_placeholder = {"valid_metric": 0}
        self.logger.log_hyperparams(self.hparams, metrics = metric_placeholder)

    def forward(self, x):
        yhat = self.model(x)
        return yhat

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.alpha != 0:
            x, ya, yb, lam = mixup_data(x, y, self.alpha)
            yhat = self(x)
            loss = mixup_criterion(self.criterion, yhat, ya, yb, lam)
        else:
            yhat = self(x)
            loss = self.criterion(yhat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.criterion(yhat, y)
        self.log("valid_loss", loss, prog_bar = True)
        return y, yhat

    def validation_step_end(self, output):
        return output

    def validation_epoch_end(self, outputs):
        y = torch.cat([_[0] for _ in outputs]).detach().cpu().numpy()
        yhat = torch.cat([_[1] for _ in outputs]).argmax(1).detach().cpu().numpy()
        df = self.val_dataloader().dataset.df.iloc[:len(y)]
        df["pred"] = yhat
        y = df.groupby("uid").agg("mean").label.round().astype(int)
        yhat = df.groupby("uid").agg("mean").pred.round().astype(int)
        kap = cohen_kappa_score(y, yhat, weights = "quadratic")
        self.log("valid_metric", kap, prog_bar = True)

args = dict(
    learning_rate = 1e-3,
    model_name = "tf_efficientnet_b0_ns",
    num_epochs = 30,
    batch_size = 64,
    fold = 4,
    num_classes = 3,
    smoothing = 0.,
    alpha = 1,
    image_size = 384,
    drop_rate = 0.5,
    swa = False,
    name = "OCT/b0ns",
    version = "v2_0.2"
)
args['trans_train'] = A.Compose([
    A.Resize(args['image_size'], args['image_size']),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.GridDistortion(),
    A.PiecewiseAffine(),
    A.Normalize([0], [1]),
    ToTensorV2()])
args['trans_valid'] = A.Compose([
    A.Resize(args['image_size'], args['image_size']),
    A.Normalize([0], [1]),
    ToTensorV2()])

if __name__ == "__main__":
    logger = TensorBoardLogger("./logs", name = args["name"], version = args["version"], default_hp_metric = False)
    callback = pl.callbacks.ModelCheckpoint(
        filename = '{epoch}_{valid_metric:.3f}',
        save_last = True,
        mode = "max",
        monitor = 'valid_metric'
    )
    model = Model(**args)
    trainer = pl.Trainer(
        gpus = len(gpus.split(",")), 
        precision = 16, amp_backend = "native", amp_level = "O1", 
        accelerator = "dp",
        gradient_clip_val = 10,
        max_epochs = args["num_epochs"],
        stochastic_weight_avg = args["swa"],
        logger = logger,
        progress_bar_refresh_rate = 10,
        callbacks = [callback]
    )
    trainer.fit(model)