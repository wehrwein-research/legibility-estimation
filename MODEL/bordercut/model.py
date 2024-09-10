from pathlib import Path
import torch
import wandb
import torchvision.models as models
from torchvision.transforms import v2
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torchmetrics.classification import Accuracy
from einops import rearrange
import sys
from kornia.utils import image_to_tensor
from sultan.api import Sultan as Bash

PROJ_ROOT = Path(*Path.cwd().parts[:Path().cwd().parts.index('legibility-estimation')+1])

from dataset import get_train_and_val_loader

sys.path.append(str(PROJ_ROOT) + '/data')
import border_utils as bu
sys.path.append(str(PROJ_ROOT) + '/MODEL/binary')

class BinaryFinetunerDataset(Dataset):
    def __init__(self, gt_file, augment=False, blur_kernal_size=9, blur_sigma=1.3,
            colorshift=0.25, prefix=f'{str(PROJ_ROOT)}/bing_maps/global-rectified-point4/'):
        self.prefix = prefix
        self.data, self.labels =  self.format_data(gt_file)
        self.augment = augment
        if self.augment:
            self.transforms = v2.Compose([
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
                v2.GaussianBlur(kernel_size=blur_kernal_size, sigma=blur_sigma),
                v2.ColorJitter(hue=colorshift),
                v2.ConvertImageDtype(torch.float32)
            ])
        else:
            self.transforms = None

    def format_data(self, gt_file):
        id_to_file = lambda idee: idee[:7] + "/" + idee + "/" + idee + ".jpeg"
        img_labels = []
        imgs = []
        for row in gt_file:
            split = row.split(',')
            img = bu.imread(self.prefix + id_to_file(split[0]))
            imgs.append(img)
            agreement = float(split[2])
            img_labels.append(torch.FloatTensor([agreement]))

        return imgs, img_labels

    ''' Turns np.array into a model ready torch array with augmentations applied '''
    def prep_img(self, x, augment):
        x = image_to_tensor(x, keepdim=False)
        if augment:
            x = self.transforms(x)
        else:
            convert = v2.ConvertImageDtype(torch.float32)
            x = convert(x)
        return x.squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        return self.prep_img(x, self.augment), self.labels[idx]

''' Creates and returns model ready train and validation loaders '''
def get_binary_train_and_val_loader(train_path, prefix, augment, val_path=None, shuffle_train=True, batch_size=4, 
        blur_kernal_size=9, colorshift=0.25):

    blur_sigma = (blur_kernal_size-1)/6
    
    if val_path is not None:
        bash = Bash()
        v = bash.cat(f'{val_path}').run().stdout
        valset = BinaryFinetunerDataset(v, False, blur_kernal_size, blur_sigma, colorshift, prefix)
        val = DataLoader(valset, batch_size=batch_size, num_workers=8)
    else:
        val = None
    
    if shuffle_train:
        t = bu.shuf_file(train_path)
    else:
        bash = Bash()
        t = bash.cat(f'{train_path}').run().stdout
    trainset = BinaryFinetunerDataset(t, augment, blur_kernal_size, blur_sigma, colorshift, prefix)
    train = DataLoader(trainset, batch_size=batch_size, num_workers=8)

    return train, val

class BinaryFinetuner(pl.LightningModule):
    def __init__(self, trained_model, prefix=f'{str(PROJ_ROOT)}/bing_maps/global-rectified-point4/',
        train_path=None, val_path=None, sweep=False, dropout=0.3, concat=True, n=None, 
        split=None, lr=None, batch_size=4, weight_decay=0, blur_kernal_size=9, colorshift=0.25):
        super().__init__()

        self.back = nn.Sequential(*list(trained_model.children())[:-2])
        self.mid = nn.Sequential(*list(trained_model.children())[1][:-1])

        self.concat = concat
        if self.concat:
            self.binary_head = nn.Sequential(
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
        else:
            self.binary_head = nn.Sequential(
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )

        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task='binary', threshold=0.5)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(dropout)

        self.prefix = prefix
        self.sweep = sweep

        if train_path is not None:
            t, v = get_binary_train_and_val_loader(train_path, prefix, True, val_path, True, batch_size, 
                    blur_kernal_size, colorshift)
        else:
            t, v = None, None
        self.trainloader = t
        self.valloader = v

        self.training_step_outputs = []
        self.training_step_gts = []
        self.training_losses = []

        self.validation_step_outputs = []
        self.validation_step_gts = []
        self.validation_losses = []

    # Trains on a single tile
    def forward_no_concat(self, x):
        z1 = self.back(x)
        z = rearrange(z1, 'b h w c -> b (c h w)')
        z = z.to(torch.float)
        z = self.dropout(z)
        return self.binary_head(z)

    # Trains on a single tile concatenated to itself
    def forward_concat(self, x):
        z1 = self.back(x)
        z2 = z1.detach().clone()
        z3 = rearrange([z1, z2], 'n b c h w -> b (n c h w)')
        if self.mid is not None:
            z = self.mid(z3)
        else:
            z = z3
        z = self.dropout(z)
        return self.binary_head(z)

    def forward(self, x):
        return self.forward_concat(x) if self.concat else self.forward_no_concat(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        for pred in y_hat:
            self.training_step_outputs.append(round(pred.tolist()[0]))
        for gt in y:
            self.training_step_gts.append(round(gt.tolist()[0]))
        self.training_losses.append(loss)
        return loss

    def on_train_epoch_end(self):
        acc = self.accuracy(torch.as_tensor(self.training_step_outputs), torch.as_tensor(self.training_step_gts))
        loss = sum(training_loss for training_loss in self.training_losses) / len(self.training_losses)
        self.training_step_outputs.clear()
        self.training_step_gts.clear()
        self.training_losses.clear()
        self.log("train_loss", loss) 
        self.log("train_acc", acc) 
        if self.sweep:
            wandb.log({"train_loss": loss, "train_acc":acc}, commit=True) 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        for pred in y_hat:
            self.validation_step_outputs.append(pred.tolist()[0])
        for gt in y:
            self.validation_step_gts.append(round(gt.tolist()[0])) 
        self.validation_losses.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        acc = self.accuracy(torch.as_tensor(self.validation_step_outputs), torch.as_tensor(self.validation_step_gts))
        loss = sum(validation_loss for validation_loss in self.validation_losses) / len(self.validation_losses)
        self.validation_step_outputs.clear()
        self.validation_step_gts.clear()
        self.validation_losses.clear()
        self.log("val_loss", loss) 
        self.log("val_acc", acc) 
        if self.sweep:
            wandb.log({"val_loss": loss, "val_acc":acc}, commit=True) 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader

class BinaryModel(pl.LightningModule):
    def __init__(self, prefix=f'{str(PROJ_ROOT)}/bing_maps/global-rectified-point4/',
        train_path=None, val_path=None, sweep=False, dropout=0.3, n=None, 
        split=None, lr=None, batch_size=4, weight_decay=0, blur_kernal_size=7, colorshift=0.25, backbone="vit"):
        super().__init__()

        assert backbone in ["vit", "resnet"]
        self.backbone = backbone

        if self.backbone == "resnet":
            resnet = models.resnet18(pretrained=True)
            self.back = nn.Sequential(*list(resnet.children())[:-1])

            self.head = nn.Sequential(
                nn.Linear(512,1),
                nn.Sigmoid()
            )

        elif backbone == "vit":
            weights = models.ViT_B_16_Weights.DEFAULT
            self.back = models.vit_b_16(weights=weights)
            head = nn.Sequential(
                nn.Linear(768,1),
                nn.Sigmoid()
            )
            self.preprocess = weights.transforms()
            self.back.heads = head

        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task='binary', threshold=0.5)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(dropout)

        self.prefix = prefix
        self.sweep = sweep

        if train_path is not None:
            t, v = get_binary_train_and_val_loader(train_path, prefix, True, val_path, True, batch_size, 
                    blur_kernal_size, colorshift)
        else:
            t, v = None, None
        self.trainloader = t
        self.valloader = v

        self.training_step_outputs = []
        self.training_step_gts = []
        self.training_losses = []

        self.validation_step_outputs = []
        self.validation_step_gts = []
        self.validation_losses = []

    def forward(self, x):
        if self.backbone == "vit":
            x_p = self.preprocess(x)
        else:
            x_p = x
        x_d = self.dropout(x_p)
        z1 = self.back(x_d)
        if self.backbone == "resnet":
            z1 = rearrange([z1], 'n b c h w -> b (n c h w)')
            return self.head(z1)
        else:
            return z1

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        for pred in y_hat:
            self.training_step_outputs.append(round(pred.tolist()[0]))
        for gt in y:
            self.training_step_gts.append(round(gt.tolist()[0]))
        self.training_losses.append(loss)
        return loss

    def on_train_epoch_end(self):
        acc = self.accuracy(torch.as_tensor(self.training_step_outputs), torch.as_tensor(self.training_step_gts))
        loss = sum(training_loss for training_loss in self.training_losses) / len(self.training_losses)
        self.training_step_outputs.clear()
        self.training_step_gts.clear()
        self.training_losses.clear()
        self.log("train_loss", loss) 
        self.log("train_acc", acc) 
        if self.sweep:
            wandb.log({"train_loss": loss, "train_acc":acc}, commit=True) 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        for pred in y_hat:
            self.validation_step_outputs.append(pred.tolist()[0])
        for gt in y:
            self.validation_step_gts.append(round(gt.tolist()[0])) 
        self.validation_losses.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        acc = self.accuracy(torch.as_tensor(self.validation_step_outputs), torch.as_tensor(self.validation_step_gts))
        loss = sum(validation_loss for validation_loss in self.validation_losses) / len(self.validation_losses)
        self.validation_step_outputs.clear()
        self.validation_step_gts.clear()
        self.validation_losses.clear()
        self.log("val_loss", loss) 
        self.log("val_acc", acc) 
        if self.sweep:
            wandb.log({"val_loss": loss, "val_acc":acc}, commit=True) 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader


class SiameseDeciderOld(pl.LightningModule):
    def __init__(self, save_file=None, annotations=None, prefix=None, img_file=None, n=None, split=None, lr=None,
                 batch_size=4, weight_decay=0, same_border=False):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.back = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.img_file = img_file
        self.n = n
        self.split = split
        self.same_border = same_border
        if img_file is not None:
            t, v = get_train_and_val_loader(img_file, n, split, batch_size=batch_size, same_border=same_border)
        else:
            t, v = None, None
        self.trainloader = t
        self.valloader = v
        self.save_file = save_file
        self.annotations = annotations
        self.prefix = prefix

    def forward(self, x1, x2):
        z1 = self.back(x1)
        z2 = self.back(x2)
        # concat features into each other for one batch with double the features 
        z = rearrange([z1, z2], 'n b c h w -> b (n c h w)')
        return self.head(z)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat = self.forward(x1, x2)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False) 
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def train_dataloader(self):
        return self.trainloader 

    def val_dataloader(self):
        return self.valloader 
