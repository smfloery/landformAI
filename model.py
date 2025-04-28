import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics.functional import f1_score
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
from datetime import datetime
from loader import GMKDataset
from torchvision import transforms
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

class GMM:

    def __init__(self, params):
        self.params = params        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = self.params["training"]["epochs"]
        self.batch_size = self.params["training"]["batch_size"]
        self.out_dir = self.params["data"]["out"]
        
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        else:
            raise FileExistsError("%s already exists." % (self.out_dir))
        
        print("....initializing model.")
        if self.params["model"]["name"] == "DeepLabV3Plus":
            model = smp.DeepLabV3Plus(
                encoder_name = self.params["model"]["encoder"],
                encoder_weights = self.params["model"]["weights"],
                in_channels = self.params["model"]["channels"],
                classes = self.params["model"]["classes"]
            )
        elif self.params["model"]["name"] == "Unet++":
            model = smp.UnetPlusPlus(
                encoder_name=self.params["model"]["encoder"],
                encoder_weights=self.params["model"]["weights"],
                in_channels=self.params["model"]["channels"],
                classes=self.params["model"]["classes"]
            )
        elif self.params["model"]["name"] == "Segformer":
            model = smp.Segformer(
                encoder_name=self.params["model"]["encoder"],
                encoder_weights=self.params["model"]["weights"],
                in_channels=self.params["model"]["channels"],
                classes=self.params["model"]["classes"]
            )
        elif self.params["model"]["name"] == "MAnet":
            model = smp.MAnet(
                encoder_name=self.params["model"]["encoder"],
                encoder_weights=self.params["model"]["weights"],
                in_channels=self.params["model"]["channels"],
                classes=self.params["model"]["classes"]
            )
        else:
            raise ValueError("%s not supported." % (self.params["model"]["name"]))

        
        if self.params["training"]["optimizer"]["type"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), 
                                         lr=self.params["training"]["optimizer"]["lr"])
        elif self.params["training"]["optimizer"]["type"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), 
                                        lr=self.params["training"]["optimizer"]["lr"],
                                        momentum=0.9)
        else:
            raise ValueError("%s not supported." % (self.params["training"]["optimizer"]["type"]))
        
        if self.params["training"]["loss"] == "dice":
            loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        elif self.params["training"]["loss"] == "focal":
            loss_fn = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
        elif self.params["training"]["loss"] == "lovasz":
            loss_fn = smp.losses.LovaszLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        else:
             raise ValueError("%s not supported." % (self.params["training"]["loss"]))

        
        if self.params["model"]["weights"] is not None:
            preprocessing_params = smp.encoders.get_preprocessing_params(self.params["model"]["encoder"], self.params["model"]["weights"])
            pre_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=preprocessing_params["mean"],
                                     std=preprocessing_params["std"])
                ])
        else:
            # pre_transform = None   
            pre_transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5],  #https://medium.com/@piyushkashyap045/image-normalization-in-pytorch-from-tensor-conversion-to-scaling-3951b6337bc8
                #                      std=[0.5, 0.5, 0.5])
                ])
        
        print("...preparing training.")
        train_data = GMKDataset(self.params["data"]["train"], 
                                device=self.device,
                                classes=self.params["model"]["classes"], 
                                transform=pre_transform)
        
        valid_data = GMKDataset(self.params["data"]["valid"],
                                device=self.device, 
                                classes=self.params["model"]["classes"], 
                                transform=pre_transform)
        
        self.train_loader = DataLoader(train_data, batch_size=self.params["training"]["batch_size"], shuffle=True, drop_last=True) #drop last incomplete batch size; 
        self.valid_loader = DataLoader(valid_data, batch_size=self.params["training"]["batch_size"], shuffle=False) #validation set must not be shuffled; 
        
        if self.params["training"]["scheduler"] is not None:
            scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                                 milestones=self.params["training"]["scheduler"]["steps"], 
                                                 gamma=self.params["training"]["scheduler"]["factor"])    
        
        model.to(self.device)
        self.model = model        
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        
    def train_loop(self):
        
        self.log_path = os.path.join(self.out_dir, "train.log")
        log_file = open(self.log_path, "w")
        log_file.write("epoch;time;train_loss;valid_loss;valid_iou_micro;valid_f1_micro\n")
        
        chkpnts_dir = os.path.join(self.out_dir, "chkpnts")
        os.makedirs(chkpnts_dir, exist_ok=True)
        
        with open(os.path.join(self.out_dir, 'config.yaml'), 'w') as yaml_file:
            yaml.dump(self.params, yaml_file)
        
        for epoch in range(self.epochs):
            print('EPOCH {}:'.format(epoch+1))
        
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            train_loss = 0
            
            for batch in tqdm(self.train_loader, desc="Training"):
                
                # Every batch instance is an input + label pair
                inputs = batch[0].to(self.device)
                labels = torch.squeeze(batch[1]).to(self.device)
                    
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()

                # Make predictions for this batch
                outputs = torch.squeeze(self.model(inputs))
                
                # Compute the loss and its gradients
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                # # to check if valid parameters; if none something is off
                # for param in model.parameters():
                #     print(param.grad)
                
                # Adjust learning weights
                self.optimizer.step()

                train_loss += loss.item()
            
            # scheduler.step()
            avg_train_loss = train_loss / len(self.train_loader)
            
            self.scheduler.step()
            
            # Disable gradient computation and reduce memory consumption.
            self.model.eval()
            val_los = 0
            
            tp, fp, fn, tn = 0, 0, 0, 0
            
            with torch.no_grad():
                for batch in tqdm(self.valid_loader, desc="Evaluate"):
                    val_inputs = batch[0].to(self.device)               
                    val_labels_true = batch[1].to(self.device)
                        
                    val_outputs = self.model(val_inputs)
                    assert (val_outputs.shape[1] ==self.params["model"]["classes"])  # [batch_size, number_of_classes, H, W]
                    
                    #loss_fn requires "raw" model outputs; accrordingly, we apply the log_softmax afterwards
                    curr_val_loss = self.loss_fn(val_outputs, val_labels_true)
                    val_los += curr_val_loss
                    
                    val_probs = val_outputs.log_softmax(dim=1).exp() #Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on extreme values 0 and 1
                    
                    val_labels_pred = torch.argmax(val_probs, dim=1, keepdim=True)
                    val_labels_pred = val_labels_pred.long()
                    
                    batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(val_labels_pred, val_labels_true, mode='multiclass', num_classes = self.params["model"]["classes"])               
                    tp += batch_tp.sum().item() 
                    fp += batch_fp.sum().item()
                    fn += batch_fn.sum().item()
                    tn += batch_tn.sum().item()
                    
            epoch_iou_score = smp.metrics.iou_score(torch.tensor([tp]),
                                            torch.tensor([fp]),
                                            torch.tensor([fn]),
                                            torch.tensor([tn]),
                                            reduction="micro").item()
            
            epoch_f1_score = smp.metrics.f1_score(torch.tensor([tp]),
                                            torch.tensor([fp]),
                                            torch.tensor([fn]),
                                            torch.tensor([tn]),
                                            reduction="micro").item()
            
            avg_val_loss = val_los / len(self.valid_loader)                
        
            print('LOSS train: %.4f valid: %.4f - IOU: %.2f - F1: %.2f' % (avg_train_loss, avg_val_loss, epoch_iou_score*100, epoch_f1_score*100))
        
            log_file.write("%i;%s;%.4f;%.4f;%.2f;%.2f\n" % (epoch+1, datetime.now().strftime('%Y%m%d_%H%M%S'), avg_train_loss, avg_val_loss, epoch_iou_score*100, epoch_f1_score*100))
            log_file.flush()
            os.fsync(log_file)

            # Track best performance, and save the model's state
            model_path = os.path.join(chkpnts_dir, 'epoch_{}'.format(epoch+1))
            self.model.save_pretrained(model_path)
    
    def plot_train_val_loss(self):
        
        out_path = os.path.join(os.path.dirname(self.log_path), "train_val_loss.png")
        pd_train_log = pd.read_csv(self.log_path, sep=";")

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        ax1.plot(pd_train_log.epoch, pd_train_log.train_loss, label="Training")
        ax1.plot(pd_train_log.epoch, pd_train_log.valid_loss, label="Validation")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--")

        ax2.plot(pd_train_log.epoch, pd_train_log.valid_iou_micro, label="IOU")
        ax2.plot(pd_train_log.epoch, pd_train_log.valid_f1_micro, label="F1")
        ax2.set_title("Quality")
        ax2.legend()
        ax2.grid(True, linestyle="--")

        plt.savefig(out_path, dpi=300, bbox_inches="tight")