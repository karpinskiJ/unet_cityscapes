import torch.nn as nn
from unet_cityscapes.network.modules import DoubleConv2D, OutLayer
import torch
import logging
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from torchmetrics import Accuracy
import wandb


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.left_conv_1 = DoubleConv2D(in_channels, 64)
        self.left_pool_1 = nn.MaxPool2d(2)
        self.left_conv_2 = DoubleConv2D(64, 128)
        self.left_pool_2 = nn.MaxPool2d(2)
        self.left_conv_3 = DoubleConv2D(128, 256)
        self.left_pool_3 = nn.MaxPool2d(2)
        self.left_conv_4 = DoubleConv2D(256, 512)
        self.left_pool_4 = nn.MaxPool2d(2)
        self.bottom_conv = DoubleConv2D(512, 1024)
        self.right_up_4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.right_conv_4 = DoubleConv2D(1024, 512)
        self.right_up_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.right_conv_3 = DoubleConv2D(512, 256)
        self.right_up_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.right_conv_2 = DoubleConv2D(256, 128)
        self.right_up_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.right_conv_1 = DoubleConv2D(128, 64)
        self.out = OutLayer(64, out_channels)



    def forward(self, input):
        left_conv_1 = self.left_conv_1(input)
        left_pool_1 = self.left_pool_1(left_conv_1)
        left_conv_2 = self.left_conv_2(left_pool_1)
        left_pool_2 = self.left_pool_2(left_conv_2)
        left_conv_3 = self.left_conv_3(left_pool_2)
        left_pool_3 = self.left_pool_3(left_conv_3)
        left_conv_4 = self.left_conv_4(left_pool_3)
        left_pool_4 = self.left_pool_4(left_conv_4)
        bottom_conv = self.bottom_conv(left_pool_4)
        right_up_4 = self.right_up_4(bottom_conv)
        cat_4 = torch.cat((right_up_4, left_conv_4), dim=1)
        right_conv_4 = self.right_conv_4(cat_4)
        right_up_3 = self.right_up_3(right_conv_4)
        cat_3 = torch.cat((right_up_3, left_conv_3), dim=1)
        right_conv_3 = self.right_conv_3(cat_3)
        right_up_2 = self.right_up_2(right_conv_3)
        cat_2 = torch.cat((right_up_2, left_conv_2), dim=1)
        right_conv_2 = self.right_conv_2(cat_2)
        right_up_1 = self.right_up_1(right_conv_2)
        cat_1 = torch.cat((right_up_1, left_conv_1), dim=1)
        right_conv_1 = self.right_conv_1(cat_1)
        return self.out(right_conv_1)
    
    def compile(self,
                learning_rate,
                loss_fn,
                training_loader,
                validation_loader,
                wandb_run=None):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            self.loss_fn = loss_fn
            self.training_loader = training_loader
            self.validation_loader = validation_loader
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.iou_metric = JaccardIndex(task="multiclass", num_classes=self.out_channels).to(self.device)
            self.iou_per_class_metric = JaccardIndex(task="multiclass", num_classes=self.out_channels, average=None).to(self.device)
            self.accuracy_metric = Accuracy(task="multiclass", num_classes=self.out_channels).to(self.device)
            self.accuracy_metric_per_class = Accuracy(task="multiclass", num_classes=self.out_channels, average=None).to(self.device)
            self.wandb_run = wandb_run
            print("Model compiled with learning rate: {},device: {} and loss function: {}".format(learning_rate,self.device, loss_fn))
    
    def eval(self,input):
        self.train(False)
        with torch.no_grad():
            prediction = self.forward(input)
            output = F.softmax(prediction, dim=1)
            return torch.argmax(output, dim=1)
        


    def __train_one_epoch(self):
        train_loss = 0
        for i, data in enumerate(self.training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss/len(self.training_loader)
    
    def _calc_metrics(self, outputs, labels):
        outputs = torch.argmax(F.softmax(outputs, dim=1),dim=1)
        return self.iou_metric(outputs, labels), self.accuracy_metric(outputs, labels)
    def _calc_metrics_per_class(self, outputs, labels):
        outputs = torch.argmax(F.softmax(outputs, dim=1),dim=1)
        return self.iou_per_class_metric(outputs, labels), self.accuracy_metric_per_class(outputs, labels)
        
    def __eval_one_epoch(self):
        validation_loss = 0
        validation_iou = 0
        validation_accuracy = 0
        with torch.no_grad():
            for i, data in enumerate(self.validation_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, labels)
                
                validation_loss += loss.item()
                validation_iou, validation_accuracy += self._calc_metrics(outputs, labels)
            
            val_metrics = [validation_loss,validation_iou,validation_accuracy]


        return  [metric/ len(self.validation_loader) for metric in val_metrics]
    
    def __eval_model_per_class(self):
         total_validaiton_iou_per_class = torch.tensor([0.0]*self.out_channels,dtype=torch.float32).to(self.device)
         total_validation_accuracy_per_class = torch.tensor([0.0]*self.out_channels,dtype=torch.float32).to(self.device)
         with torch.no_grad():
            for i, data in enumerate(self.validation_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
           
                validation_iou_per_class, validation_accuracy_per_class = self._calc_metrics_per_class(outputs, labels)
                total_validaiton_iou_per_class += validation_iou_per_class
                total_validation_accuracy_per_class += validation_accuracy_per_class

            total_validaiton_iou_per_class = (total_validaiton_iou_per_class/len(self.validation_loader)).cpu().numpy().tolist()
            total_validation_accuracy_per_class = (total_validation_accuracy_per_class/len(self.validation_loader)).cpu().numpy().tolist()

            total_validation_accuracy_per_class = dict(zip(range(0,self.out_channels), total_validation_accuracy_per_class ))
            toal_validation_iou_per_class =  dict(zip(range(0,self.out_channels), total_validaiton_iou_per_class ) )
            return  toal_validation_iou_per_class, total_validation_accuracy_per_class
    
    def __log_wandb(self, metrics):
        if self.wandb_run:
            self.wandb_run.log(metrics)

    
    def fit(self,
            epochs: int):
        print("Training started with {} epochs.".format(epochs))
        self.train(True)
        for epoch in range(epochs):
            avg_train_loss = self.__train_one_epoch()
            train_metrics = {
                "train_loss": avg_train_loss,
                "epoch": epoch
            }
            avg_validation_loss, avg_validation_iou, avg_validation_accuracy = self.__eval_one_epoch()

            val_metrics = {
                "val_loss": avg_validation_loss,
                "val_mIoU": avg_validation_iou,
                "val_mAcc": avg_validation_accuracy,
                "epoch": epoch
            }
            self.__log_wandb({**train_metrics, **val_metrics})
            print("Epoch: {} Train Loss: {} Validation Loss: {}, Validation mIoU: {}, Validation mAcurracy: {}".format(epoch, avg_train_loss, avg_validation_loss,avg_validation_iou,avg_validation_accuracy))
            if epoch == epochs-1:
                iou_per_class, accuracy_per_class = self.__eval_model_per_class()
                print("Validation mIoU per class:\n {}".format(iou_per_class))
                print("Validation mAccuracy per class:\n  {}".format(accuracy_per_class))
                self.__log_wandb({"iou_per_class": iou_per_class, "accuracy_per_class": accuracy_per_class})
            
    
    def save(self,path,model_name):
        torch.save(self.state_dict(), path)
        print("Model saved at {}".format(path))
        if self.wandb_run:
            artifact = wandb.Artifact(model_name, type='model')
            artifact.add_file(path)
            self.wandb_run.log_artifact(artifact)
            self.wandb_run.finish()
            print("Model saved in wandb as artifact")
    
            
        






