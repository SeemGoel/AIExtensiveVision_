# Cifar10 Classification - PyTorch Lightning -

## Model
```
 Name        | Type              | Params
--------------------------------------------------
0 | model       | ResNet            | 11.2 M
1 | convblock1  | Sequential        | 1.9 K 
2 | convblock2  | Sequential        | 74.0 K
3 | R1          | Sequential        | 295 K 
4 | convblock3  | Sequential        | 295 K 
5 | convblock4  | Sequential        | 1.2 M 
6 | R2          | Sequential        | 4.7 M 
7 | maxpool     | MaxPool2d         | 0     
8 | global_pool | AdaptiveAvgPool2d | 0     
9 | fc          | Linear            | 5.1 K 
--------------------------------------------------
17.7 M    Trainable params
0         Non-trainable params
17.7 M    Total params
70.988    Total estimated model params size (MB)


```

### Image Augmentation
```
train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)
```

### OneCycle LR 

```
def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,

            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.01,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
```

## Misclassified Images

![image](https://github.com/SeemGoel/AIExtensiveVision_/assets/59606392/f966811f-2769-4977-9853-ea046f927f3b)


## Gradio Images:
![image](https://github.com/SeemGoel/AIExtensiveVision_/assets/59606392/2bba0174-3811-4990-8455-d14426c7622b)



## Loss & LR:
![image](https://github.com/SeemGoel/AIExtensiveVision_/assets/59606392/ba83dff3-d8e6-4d23-b823-6b4b6b684bcd)





