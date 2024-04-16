## Training a custom resnet architecture for CIFAR-10 with 90% validation accuracy in 24 epochs. 

Training a custom resnet convolutional neural network on CIFAR10 dataset. Using one cycle LR and cutouts.

## Model:

    1 - Dataset
       CIFAR10 dataset loaded and augmented
       train_loader,test_loader = dataloader.load_dataset()

    2 - Model:
        _model = Net_10.to(device)
        summary(_model, input_size=(3, 32, 32))
    3 - Lerning Rate
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(_model.parameters(), lr=0.01, weight_decay=1e-4)
        lr_finder = LRFinder(_model, optimizer, criterion, device="cuda")
        lr_finder.range_test(train_loader, end_lr=10, num_iter=200,step_mode="exp")
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset() # to reset the model and optimizer to their initial state
    4 - Train the model using one-cycle LR
        EPOCHS = 24
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,  max_lr= 1.63E-02,
                                                  pct_start = 5/EPOCHS, div_factor = 100,
                                                  epochs=24, steps_per_epoch=len(train_loader),
                                                  three_phase=False,verbose = True)


        for epoch in range(EPOCHS):
            print("EPOCH:", epoch)
            train(_model, device, train_loader, optimizer, epoch, criterion, scheduler)
            test(_model, device, test_loader,criterion)
