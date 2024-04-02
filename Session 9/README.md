# Classification of CIFAR-10 dataset
## Objective: 
   - Learn about advanced convolution and augmentation

     
Some constraints:
 - Achieve 85% accuracy
 - Total RF must be more than 44
 - One of the layers must use Depthwise Separable Convolution and one of the layers must use Dilated Convolution
 - Use GAP (compulsory)
 - Use albumentation library and apply:
        - horizontal flip
        - shiftScaleRotate
        - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)

- Achieve 85% accuracy
- Total Params to be less than 200k

## My Network Summary

- Batch Size: 512

- Total Parameters: 134,722

- Image Augmentation applied
   - horizontal flip
   - Shift Scale Rotate
   - Coarse Dropout
- Dropout as regularization with probability or 0.02

85% as target accuracy obtained in all cases.
