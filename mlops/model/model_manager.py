# This will load a specific model :
"""
| Era   | Model                            | Key Idea                            |
| ----- | -------------------------------- | ----------------------------------- |
| 1998  | LeNet-5                          | First CNN, simple digit recognition |
| 2012  | AlexNet                          | Deep CNN + ReLU + GPU               |
| 2014  | VGGNet                           | Deep stack of 3x3 convs             |
| 2014  | GoogLeNet                        | Inception modules                   |
| 2015  | ResNet                           | Residual connections                |
| 2017  | DenseNet                         | Dense connections                   |
| 2017  | MobileNet / Xception             | Lightweight / separable convs       |
| 2019  | EfficientNet                     | Compound scaling                    |
| 2020+ | ConvNeXt, RegNet, EfficientNetV2 | Modern, efficient CNNs              |
"""
from mlops.model.convolutional.LeNet import LeNet

# To fill later on 