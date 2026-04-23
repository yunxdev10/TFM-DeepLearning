import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

class CovidClassifier(nn.Module):
    """
    Wrapper for classification architectures (ResNet-50, DenseNet-121, EfficientNet-B0).
    Allows modifying input channels (e.g., 1 for CT) and output classes.
    """
    def __init__(self, architecture_name: str, num_classes: int, in_channels: int = 3, pretrained: bool = True):
        super(CovidClassifier, self).__init__()
        
        self.architecture_name = architecture_name.lower()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        if self.architecture_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = resnet50(weights=weights)
            
            # Adapt input channels if necessary
            if self.in_channels != 3:
                self.model.conv1 = self._adapt_first_conv(self.model.conv1, in_channels)
            
            # Modify classification head
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            
        elif self.architecture_name == 'densenet121':
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            self.model = densenet121(weights=weights)
            
            if self.in_channels != 3:
                self.model.features.conv0 = self._adapt_first_conv(self.model.features.conv0, in_channels)
                
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
            
        elif self.architecture_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = efficientnet_b0(weights=weights)
            
            if self.in_channels != 3:
                self.model.features[0][0] = self._adapt_first_conv(self.model.features[0][0], in_channels)
                
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        else:
            raise ValueError(f"Architecture {architecture_name} is not supported.")

    def _adapt_first_conv(self, conv_layer: nn.Conv2d, in_channels: int) -> nn.Conv2d:
        """
        Modifies the first convolutional layer to accept a different number of input channels,
        while maintaining the pretrained weights if originally 3 channels.
        """
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=(conv_layer.bias is not None)
        )
        
        # If we are adapting from 3 channels to 1 (e.g., CT scans), we can sum the weights
        # across the channel dimension to preserve the learned feature extraction.
        if conv_layer.in_channels == 3 and in_channels == 1:
            with torch.no_grad():
                new_conv.weight[:] = torch.sum(conv_layer.weight, dim=1, keepdim=True)
                if new_conv.bias is not None:
                    new_conv.bias[:] = conv_layer.bias
        
        return new_conv

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Freezes all layers except the final classification head."""
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze classification head
        if self.architecture_name == 'resnet50':
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif self.architecture_name == 'densenet121':
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif self.architecture_name == 'efficientnet_b0':
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
