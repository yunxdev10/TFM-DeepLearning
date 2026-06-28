import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)

ARCHITECTURE_ALIASES = {
    "resnet": "resnet50",
    "resnet-50": "resnet50",
    "resnet50": "resnet50",
    "densenet": "densenet121",
    "densenet-121": "densenet121",
    "densenet121": "densenet121",
    "efficientnet": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnetv2": "efficientnet_v2_s",
    "efficientnet-v2": "efficientnet_v2_s",
    "efficientnet-v2-s": "efficientnet_v2_s",
    "efficientnet_v2": "efficientnet_v2_s",
    "efficientnet_v2_s": "efficientnet_v2_s",
    "convnext": "convnext_tiny",
    "convnext-tiny": "convnext_tiny",
    "convnext_tiny": "convnext_tiny",
}


class CovidClassifier(nn.Module):
    """
    Wrapper for classification architectures (ResNet-50, DenseNet-121, EfficientNet-B0).
    Allows modifying input channels (e.g., 1 for CT) and output classes.
    """
    def __init__(self, architecture_name: str, num_classes: int, in_channels: int = 3, pretrained: bool = True):
        super(CovidClassifier, self).__init__()
        
        self.architecture_name = ARCHITECTURE_ALIASES.get(architecture_name.lower(), architecture_name.lower())
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

        elif self.architecture_name == 'efficientnet_v2_s':
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.model = efficientnet_v2_s(weights=weights)

            if self.in_channels != 3:
                self.model.features[0][0] = self._adapt_first_conv(self.model.features[0][0], in_channels)

            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        elif self.architecture_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.model = convnext_tiny(weights=weights)

            if self.in_channels != 3:
                self.model.features[0][0] = self._adapt_first_conv(self.model.features[0][0], in_channels)

            num_ftrs = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(num_ftrs, num_classes)
            
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
        
        # If we are adapting from 3 channels to 1 (e.g., CT scans), we average the weights
        # across the channel dimension to preserve the learned feature extraction.
        if conv_layer.in_channels == 3 and in_channels == 1:
            with torch.no_grad():
                new_conv.weight[:] = torch.mean(conv_layer.weight, dim=1, keepdim=True)
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
        elif self.architecture_name in {'efficientnet_b0', 'efficientnet_v2_s'}:
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif self.architecture_name == 'convnext_tiny':
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def classifier_parameters(self):
        """Returns trainable parameters for the final classification head."""
        if self.architecture_name == 'resnet50':
            return self.model.fc.parameters()
        if self.architecture_name == 'densenet121':
            return self.model.classifier.parameters()
        if self.architecture_name in {'efficientnet_b0', 'efficientnet_v2_s'}:
            return self.model.classifier.parameters()
        if self.architecture_name == 'convnext_tiny':
            return self.model.classifier.parameters()
        raise ValueError(f"Architecture {self.architecture_name} is not supported.")
