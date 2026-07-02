import torch.nn as nn
from torchvision import models
import timm

MODELS = {
    "resnet18": lambda pretrained: models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
    ),
    "resnet50": lambda pretrained: models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT if pretrained else None
    ),
    "resnet152": lambda pretrained: models.resnet152(
        weights=models.ResNet152_Weights.DEFAULT if pretrained else None
    ),
    "vgg16": lambda pretrained: models.vgg16(
        weights=models.VGG16_Weights.DEFAULT if pretrained else None
    ),
    "vgg19": lambda pretrained: models.vgg19(
        weights=models.VGG19_Weights.DEFAULT if pretrained else None
    ),
    "efficientnet_b5": lambda pretrained: models.efficientnet_b5(
        weights=models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
    ),
    "efficientnet_v2_m": lambda pretrained: models.efficientnet_v2_m(
        weights=models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
    ),
    "densenet121": lambda pretrained: models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
    ),
    "densenet169": lambda pretrained: models.densenet169(
        weights=models.DenseNet169_Weights.DEFAULT if pretrained else None
    ),
    "xception": lambda pretrained: timm.create_model("xception", pretrained=pretrained),
}

def get_model(config):
    pretrained = config.get("pretrained", True)
    try:
        model = MODELS[config["model_name"]](pretrained)
    except KeyError:
        raise ValueError(f"Unsupported model '{config['model_name']}'")

    if config["model_task"]=="binary":
        model = replace_head(model, 1)
    elif config["model_task"]=="regression":
        model = replace_head(model, 1, config["use_regression_sigmoid"])
    elif config["model_task"]=="multiclass":
        model = replace_head(model, 7) # 7 classes: <10, 10-12, 12-14, 14-16, 16-18, 18-21, 21+
    elif config["model_task"]=="stages":
        stage_method = config.get("stage_method", "H")
        if stage_method == "H":
            num_stages = 12
        elif stage_method == "K":
            num_stages = 7
        elif stage_method == "D":
            num_stages = 8
        model = replace_head(model, num_stages)
        
    return model

def make_head(in_feats, num_outputs, activation):
    layers = [nn.Linear(in_feats, num_outputs)]
    if activation == "sigmoid":
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]

def replace_head(model, num_outputs, activation=None):
    if hasattr(model, 'fc'):
        in_feats = model.fc.in_features
        model.fc = make_head(in_feats, num_outputs, activation)

    elif hasattr(model, 'classifier'):
        cls = model.classifier
        if isinstance(cls, nn.Sequential):
            in_feats = cls[-1].in_features
            cls[-1] = make_head(in_feats, num_outputs, activation)
        else:
            in_feats = cls.in_features
            model.classifier = make_head(in_feats, num_outputs, activation)
    else:
        raise ValueError(f"Don't know how to replace head for {type(model)}")

    return model
