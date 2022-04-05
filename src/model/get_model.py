def get_model(cfg):
    if cfg["model"]["type"] == "cnn":
        from .cnn import CNN

        return CNN()
    elif cfg["model"]["type"] == "resnet18":
        from .resnet import ResNetFace
        return ResNetFace(cfg["model"]["type"])

