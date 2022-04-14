def get_model(cfg):
    if cfg["model"]["type"] == "cnn":
        from .cnn import CNN

        return CNN()
    elif cfg["model"]["type"] in  ["resnet18", "resnet50"]:
        from .resnet import ResNetFace
        return ResNetFace(cfg)

