def get_model(cfg):
    if cfg["model"]["type"] == "cnn":
        from .cnn import CNN

        return CNN()
    
