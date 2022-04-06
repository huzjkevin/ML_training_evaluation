from torch.utils.data import DataLoader


def get_train_dataloader(cfg):
    from .arcface_dataset import MXFaceDataset
    dataset = MXFaceDataset(cfg["dataset"]["data_dir"])

    return DataLoader(
        dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        pin_memory=True,
        num_workers=cfg["dataloader"]["num_workers"],
        shuffle=False,
        # collate_fn=dataset.collate_batch,
    )

def get_test_dataloader(cfg):
    from lfw_dataset import LFWDataset
    dataset = LFWDataset(cfg["dataset"])

    return DataLoader(
        dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        pin_memory=True,
        num_workers=cfg["dataloader"]["num_workers"],
        shuffle=False,
        # collate_fn=dataset.collate_batch,
    )

if __name__ == "__main__":
    cfg = {
            "dataset":
                {"data_dir": "./data/LFW"}, 
            "dataloader":
                {"batch_size": 64, "num_workers": 8}
        }

    dataloader = get_train_dataloader(cfg)
    for idx, batch in enumerate(dataloader):
        input, target = batch["input"], batch["target"]
    print("Finish")