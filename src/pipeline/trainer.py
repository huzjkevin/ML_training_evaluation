import signal
import tqdm

import torch
from torch.nn.utils import clip_grad_norm_


class Trainer(object):
    def __init__(self, logger, optimizer, cfg):
        self.logger = logger
        self.optim = optimizer
        self.epoch, self.step = 0, 0

        self.grad_norm_clip = cfg["grad_norm_clip"]
        self.ckpt_interval = cfg["ckpt_interval"]
        self.eval_interval = cfg["eval_interval"]
        self.max_epoch = cfg["epoch"]

        self.__sigterm = False
        signal.signal(signal.SIGINT, self.sigterm_cb)
        signal.signal(signal.SIGTERM, self.sigterm_cb)

    # hello world method for general demonstration
    def hw_evaluate(self, model, eval_loader, tb_prefix=None):
        model.eval()
        correct = 0
        total = 0
        pbar = tqdm.tqdm(total=len(eval_loader), leave=False, desc="eval")

        with torch.no_grad():
            for idx, batch in enumerate(eval_loader):
                if self.__sigterm:
                    pbar.close()
                    return 1

                input, target = batch
                #Move data to GPU
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # calculate outputs by running images through the network
                output = model(input)
                
                # the class with the highest energy is what we choose as prediction
                _, pred = torch.max(output.data, 1)
                total += target.size(0)
                correct += (pred == target).sum().item()

                pbar.update()
                
        pbar.close()
        self.logger.log_info(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def evaluate(self, model, eval_loader, tb_prefix):
        model.eval()
        tb_dict_list = []
        eval_dict_list = []
        pbar = tqdm.tqdm(total=len(eval_loader), leave=False, desc="eval")

        for b_idx, batch in enumerate(eval_loader):
            if self.__sigterm:
                pbar.close()
                return 1

            with torch.no_grad():
                tb_dict, eval_dict, fig_list_dict = model.model_eval_fn(model, batch)
                tb_dict_list.append(tb_dict)
                eval_dict_list.append(eval_dict)

                # save images
                for key, fig_list in fig_list_dict.items():
                    for f_idx, (fig, ax) in enumerate(fig_list):
                        key_str = (
                            f"{tb_prefix}_{key}_"
                            f"e{self.epoch}s{self.step}_"
                            f"b{b_idx}f{f_idx}"
                        )
                        # self.logger.add_fig(key_str, fig, self.step)
                        # self.logger.save_fig(fig, f"{key_str}.png", close_fig=True)

                pbar.update()

        tb_dict, epoch_dict = model.model_eval_collate_fn(tb_dict_list, eval_dict_list)

        for key, val in tb_dict.items():
            self.logger.add_scalar(f"{tb_prefix}_{key}", val, self.step)

        self.logger.save_dict(f"{tb_prefix}_e{self.epoch}s{self.step}", epoch_dict)
        pbar.close()

        return 0

    def train(self, model, train_loader, eval_loader=None):
        for self.epoch in tqdm.trange(0, self.max_epoch, desc="epochs"):
            if self.__sigterm:
                self.logger.save_sigterm_ckpt(
                    model, self.optim, self.epoch, self.step,
                )
                return 1

            self.train_epoch(model, train_loader)

            if not self.__sigterm:
                if self.is_ckpt_epoch():
                    self.logger.save_ckpt(
                        f"ckpt_e{self.epoch}.pth",
                        model,
                        self.optim,
                        self.epoch,
                        self.step,
                    )

                if eval_loader is not None and self.is_evaluation_epoch():
                    self.evaluate(model, eval_loader, tb_prefix="VAL")

            self.logger.flush()

        return 0

    def is_ckpt_epoch(self):
        return self.epoch % self.ckpt_interval == 0 or self.epoch == self.max_epoch

    def is_evaluation_epoch(self):
        return self.epoch % self.eval_interval == 0 or self.epoch == self.max_epoch

    def sigterm_cb(self, signum, frame):
        self.__sigterm = True
        self.logger.log_info(f"Received signal {signum} at frame {frame}.")

    def train_batch(self, model, batch, ratio):
        """Train one batch. `ratio` in between [0, 1) is the progress of training
        current epoch. It is used by the scheduler to update learning rate.
        """
        model.train()
        self.optim.zero_grad()
        self.optim.set_lr(self.epoch + ratio)

        loss, tb_dict, _ = model.model_fn(model, batch)
        loss.backward()

        if self.grad_norm_clip > 0:
            clip_grad_norm_(model.parameters(), self.grad_norm_clip)

        self.optim.step()

        self.logger.add_scalar("TRAIN_lr", self.optim.get_lr(), self.step)
        self.logger.add_scalar("TRAIN_loss", loss, self.step)
        self.logger.add_scalar("TRAIN_epoch", self.epoch + ratio, self.step)
        for key, val in tb_dict.items():
            self.logger.add_scalar(f"TRAIN_{key}", val, self.step)

        return loss.item()

    def train_epoch(self, model, train_loader):
        pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc="train")
        for idx, batch in enumerate(train_loader):
            if self.__sigterm:
                pbar.close()
                return

            loss = self.train_batch(model, batch, ratio=(idx / len(train_loader)))
            self.step += 1
            pbar.set_postfix({"total_it": self.step, "loss": loss})
            pbar.update()

        self.epoch = self.epoch + 1
        pbar.close()
