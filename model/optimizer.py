import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, configs, current_step):
        self._optimizer = torch.optim.AdamW(
            model.parameters(),
            betas=configs["optimizer"]["betas"],
            eps=configs["optimizer"]["eps"],
            weight_decay=configs["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = configs["optimizer"]["warm_up_step"]
        self.anneal_steps = configs["optimizer"]["anneal_steps"]
        self.anneal_rate = configs["optimizer"]["anneal_rate"]
        self.current_step = current_step

        self.current_step -= configs["step"]["total_step_aux"]

        self.init_lr = configs["optimizer"]["init_lr"]

    def step_and_update_lr(self):
        lr = self._update_learning_rate()
        self._optimizer.step()
        return lr

    def zero_grad(self):
        # print("self.init_lr:", self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = self.init_lr
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr
