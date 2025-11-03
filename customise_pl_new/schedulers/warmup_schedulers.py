"""Schedulers wit warm-up policy.
"""
from torch.optim import Optimizer
import pytorch_lightning as pl


class LearningRateScheduler(object):
    """
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, optimizer, init_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr

    def set_lr(self, optimizer, factor):
        for idx, g in enumerate(optimizer.param_groups):
            g['lr'] = self.init_lr[idx] * factor

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class PolyLRScheduler(LearningRateScheduler):
    """
    Transformer Learning Rate Scheduler proposed in "Attention Is All You Need"
    @inproceedings{10.5555/3295222.3295349,
    author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, \L{}ukasz and Polosukhin, Illia},
    title = {Attention is All You Need},
    year = {2017},
    isbn = {9781510860964},
    publisher = {Curran Associates Inc.},
    address = {Red Hook, NY, USA},
    abstract = {The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.},
    booktitle = {Proceedings of the 31st International Conference on Neural Information Processing Systems},
    pages = {6000–6010},
    numpages = {11},
    location = {Long Beach, California, USA},
    series = {NIPS'17}
    }

    Args:
        optimizer (Optimizer): Optimizer.
        final_lr (float): Final learning rate.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates
    """

    def __init__(
            self,
            optimizer: Optimizer,
            power: float,
            num_epochs: int,
            final_lr: float,
            warmup_steps: int,
            start_epoch=0
    ) -> None:
        super(PolyLRScheduler, self).__init__(optimizer, [pg["lr"] for pg in optimizer.param_groups])
        self.stage = 0
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        self.final_lr = final_lr  # final lr after decay
        self.warmup_steps = warmup_steps
        self.update_steps = 1
        self.power = power
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch

        self.step(0, init=True)  # start from 1/warmup_steps

    def _decide_stage(self):
        if self.update_steps <= self.warmup_steps:
            return 0
        else:
            return 1

    def step(self, epoch, init=False):
        # print(f"warup epoch: {epoch}")
        if epoch >= self.start_epoch or init:
            if self.stage == 0:
                warmup_rate = self.update_steps / self.warmup_steps
                self.set_lr(self.optimizer, warmup_rate)
                self.update_steps += 1
                if self.update_steps == self.warmup_steps:
                    self.start_epoch = epoch
                self.stage = self._decide_stage()
            elif self.stage == 1:  # start to decay with epoch
                decay_rate = (1 - (epoch - self.start_epoch) / (self.num_epochs - self.start_epoch)) ** self.power
                self.set_lr(self.optimizer, decay_rate)
            else:
                raise ValueError("Undefined stage")

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
