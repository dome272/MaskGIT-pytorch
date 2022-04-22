from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam


class WarmupLinearLRSchedule:
    """
    Implements Warmup learning rate schedule until 'warmup_steps', going from 'init_lr' to 'peak_lr' for multiple optimizers.
    """
    def __init__(self, optimizer, init_lr, peak_lr, end_lr, warmup_epochs, epochs=100, current_step=0):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.optimizer = optimizer
        self.warmup_rate = (peak_lr - init_lr) / warmup_epochs
        self.decay_rate = (end_lr - peak_lr) / (epochs - warmup_epochs)
        self.update_steps = current_step
        self.lr = init_lr
        self.warmup_steps = warmup_epochs
        self.epochs = epochs
        if current_step > 0:
            self.lr = self.peak_lr + self.decay_rate * (current_step - 1 - warmup_epochs)

    def set_lr(self, lr):
        print(f"Setting lr: {lr}")
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def step(self):
        if self.update_steps <= self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
        # elif self.warmup_steps < self.update_steps <= self.epochs:
        else:
            lr = max(0., self.lr + self.decay_rate)
        self.set_lr(lr)
        self.lr = lr
        self.update_steps += 1
        return self.lr


if __name__ == '__main__':
    m = nn.Linear(10, 10)
    opt = Adam(m.parameters(), lr=1e-4)
    s = WarmupLinearLRSchedule(opt, 1e-6, 1e-4, 0., 2)
    lrs = []
    for i in range(101):
        s.step()
        lrs.append(s.lr)
        print(s.lr)

    m = nn.Linear(10, 10)
    opt = Adam(m.parameters(), lr=1e-4)
    s = WarmupLinearLRSchedule(opt, 1e-6, 1e-4, 0., 2, current_step=50)
    lrs_s = []
    for i in range(50, 101):
        s.step()
        lrs_s.append(s.lr)
        print(s.lr)

    plt.plot(lrs)
    plt.plot(range(50, 101), lrs_s)
    plt.show()
