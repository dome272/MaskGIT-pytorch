import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images
from lr_schedule import WarmupLinearLRSchedule
from torch.utils.tensorboard import SummaryWriter


class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()
        self.lr_schedule = WarmupLinearLRSchedule(
            optimizer=self.optim,
            init_lr=1e-6,
            peak_lr=args.learning_rate,
            end_lr=0.,
            warmup_epochs=10,
            epochs=args.epochs,
            current_step=args.start_from_epoch
        )

        if args.start_from_epoch > 1:
            self.model.load_checkpoint(args.start_from_epoch)
            print(f"Loaded Transformer from epoch {args.start_from_epoch}.")
        if args.run_name:
            self.logger = SummaryWriter(f"./runs/{args.run_name}")
        else:
            self.logger = SummaryWriter()
        self.train(args)

    def train(self, args):
        train_dataset = load_data(args)
        len_train_dataset = len(train_dataset)
        step = args.start_from_epoch * len_train_dataset
        for epoch in range(args.start_from_epoch+1, args.epochs+1):
            print(f"Epoch {epoch}:")
            with tqdm(range(len(train_dataset))) as pbar:
                self.lr_schedule.step()
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    logits, target = self.model(imgs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                    loss.backward()
                    if step % args.accum_grad == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    step += 1
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
                    self.logger.add_scalar("Cross Entropy Loss", np.round(loss.cpu().detach().numpy().item(), 4), (epoch * len_train_dataset) + i)
            try:
                log, sampled_imgs = self.model.log_images(imgs[0:1])
                vutils.save_image(sampled_imgs.add(1).mul(0.5), os.path.join("results", f"{epoch}.jpg"), nrow=4)
                plot_images(log)
            except:
                pass
            if epoch % args.ckpt_interval == 0:
                torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_epoch_{epoch}.pt"))
            torch.save(self.model.state_dict(), os.path.join("checkpoints", "transformer_current.pt"))

    def configure_optimizers(self):
        # decay, no_decay = set(), set()
        # whitelist_weight_modules = (nn.Linear,)
        # blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        # for mn, m in self.model.transformer.named_modules():
        #     for pn, p in m.named_parameters():
        #         fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
        #
        #         if pn.endswith('bias'):
        #             no_decay.add(fpn)
        #
        #         elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
        #             decay.add(fpn)
        #
        #         elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
        #             no_decay.add(fpn)
        #
        # # no_decay.add('pos_emb')
        #
        # param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}
        #
        # optim_groups = [
        #     {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 4.5e-2},
        #     {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        # ]
        optimizer = torch.optim.Adam(self.model.transformer.parameters(), lr=1e-4, betas=(0.9, 0.96), weight_decay=4.5e-2)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=8192, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--start-from-epoch', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--sos-token', type=int, default=1025, help='Start of Sentence token.')

    parser.add_argument('--n-layers', type=int, default=24, help='Number of layers of transformer.')
    parser.add_argument('--dim', type=int, default=768, help='Dimension of transformer.')
    parser.add_argument('--hidden-dim', type=int, default=3072, help='Dimension of transformer.')
    parser.add_argument('--num-image-tokens', type=int, default=256, help='Number of image tokens.')

    args = parser.parse_args()
    args.run_name = "<name>"
    args.dataset_path = r"C:\Users\dome\datasets\landscape"
    args.checkpoint_path = r".\checkpoints"
    args.n_layers = 24
    args.dim = 768
    args.hidden_dim = 3072
    args.batch_size = 4
    args.accum_grad = 25
    args.epochs = 1000

    args.start_from_epoch = 0

    args.num_codebook_vectors = 1024
    args.num_image_tokens = 256


    train_transformer = TrainTransformer(args)
    
