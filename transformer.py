import torch
import torch.nn as nn
import torch.nn.functional as F
from bidirectional_transformer import BidirectionalTransformer
from vqgan import VQGAN
import numpy as np


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.sos_token = args.sos_token

        self.vqgan = self.load_vqgan(args)

        self.transformer = BidirectionalTransformer()

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    def forward(self, x):
        _, z_indices = self.encode_to_z(x)

        r = np.random.uniform()
        mask = torch.bernoulli(r * torch.ones(z_indices.shape, device=z_indices.device))
        mask = mask.round().to(dtype=torch.int64)
        masked_indices = torch.zeros_like(z_indices)
        a_indices = mask * z_indices + (1 - mask) * masked_indices

        target = z_indices

        logits = self.transformer(a_indices)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = 0
        return out

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r**2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    @torch.no_grad()
    def sample(self, condition=None, num=1, T=10, temperature=1.0, mode="cosine"):
        N = 256
        if condition is None:
            indices = torch.zeros((num, N), device="cuda", dtype=torch.int)
        else:
            indices = torch.hstack((condition, torch.zeros((condition.shape[0], N-condition.shape[1]), device="cuda", dtype=torch.int)))

        gamma = self.gamma(mode)

        for t in range(T+1):
            # define a mask for the indices which have already been sampled
            unmasked = torch.clamp(indices, 0, 1).type(torch.int)

            n = np.ceil(gamma(t/T) * N)
            logits = self.transformer(indices)

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            shape = probs.shape
            probs = probs.reshape(shape[0]*shape[1], shape[2])

            # sample token for EACH position
            sample_indices = torch.multinomial(probs, num_samples=1)

            # get confidence score for each sampled token
            probability_indices = torch.gather(probs, 1, sample_indices).reshape(shape[0], shape[1])

            # set all probability indices to 1 if the position has a 1 in unmasked (meaning that this index has already
            # been sampled before and we don't want to replace it
            probability_indices = torch.where(unmasked == 1, torch.ones_like(unmasked, dtype=torch.float), probability_indices)

            # create a mask where 1 indicates indices which we will keep and 0 will be resampled due to low confidence
            mask = torch.ceil(self.top_k_logits(probability_indices, int(N-n))).type(torch.int)

            # print(f"Number of non_zero elements in masked: {torch.count_nonzero(mask)}. Expected: {(N-n)*shape[0]}")

            sample_indices = sample_indices.reshape(shape[0], shape[1])

            # create the new indices where already established indices will also be replaced
            new_indices = sample_indices * mask
            new_indices = new_indices.type(indices.dtype)

            # and here we ignore the new sampled indices for which we already have an index
            indices = torch.where(unmasked == 1, indices, new_indices)

            # print(t)
            # print(indices)  # uncomment this to see how the indices are iteratively generated
        return indices

    @torch.no_grad()
    def log_images(self, x, mode="cosine"):
        log = dict()

        _, z_indices = self.encode_to_z(x)

        # create a "half" sample
        z_start_indices = z_indices[:, :z_indices.shape[1]//2]
        half_index_sample = self.sample(condition=z_start_indices, mode=mode)
        x_sample = self.indices_to_image(half_index_sample)

        # create new sample
        index_sample = self.sample(mode=mode)
        x_new = self.indices_to_image(index_sample)

        # create reconstruction
        x_rec = self.indices_to_image(z_indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = x_sample
        log["new_sample"] = x_new
        return log, torch.concat((x, x_rec, x_sample, x_new))

    def indices_to_image(self, indices, p1=16, p2=16):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image





