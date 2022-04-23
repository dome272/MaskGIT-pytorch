import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from bidirectional_transformer import BidirectionalTransformer
_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to("cuda")


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_image_tokens = args.num_image_tokens
        self.sos_token = args.num_codebook_vectors + 1
        self.mask_token_id = args.num_codebook_vectors
        self.choice_temperature = 4.5

        self.gamma = self.gamma_func("cosine")

        # self.transformer = BidirectionalTransformer(
        #                         patch_size=8, embed_dim=args.dim, depth=args.n_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192+1)
        self.transformer = BidirectionalTransformer(args)
        self.vqgan = self.load_vqgan(args)
        print(f"Transformer parameters: {sum([p.numel() for p in self.transformer.parameters()])}")

    def load_checkpoint(self, epoch):
        self.load_state_dict(torch.load(os.path.join("checkpoints", f"transformer_epoch_{epoch}.pt")))
        print("Check!")

    @staticmethod
    def load_vqgan():
        from vq_f16 import VQModel
        model = VQModel(ckpt_path="./checkpoints/vq-flickr.pt")
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        # quant_z, indices, _ = self.vqgan.encode(x)
        quant_z, _, (_, _, indices) = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    def forward(self, x):
        # _, z_indices = self.encode_to_z(x)
        #
        # r = np.random.uniform()
        # mask = torch.bernoulli(r * torch.ones(z_indices.shape[-1], device=z_indices.device))
        # mask = mask.round().bool()
        #
        # target = z_indices[:, mask]
        #
        # logits = self.transformer(z_indices, mask)

        _, z_indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=z_indices.device) * self.sos_token

        r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        # torch.rand(z_indices.shape, device=z_indices.device)
        # mask = torch.bernoulli(r * torch.ones(z_indices.shape, device=z_indices.device))
        # mask = torch.bernoulli(torch.rand(z_indices.shape, device=z_indices.device))
        # mask = mask.round().to(dtype=torch.int64)
        # masked_indices = torch.zeros_like(z_indices)
        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        a_indices = mask * z_indices + (~mask) * masked_indices

        a_indices = torch.cat((sos_tokens, a_indices), dim=1)

        target = torch.cat((sos_tokens, z_indices), dim=1)

        logits = self.transformer(a_indices)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        if k == 0:
            out[:, :] = self.sos_token
        else:
            out[out < v[..., [-1]]] = self.sos_token
        return out

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def create_input_tokens_normal(self, num, label=None):
        # label_tokens = label * torch.ones([num, 1])
        # Shift the label by codebook_size
        # label_tokens = label_tokens + self.vqgan.codebook.num_codebook_vectors
        # Create blank masked tokens
        blank_tokens = torch.ones((num, self.num_image_tokens), device="cuda")
        masked_tokens = self.mask_token_id * blank_tokens
        # Concatenate the two as input_tokens
        # input_tokens = torch.concat([label_tokens, masked_tokens], dim=-1)
        # return input_tokens.to(torch.int32)
        return masked_tokens.to(torch.int64)

    def tokens_to_logits(self, seq):
        logits = self.transformer(seq)
        # logits = logits[..., :self.vqgan.codebook.num_codebook_vectors]  # why is maskgit returning [8, 257, 2025]?
        return logits

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    @torch.no_grad()
    def sample_good(self, inputs=None, num=1, T=11, mode="cosine"):
        # self.transformer.eval()
        N = self.num_image_tokens
        if inputs is None:
            inputs = self.create_input_tokens_normal(num)
        else:
            inputs = torch.hstack(
                (inputs, torch.zeros((inputs.shape[0], N - inputs.shape[1]), device="cuda", dtype=torch.int).fill_(self.mask_token_id)))

        sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * self.sos_token
        inputs = torch.cat((sos_tokens, inputs), dim=1)

        unknown_number_in_the_beginning = torch.sum(inputs == self.mask_token_id, dim=-1)
        gamma = self.gamma_func(mode)
        cur_ids = inputs  # [8, 257]
        for t in range(T):
            logits = self.tokens_to_logits(cur_ids)  # call transformer to get predictions [8, 257, 1024]
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()

            unknown_map = (cur_ids == self.mask_token_id)  # which tokens need to be sampled -> bool [8, 257]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)  # replace all -1 with their samples and leave the others untouched [8, 257]

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs [8, 257, 1024]
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]

            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # ignore tokens which are already sampled [8, 257]

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True)-1, mask_len))  # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_ids = torch.where(masking, self.mask_token_id, sampled_ids)
            # print((cur_ids == 8192).count_nonzero())

        # self.transformer.train()
        return cur_ids[:, 1:]

    @torch.no_grad()
    def log_images(self, x, mode="cosine"):
        log = dict()

        _, z_indices = self.encode_to_z(x)

        # create new sample
        index_sample = self.sample_good(mode=mode)
        x_new = self.indices_to_image(index_sample)

        # create a "half" sample
        z_start_indices = z_indices[:, :z_indices.shape[1] // 2]
        half_index_sample = self.sample_good(z_start_indices, mode=mode)
        x_sample = self.indices_to_image(half_index_sample)

        # create reconstruction
        x_rec = self.indices_to_image(z_indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = x_sample
        log["new_sample"] = x_new
        return log, torch.concat((x, x_rec, x_sample, x_new))

    def indices_to_image(self, indices, p1=32, p2=32):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 32)
        # ix_to_vectors = self.vqgan.quantize.embedding(indices).reshape(indices.shape[0], 16, 16, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    @staticmethod
    def create_masked_image(image: torch.Tensor, x_start: int = 100, y_start: int = 100, size: int = 50):
        mask = torch.ones_like(image, dtype=torch.int)
        mask[:, :, x_start:x_start + size, y_start:y_start + size] = 0
        return image * mask, mask

    def inpainting(self, image: torch.Tensor, x_start: int = 100, y_start: int = 100, size: int = 50):
        # Note: this function probably doesnt work yet lol
        # apply mask on image
        masked_image, mask = self.create_masked_image(image, x_start, y_start, size)

        # encode masked image
        # _, indices = self.encode_to_z(masked_image)
        indices = torch.randint(1024, (1, 256), dtype=torch.int)
        mask = mask[:, 0, :, :]

        # set masked patches to be 0 -> so that the sampling part only samples indices for these patches
        # 1. idea: just calculate the ratio between 256x256 image and 16x16 latent image and set the area
        #          which was masked in the original image to 0 in the encoded image
        # 2. idea: check if patches which were masked in the original image are always the same in the latent space
        #          If so: set these to 0
        p = 16
        patched_mask = mask.unfold(2, p, p).unfold(1, p, p)
        patched_mask = torch.transpose(patched_mask, 3, 4)
        patched_mask = patched_mask.permute(1, 2, 0, 3, 4)
        patched_mask = patched_mask.contiguous().view(patched_mask.size(0) * patched_mask.size(1),
                                                      -1)  # 256 x 256 i.e. 16x16 x 256

        indices_mask, _ = torch.min(patched_mask, dim=-1)
        indices = indices_mask * indices

        # inpaint the image by using the sample method and provide the masked image indices and condition
        sampled_indices = self.sample(indices)

        # reconstruct inpainted image
        inpainted_image = self.indices_to_image(sampled_indices)

        # linearly blend the input image and inpainted image at border of mask (to avoid sharp edges at border of mask)
        indices_mask = indices_mask.reshape(1, 1, 16, 16).type(torch.float)
        upsampled_indices_mask = F.interpolate(indices_mask, scale_factor=16).squeeze(0)
        intra = torch.where(mask != upsampled_indices_mask, 1, 0)

        # define mask for blending
        n = 128
        base = torch.arange(n).view(1, -1).max(torch.arange(n).view(-1, 1))
        right = torch.stack((torch.rot90(base, 1, [0, 1]), base)).reshape(n * 2, n)
        left = torch.stack((torch.rot90(base, 2, [0, 1]), torch.rot90(base, 3, [0, 1]))).reshape(n * 2, n)
        full = torch.cat((left, right), 1)

        # construct opacity matrix for intra region
        min_blend = torch.min(torch.where(intra == 1, full, 1000000))
        max_blend = torch.max(torch.where(intra == 1, full, -1000000))
        mask_blend = torch.where(intra == 1, (full - min_blend) / max_blend, torch.ones_like(intra, dtype=torch.float))

        mask_real = torch.where(mask == 0, mask.type(torch.float), mask_blend)
        mask_fake = torch.where(mask == 0, (1 - mask).type(torch.float), mask_blend)

        blended_image = mask_real * image + mask_fake * inpainted_image

        return blended_image, inpainted_image

