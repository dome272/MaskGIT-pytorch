# MaskGIT-pytorch
Pytorch implementation of MaskGIT: Masked Generative Image Transformer (https://arxiv.org/pdf/2202.04200.pdf)
<p align="center">
<img width="718" alt="results" src="https://user-images.githubusercontent.com/61938694/154553460-3eb2b55e-e313-4100-bc5e-b9d8c4dd8cd7.png">
</p>

#### Note: this is work in progress

MaskGIT is an extension to the VQGAN paper which improves the second stage transformer part (and leaves the first stage untouched). It switches the unidirectional transformer for a bidirectional transformer. The (second stage) training is pretty similar to BERT by randomly masking out tokens and trying to predict these using the bidirectional transformer (the original work used a GPT architecture randomly replaced tokens by other tokens). Different from BERT, the percentage for the masking is not fixed and uniformly distributed between 0 and 1 for each batch. Furhtermore, a new inference algorithm is suggested in which we start off by a completely masked-out image and then iteratively sample vectors where the model has a high confidence.

If you are only interested in the part of the code that comes from this paper check out [transformer.py](https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py).

## Run the code
The code is ready for training both the VQGAN and the Bidirectional Transformer and can also be used for inference

```python training_vqgan.py```

```python training_transformer.py```

(Make sure to edit the path for the dataset etc.)

## TODO
- [x] Implement the gamma functions
- [ ] Implement functions for image editing tasks: inpainting, extrapolation, image manipulation
  - started working on inpainting function. [transformer.py](https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py#L152)
- [ ] Tune hyperparameters
- [ ] (Provide visual results)
- [ ] Pre-Norm instead of Post-Norm

