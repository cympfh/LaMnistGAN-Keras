# LaMnistGAN-Keras

autoencoder + Latent Space GAN with MNIST

## Architecture

```
Encoder : MNIST -> RealZ
Decoder : RealZ -> MNIST
(Autoencoder = Decoder . Encoder)
Generator : U -> FakeZ
Discriminator : Z -> R
    : RealZ -> R[>0]
    : FakeZ -> R[<0]
(GAN : Discriminator . Generator)
```

# ref

- [arxiv:1810.06640 (LaTextGAN)](https://arxiv.org/abs/1810.06640)

