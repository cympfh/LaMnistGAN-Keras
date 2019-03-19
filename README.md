# LaMnistGAN-Keras

autoencoder + Latent Space GAN with MNIST

## Architecture

```
Encoder : MNIST -> RealZ
Decoder : RealZ -> MNIST
(Autoencoder = Decoder . Encoder)
Generator : U -> FakeZ
Discriminator : Z -> Class[10] or Bool[0,1]
    : RealZ -> Class
    : FakeZ -> Bool[0]
(GAN : Discriminator . Generator)
```

## Training

```
1. train `Autoencoder`
    1. training `Encoder` and `Decoder`
        - to output self (with L2 loss)
1. train `GAN`
    1. training `Discriminator` /fixed `Generator` and `Encoder`
        - to output 1.0 from `Encoder`
        - to output 0.0 from `Generator`
    1. training `Generator` /fixed `Discriminator`
        - to output 1.0 by `Discriminator`
    1. training `Encoder` /fixed `Discriminator`
        - to output 1.0 by `Discriminator`
```

# ref

- [arxiv:1810.06640 (LaTextGAN)](https://arxiv.org/abs/1810.06640)

