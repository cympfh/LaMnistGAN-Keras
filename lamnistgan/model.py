from keras import backend as K
from keras import regularizers
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, Lambda, Reshape, UpSampling2D)
from keras.models import Sequential
from keras.optimizers import Adam

X_SHAPE = (28, 28)  # sizeof MNIST
Z_DIM = 32  # sizeof Z (featur space of MNIST)
U_DIM = 128  # sizeof U (generative space of Z)


def build_enc():
    """MNIST -> (Real)Z"""
    model = Sequential(name='enc')
    model.add(Reshape(X_SHAPE + (1,), input_shape=X_SHAPE))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(.3))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(.3))
    model.add(Conv2D(128, (4, 4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(Z_DIM, activation='relu'))
    model.add(Dense(Z_DIM))
    return model


def build_dec():
    """(Real)Z -> MNIST"""
    model = Sequential(name='dec')
    model.add(Dense(128 * 7 * 7, activation='relu', input_shape=(Z_DIM,)))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (5, 5), activation='sigmoid', padding='same'))
    model.add(Reshape(X_SHAPE))
    return model


def build_auto(enc, dec):
    model = Sequential(name='auto')
    model.add(enc)
    model.add(dec)
    return model


def build_gen():
    """U -> (Fake)Z"""
    model = Sequential(name='gen')
    model.add(Dense(Z_DIM, activation='relu', input_shape=(U_DIM,)))
    model.add(Dense(Z_DIM, activation='relu'))
    model.add(Dense(Z_DIM, activation='relu'))
    model.add(Dense(Z_DIM))
    return model


def build_clf_core():
    """Z -> 10"""
    model = Sequential(name='core')
    model.add(Dense(Z_DIM,
                    kernel_regularizer=regularizers.l2(0.01),
                    activation='relu',
                    input_shape=(Z_DIM,)))
    model.add(Dense(Z_DIM // 2,
                    kernel_regularizer=regularizers.l2(0.01),
                    activation='relu'))
    model.add(Dense(Z_DIM // 2,
                    kernel_regularizer=regularizers.l2(0.01),
                    activation='relu'))
    model.add(Dense(10))
    return model


def build_dis(core):
    """Z -> R[0, 1]

    nearly 1 if it is likely real,
    nearly 0 if it seems fake
    """
    def predict(y):
        p = 1.0 - (1.0 / (K.sum(K.exp(y), axis=-1, keepdims=True) + 1.0))
        return p
    model = Sequential(name='dis')
    model.add(core)
    model.add(Lambda(predict))
    return model


def build_clf(extracter):
    """Z -> 10"""
    model = Sequential(name='clf')
    model.add(extracter)
    model.add(Activation('softmax'))
    return model


def build_gan_real(enc, dis):
    """Training Encoder /fixed Discriminator"""
    model = Sequential(name='gan_real')
    model.add(enc)
    model.add(dis)
    return model


def build_gan_fake(gen, dis):
    """Training Generator /fixed Discriminator"""
    model = Sequential(name='gan_fake')
    model.add(gen)
    model.add(dis)
    return model


def build():

    # NNs
    enc = build_enc()
    dec = build_dec()
    auto = build_auto(enc, dec)

    core = build_clf_core()

    gen = build_gen()
    dis = build_dis(core)
    gan_real = build_gan_real(enc, dis)
    gan_fake = build_gan_fake(gen, dis)

    clf = build_clf(core)

    def opt(lr_eff=1.0, beta_1_eff=1.0, beta_2_eff=1.0, decay=0.0, clip_value=1.0):
        lr = 0.001 * lr_eff
        beta_1 = 0.9 * beta_1_eff
        beta_2 = 0.999 * beta_2_eff
        return Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay, clipvalue=clip_value)

    # autoencoder
    enc.trainable = True
    dec.trainable = True
    auto.compile(loss='mse', optimizer=opt(lr_eff=0.1))

    # classifier
    clf.trainable = True
    clf_loss = 'sparse_categorical_crossentropy'
    metrics = ['acc']
    clf.compile(loss=clf_loss, metrics=metrics, optimizer=opt(lr_eff=8.0))

    dis_loss = 'binary_crossentropy'

    # GAN (for dis)
    dis.trainable = True
    gen.trainable = False
    enc.trainable = False
    dis.compile(loss=dis_loss, optimizer=opt(lr_eff=0.4, decay=0.001))

    # GAN (for gen)
    gen.trainable = True
    dis.trainable = False
    gan_fake.compile(loss=dis_loss, optimizer=opt(lr_eff=1.0, decay=0.001))

    # GAN (for enc)
    enc.trainable = True
    dis.trainable = False
    gan_real.compile(loss=dis_loss, optimizer=opt(lr_eff=0.1))

    return enc, dec, auto, gen, dis, gan_fake, gan_real, clf
