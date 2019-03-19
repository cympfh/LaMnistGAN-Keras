import os

import click
import numpy
import tensorflow as tf
from keras import backend as K

import dataset
from lamnistgan import logging
from lamnistgan import model as Model
from lamnistgan import testing


def echo(*args, fg='green'):
    click.secho(' '.join(str(arg) for arg in args), fg=fg, err=True)


@click.group()
def main():
    pass


@main.command()
@click.option('--name', help='model name', default=None)
@click.option('--resume', help='when resume learning from the snapshot')
@click.option('--batch-size', type=int, default=32)
@click.option('--epochs', type=int, default=5)
@click.option('--verbose', type=int, default=1)
def train(name, resume, batch_size, epochs, verbose):

    # paths
    log_path = None if name is None else f"logs/{name}.json"
    out_path = None if name is None else f"snapshots/{name}.{{epoch:06d}}.h5"
    echo('log path', log_path)
    echo('out path', out_path)
    result_dir = "result/{}".format(name)
    echo('result images', result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # running parameters
    run_params = locals()

    # init
    echo('train', run_params)
    log = logging.Logger(log_path)
    log({'train': run_params})
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(1)

    if name is None:
        echo("Warning: name is None. Models will never be saved.", fg='red')

    # dataset
    echo('dataset loading...')
    seq_train, seq_valid = dataset.batch_generator(batch_size)

    # model building
    echo('model building...')
    enc, dec, auto, gen, dis, gan_fake, gan_real, clf = Model.build()

    echo('Encoder', fg='yellow')
    enc.summary()
    echo('Decoder', fg='yellow')
    dec.summary()
    echo('Generator', fg='yellow')
    gen.summary()
    echo('Discriminator', fg='yellow')
    dis.summary()
    echo('GANs', fg='yellow')
    gan_fake.summary()
    gan_real.summary()
    echo('Classifier', fg='yellow')
    clf.summary()

    # training
    echo('start learning...')

    eps = 0.001
    TRUE = numpy.ones((batch_size,)) - eps
    FALSE = numpy.zeros((batch_size,)) + eps

    def make_noise():
        """() -> U"""
        x = numpy.random.randn(batch_size, Model.U_DIM)
        return x

    def interpolating():
        U = make_noise()
        Z = gen.predict_on_batch(U)
        u = Z[0]
        v = Z[1] + 0.001
        X = numpy.array([a * u + (1 - a) * v
                        for a in numpy.linspace(0, 3, batch_size)])
        return X

    for epoch in range(epochs):

        INTERVAL = 20
        auto_loss = 0
        dis_fake_loss = 0
        dis_real_loss = 0
        gen_loss = 0
        enc_loss = 0
        clf_loss = 0
        clf_acc = 0
        last_metrics = ()

        # Training
        for i, (X, y) in enumerate(seq_train):

            xsize = len(X)

            # Training autoencodfer
            auto.trainable = True
            enc.trainable = True
            dec.trainable = True
            auto_loss += auto.train_on_batch(X, X)

            # Training GAN (dis for real)
            clf.trainable = True
            dis.trainable = True
            enc.trainable = False
            gen.trainable = False
            z_real = enc.predict_on_batch(X)
            dis_real_loss += dis.train_on_batch(z_real, TRUE[:xsize])

            # Training GAN (dis for fake)
            clf.trainable = True
            dis.trainable = True
            enc.trainable = False
            gen.trainable = False
            u = make_noise()
            z_fake = gen.predict_on_batch(u)
            dis_fake_loss += dis.train_on_batch(z_fake, FALSE)

            # Training GAN (gen)
            clf.trainable = False
            dis.trainable = False
            gen.trainable = True
            u = make_noise()
            gen_loss += gan_fake.train_on_batch(u, TRUE)

            # Training GAN (enc)
            dis.trainable = False
            clf.trainable = False
            enc.trainable = True
            enc_loss += gan_real.train_on_batch(X, TRUE[:xsize])

            # Training Classifier (clf for real data)
            clf.trainable = True
            dis.trainable = True
            enc.trainable = False
            gen.trainable = False
            z_real = enc.predict_on_batch(X)
            _loss, _acc = clf.train_on_batch(z_real, y)
            clf_loss += _loss
            clf_acc += _acc

            if i % INTERVAL == INTERVAL - 1:
                if i > INTERVAL:
                    click.echo('\r', nl=False)
                auto_loss /= INTERVAL
                dis_fake_loss /= INTERVAL
                dis_real_loss /= INTERVAL
                gen_loss /= INTERVAL
                enc_loss /= INTERVAL
                clf_loss /= INTERVAL
                clf_acc /= INTERVAL
                click.echo(
                    f"Epoch:{epoch:4d} "
                    f"Train: "
                    f"auto={auto_loss:.4f} "
                    f"dis_fake={dis_fake_loss:.4f} "
                    f"dis_real={dis_real_loss:.4f} "
                    f"gen={gen_loss:.4f} "
                    f"enc={enc_loss:.4f} "
                    f"clf={clf_loss:.4f},{clf_acc:.4f} ",
                    nl=False
                )
                last_metrics = (auto_loss, dis_fake_loss, dis_real_loss,
                                gen_loss, enc_loss,
                                clf_loss, clf_acc)
                auto_loss = 0
                dis_fake_loss = 0
                dis_real_loss = 0
                gen_loss = 0
                enc_loss = 0
                clf_loss = 0
                clf_acc = 0
        click.echo('')

        # Logging
        auto_loss, dis_fake_loss, dis_real_loss, \
            gen_loss, enc_loss, clf_loss, clf_acc = last_metrics
        log({'epoch': epoch,
             'loss': {
                 'auto': auto_loss,
                 'dis_fake_loss': dis_fake_loss,
                 'dis_real_loss': dis_real_loss,
                 'gen_loss': gen_loss,
                 'enc_loss': enc_loss,
                 'clf_loss': clf_loss,
                 'clf_acc': clf_acc,
                 }})

        # Testing
        X, _ = seq_valid[0]
        X_rec = auto.predict_on_batch(X)
        testing.imgsave(X_rec, f"{result_dir}/rec.{epoch:03d}.png")
        u = make_noise()
        z = gen.predict_on_batch(u)
        X_smp = dec.predict_on_batch(z)
        testing.imgsave(X_smp, f"{result_dir}/smp.{epoch:03d}.png")
        z = interpolating()
        X_int = dec.predict_on_batch(z)
        testing.imgsave(X_int, f"{result_dir}/int.{epoch:03d}.png")


if __name__ == '__main__':
    main()
