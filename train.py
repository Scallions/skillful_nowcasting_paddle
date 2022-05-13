import paddle
import os
import numpy as np
from datetime import datetime
from logger import logger_config
from visualdl import LogWriter
from tools import save_model
import paddle.optimizer
from paddle.io import DataLoader
from dataset import NowCastingDataset
from models.generators import Generator
from models.discriminators import Discriminator
from models.modules.loss import loss_hinge_disc, loss_hinge_gen, grid_cell_regularizer

current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
if not os.path.isdir('result'):
    os.makedirs('result')
logger = logger_config(log_path='result/DGMR_' + current_time + '.txt', logging_name='DGMR')
writer = LogWriter(log_dir='result/vd')

G = Generator(num_channels=1, lead_time=90, time_delta=5)
D = Discriminator(input_channel=1)

opt_G = paddle.optimizer.Adam(5e-5, parameters=G.parameters(), beta1=0.0, beta2=0.999)
opt_D = paddle.optimizer.Adam(2e-4, parameters=D.parameters(), beta1=0.0, beta2=0.999)

PATH = r'/home/aistudio/data/data144877/pwv.nc'
LENGTH = 18
BATCH_SIZE = 1

train_dataset = NowCastingDataset(PATH, LENGTH, 0.8, training=True)
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          drop_last=True,
                          num_workers=4)  # if windows, num_workers must be set 0

test_dataset = NowCastingDataset(PATH, LENGTH, 0.8, training=False)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         drop_last=True,
                         num_workers=4)

TOTAL_EPOCH = 100
TOTAL_STEP = len(train_loader)
all_step = 0

for epoch in range(TOTAL_EPOCH):
    # ds = [(paddle.rand([2,22,1,256,256]), paddle.rand([2,22,1,256,256]))]
    for step, (inp, target) in enumerate(train_loader):
        # for inp, target in ds:
        all_step += 1
        # b t c h w
        inp = inp.astype(paddle.float32)
        target = target.astype(paddle.float32)
        ## disc
        opt_D.clear_grad()
        pred = G(inp)
        gen_seq = paddle.concat([inp, pred], axis=1)
        real_seq = paddle.concat([inp, target], axis=1)
        concat_inps = paddle.concat([real_seq, gen_seq], axis=0)

        concat_outs = D(concat_inps)
        score_real, score_gen = paddle.split(concat_outs, num_or_sections=2, axis=0)

        disc_loss = loss_hinge_disc(score_gen, score_real)

        disc_loss.backward()
        opt_D.step()

        ## gen
        opt_G.clear_grad()
        num_samples = 6
        gen_samples = [
            G(inp) for _ in range(num_samples)
        ]
        grid_cell_reg = grid_cell_regularizer(paddle.stack(gen_samples, axis=0), target)
        gen_seqs = [
            paddle.concat([inp, gen_sample], axis=1) for gen_sample in gen_samples
        ]
        gen_disc_loss = loss_hinge_gen(D(paddle.concat(gen_seqs, axis=0)))
        gen_loss = gen_disc_loss + 20 * grid_cell_reg
        gen_loss.backward()
        opt_G.step()

        if step % 100 == 0:
            logger.info('[TRAIN] Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'.format(epoch,
                                                                                                     TOTAL_EPOCH,
                                                                                                     step,
                                                                                                     TOTAL_STEP,
                                                                                                     disc_loss.item(),
                                                                                                     gen_loss.item()))
            writer.add_scalar(tag="train/ld", step=all_step, value=disc_loss.numpy().item())
            writer.add_scalar(tag="train/lg", step=all_step, value=gen_loss.numpy().item())

        if all_step % 2000 == 0:
            save_model(G, opt_G, 'model', epoch, all_step, 'G')
            save_model(D, opt_D, 'model', epoch, all_step, 'D')

        if all_step % 1000 == 0:
            # eval
            D.eval()
            G.eval()
            disc_loss_list = []
            gen_loss_list = []
            with paddle.no_grad():
                for inp, target in test_loader:
                    inp = inp.astype(paddle.float32)
                    target = target.astype(paddle.float32)
                    pred = G(inp)
                    gen_seq = paddle.concat([inp, pred], axis=1)
                    real_seq = paddle.concat([inp, target], axis=1)
                    concat_inps = paddle.concat([real_seq, gen_seq], axis=0)
                    concat_outs = D(concat_inps)
                    score_real, score_gen = paddle.split(concat_outs, num_or_sections=2, axis=0)

                    disc_loss = loss_hinge_disc(score_gen, score_real)

                    num_samples = 6
                    gen_samples = [
                        G(inp) for _ in range(num_samples)
                    ]
                    grid_cell_reg = grid_cell_regularizer(paddle.stack(gen_samples, axis=0), target)
                    gen_seqs = [
                        paddle.concat([inp, gen_sample], axis=1) for gen_sample in gen_samples
                    ]
                    gen_disc_loss = loss_hinge_gen(D(paddle.concat(gen_seqs, axis=0)))
                    gen_loss = gen_disc_loss + grid_cell_reg

                    disc_loss_list.append(disc_loss.item())
                    gen_loss_list.append(gen_loss.item())

            test_disc_loss = np.mean(disc_loss_list)
            test_gen_loss = np.mean(gen_loss_list)
            logger.info('[EVAL] Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'.format(epoch,
                                                                                      TOTAL_EPOCH,
                                                                                      test_disc_loss,
                                                                                      test_gen_loss))
            writer.add_scalar(tag="test/ld", step=all_step, value=test_disc_loss.item())
            writer.add_scalar(tag="test/lg", step=all_step, value=test_gen_loss.item())
            writer.add_image(tag="predict",
                             img=pred[0, 0, 0, :, :].detach().numpy(),
                             step=step,
                             dataformats="HW",
                             )
            writer.add_image(tag="true",
                             img=target[0, 0, 0, :, :].detach().numpy(),
                             step=step,
                             dataformats="HW",
                             )
            D.train()
            G.train()

