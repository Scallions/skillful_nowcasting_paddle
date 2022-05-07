import paddle
import paddle.optimizer
from paddle.io import DataLoader
from dataset import NowCastingDataset
from models.generators import Generator
from models.discriminators import Discriminator
from models.modules.loss import loss_hinge_disc, loss_hinge_gen, grid_cell_regularizer

G = Generator(num_channels=1, lead_time=110, time_delta=5)
D = Discriminator(input_channel=1)

opt_G = paddle.optimizer.Adam(parameters=G.parameters())
opt_D = paddle.optimizer.Adam(parameters=D.parameters())

PATH = r'E:\dataset\pwv.nc'
LENGTH = 22
dataset = NowCastingDataset(PATH, LENGTH)
loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        drop_last=True,
                        num_workers=4) # if windows, num_workers must be set 0

epochs = 1

for epoch in range(epochs):
    #ds = [(paddle.rand([2,22,1,256,256]), paddle.rand([2,22,1,256,256]))]
    for inp, target in loader:
    #for inp, target in ds:
        # b t c h w

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
        gen_loss = gen_disc_loss + grid_cell_reg
        gen_loss.backward()
        opt_G.step()