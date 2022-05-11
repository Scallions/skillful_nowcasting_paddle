"""Discriminator implementation."""

from .modules import layers
# import tensorflow.compat.v1 as tf

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Discriminator(nn.Layer):
    """Discriminator."""

    def __init__(self,
                 input_channel = 12,
                 num_spatial_frames = 8):
        """Constructor."""
        super().__init__()
        # Number of random time steps for the spatial discriminator.
        self.num_spatial_frames = 8
        # Input size ratio with respect to crop size for the temporal discriminator.
        self.temporal_crop_ratio = 2
        # As the input is the whole sequence of the event (including conditioning
        # frames), the spatial discriminator needs to pick only the t > T+0.
        self.num_conditioning_frames = 4
        self.spatial_discriminator = SpatialDiscriminator(input_channels=input_channel)
        self.temporal_discriminator = TemporalDiscriminator(input_channels=input_channel)

    def forward(self, frames):
        """Build the discriminator.

        Args:
          frames: a tensor with a complete observation [b, 22, 256, 256, 1].

        Returns:
          A tensor with discriminator loss scalars [b, 2].
        """
        # b, t, h, w, c = tf.shape(frames).as_list()
        b, t, c, h, w = frames.shape
        assert t >= self.num_spatial_frames + self.num_conditioning_frames

        # Prepare the frames for spatial discriminator: pick 8 random time steps out
        # of 18 lead time steps, and downsample from 256x256 to 128x128.
        # target_frames_sel = paddle.arange(self.num_conditioning_frames, t)
        permutation = paddle.stack([
            (paddle.randperm(t-self.num_conditioning_frames) + self.num_conditioning_frames)[:self.num_spatial_frames]
            for _ in range(b)
        ], 0)
        # TODO: delete for loop
        # frames_for_sd = paddle.gather(frames, permutation, axis=1)
        frames_for_sd = paddle.stack([paddle.gather(f, permutation[i]) for i, f in enumerate(paddle.unstack(frames))], axis=0)
        # frames_for_sd = F.avg_pool3d(
        #     frames_for_sd, [1, 2, 2], [1, 2, 2], data_format='channels_last')
        frames_for_sd = paddle.transpose(frames_for_sd, [0, 2, 1, 3, 4])
        frames_for_sd = F.avg_pool3d(
            frames_for_sd, [1, 2, 2], [1, 2, 2])
        frames_for_sd = paddle.transpose(frames_for_sd, [0, 2, 1, 3, 4])
        # Compute the average spatial discriminator score for each of 8 picked time
        # steps.
        sd_out = self.spatial_discriminator(frames_for_sd)

        # Prepare the frames for temporal discriminator: choose the offset of a
        # random crop of size 128x128 out of 256x256 and pick full sequence samples.
        cr = self.temporal_crop_ratio
        # h_offset = tf.random_uniform([], 0, (cr - 1) * (h // cr), tf.int32)
        # w_offset = paddle.random_uniform([], 0, (cr - 1) * (w // cr), tf.int32)
        h_offset = paddle.randint(0, (cr - 1) * (h // cr))
        w_offset = paddle.randint(0, (cr - 1) * (w // cr))
        zero_offset = paddle.zeros_like(w_offset)
        begin_tensor = paddle.stack(
            [zero_offset, zero_offset, zero_offset, h_offset, w_offset], -1)
        # size_tensor = tf.constant([b, t, h // cr, w // cr, c])
        size_tensor = paddle.to_tensor([b, t, c, h // cr, w // cr])
        # frames_for_td = tf.slice(frames, begin_tensor, size_tensor)
        # frames_for_td.set_shape([b, t, h // cr, w // cr, c])
        frames_for_td = paddle.slice(frames, [0,1,2,3,4],begin_tensor, begin_tensor+size_tensor)
        frames_for_td = paddle.reshape(frames_for_td, [b, t, c, h // cr, w // cr])
        # Compute the average temporal discriminator score over length 5 sequences.
        td_out = self.temporal_discriminator(frames_for_td)

        return paddle.concat([sd_out, td_out], axis=1)


class DBlock(nn.Layer):
    """Convolutional residual block."""

    def __init__(self, input_channels, output_channels, kernel_size=3, downsample=True,
                 pre_activation=True, conv=layers.SNConv2D,
                 pooling=layers.downsample_avg_pool, activation=F.relu):
        """Constructor for the D blocks of the DVD-GAN.

        Args:
          output_channels: Integer number of channels in the second convolution, and
            number of channels in the residual 1x1 convolution module.
          kernel_size: Integer kernel size of the convolutions.
          downsample: Boolean: shall we use the average pooling layer?
          pre_activation: Boolean: shall we apply pre-activation to inputs?
          conv: TF module, either layers.Conv2D or a wrapper with spectral
            normalisation layers.SNConv2D.
          pooling: Average pooling layer. Default: layers.downsample_avg_pool.
          activation: Activation at optional preactivation and first conv layers.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.pre_activation = pre_activation
        self.conv1 = conv(input_channels=input_channels,output_channels=self.input_channels,
                          kernel_size=self.kernel_size)
        self.conv2 = conv(input_channels=input_channels,output_channels=self.output_channels,
                          kernel_size=self.kernel_size)
        self.pooling = pooling
        self.activation = activation

        if self.input_channels != self.output_channels:
            self.conv3 = conv(input_channels=input_channels,output_channels=self.output_channels,
                              kernel_size=1)

    def forward(self, inputs):
        """Build the DBlock.

        Args:
          inputs: a tensor with a complete observation [b, 256, 256, 1]

        Returns:
          A tensor with discriminator loss scalars [b].
        """
        h0 = inputs

        # Pre-activation.
        if self.pre_activation:
            h0 = self.activation(h0)

        # First convolution.
        # input_channels = h0.shape.as_list()[-1]
        # input_channels = h0.shape[-1]
        h1 = self.conv1(h0)
        h1 = self.activation(h1)

        # Second convolution.
        h2 = self.conv2(h1)

        # Downsampling.
        if self.downsample:
            h2 = self.pooling(h2)

        # The residual connection, make sure it has the same dimensionality
        # with additional 1x1 convolution and downsampling if needed.
        if self.input_channels != self.output_channels or self.downsample:
            sc = self.conv3(inputs)
            if self.downsample:
                sc = self.pooling(sc)
        else:
            sc = inputs

        # Residual connection.
        return h2 + sc


class SpatialDiscriminator(nn.Layer):
    """Spatial Discriminator."""

    def __init__(self,
                 input_channels=12):
        super().__init__()
        self.blocks = nn.Sequential(
            DBlock(input_channels=input_channels*4,output_channels=48, pre_activation=False),
            DBlock(input_channels=48,output_channels=96),
            DBlock(input_channels=96,output_channels=192),
            DBlock(input_channels=192,output_channels=384),
            DBlock(input_channels=384,output_channels=768),
            DBlock(input_channels=768,output_channels=768, downsample=False)
        )

        self.bn = layers.BatchNorm(768,calc_sigma=False)
        self.output_layer = layers.Linear(input_size=768,output_size=1)

    def forward(self, frames):
        """Build the spatial discriminator.

        Args:
          frames: a tensor with a complete observation [b, n, 128, 128, 1].

        Returns:
          A tensor with discriminator loss scalars [b].
        """
        # b, n, h, w, c = tf.shape(frames).as_list()
        b, n, c, h, w = frames.shape

        # Process each of the n inputs independently.
        # frames = tf.reshape(frames, [b * n, h, w, c])
        frames = frames.reshape([b * n, c, h, w])

        # Space-to-depth stacking from 128x128x1 to 64x64x4.
        # TODO: space to depth
        # frames = tf.nn.space_to_depth(frames, block_size=2)
        # fake impl
        # frames = frames.reshape([b*n, c*4, h // 2, w // 2])
        frames = layers.pixel_shuffle_inv(frames, 2)

        # Five residual D Blocks to halve the resolution of the image and double
        # the number of channels.
        # y = DBlock(output_channels=48, pre_activation=False)(frames)
        # y = DBlock(output_channels=96)(y)
        # y = DBlock(output_channels=192)(y)
        # y = DBlock(output_channels=384)(y)
        # y = DBlock(output_channels=768)(y)

        # # One more D Block without downsampling or increase in number of channels.
        # y = DBlock(output_channels=768, downsample=False)(y)
        y = self.blocks(frames)
        # Sum-pool the representations and feed to spectrally normalized lin. layer.
        # y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
        y = paddle.sum(F.relu(y), axis=[2, 3])
        # y = layers.BatchNorm(calc_sigma=False)(y)
        # output_layer = layers.Linear(output_size=1)
        # output = output_layer(y)
        y = self.bn(y)
        output = self.output_layer(y)

        # Take the sum across the t samples. Note: we apply the ReLU to
        # (1 - score_real) and (1 + score_generated) in the loss.
        # output = tf.reshape(output, [b, n, 1])
        # output = tf.reduce_sum(output, keepdims=True, axis=1)
        output = output.reshape([b, n, 1])
        output = output.sum(axis=1, keepdim=True)
        return output


class TemporalDiscriminator(nn.Layer):
    """Spatial Discriminator."""

    def __init__(self,
                 input_channels = 12):
        super().__init__()
        self.blocks1 = nn.Sequential(
            DBlock(input_channels=input_channels,output_channels=48,
                   conv=layers.SNConv3D, pooling=layers.downsample_avg_pool3d,
                   pre_activation=False),
            DBlock(input_channels=48,output_channels=96,
                   conv=layers.SNConv3D,
                   pooling=layers.downsample_avg_pool3d)
        )

        self.blocks2 = nn.Sequential(
            DBlock(input_channels=96, output_channels=192),
            DBlock(input_channels=192, output_channels=384),
            DBlock(input_channels=384, output_channels=768),
            DBlock(input_channels=768, output_channels=768, downsample=False)
        )
        self.bn = layers.BatchNorm(768, calc_sigma=False)
        self.output_layer = layers.Linear(input_size=768,output_size=1)

    def forward(self, frames):
        """Build the temporal discriminator.

        Args:
          frames: a tensor with a complete observation [b, ts, 128, 128, 1]

        Returns:
          A tensor with discriminator loss scalars [b].
        """
        # b, ts, hs, ws, cs = tf.shape(frames).as_list()
        b, ts, cs, hs, ws = frames.shape

        # Process each of the ti inputs independently.
        # frames = tf.reshape(frames, [b * ts, hs, ws, cs])
        frames = frames.reshape([b * ts, cs, hs, ws])

        # Space-to-depth stacking from 128x128x1 to 64x64x4.
        # TODO: space to depth
        # frames = tf.nn.space_to_depth(frames, block_size=2)
        # fake impl
        # frames = frames.reshape([b, ts, cs*4, hs // 2, ws // 2])
        frames = layers.pixel_shuffle_inv(frames, 2)

        # Stack back to sequences of length ti.
        # frames = tf.reshape(frames, [b, ts, hs, ws, cs])
        frames = frames.reshape([b, ts, cs, hs, ws])

        # Two residual 3D Blocks to halve the resolution of the image, double
        # the number of channels, and reduce the number of time steps.
        frames = frames.transpose([0, 2, 1, 3, 4])
        y = self.blocks1(frames)
        y = y.transpose([0, 2, 1, 3, 4])
        frames = frames.transpose([0, 2, 1, 3, 4])
        # Get t < ts, h, w, and c, as we have downsampled in 3D.
        # _, t, h, w, c = tf.shape(frames).as_list()
        # TODO: should use shape of y ?
        _, t, c, h, w = y.shape

        # Process each of the t images independently.
        # b t h w c -> (b x t) h w c
        # y = tf.reshape(y, [-1] + [h, w, c])
        y = y.reshape([-1] + [c, h, w])

        # Three residual D Blocks to halve the resolution of the image and double
        # the number of channels.
        # y = DBlock(output_channels=192)(y)
        # y = DBlock(output_channels=384)(y)
        # y = DBlock(output_channels=768)(y)

        # One more D Block without downsampling or increase in number of channels.
        # y = DBlock(output_channels=768, downsample=False)(y)
        y = self.blocks2(y)

        # Sum-pool the representations and feed to spectrally normalized lin. layer.
        # y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
        y = paddle.sum(F.relu(y), axis=[2, 3])
        # y = layers.BatchNorm(calc_sigma=False)(y)
        y = self.bn(y)
        # output_layer = layers.Linear(output_size=1)
        output = self.output_layer(y)

        # Take the sum across the t samples. Note: we apply the ReLU to
        # (1 - score_real) and (1 + score_generated) in the loss.
        # output = tf.reshape(output, [b, t, 1])
        # scores = tf.reduce_sum(output, keepdims=True, axis=1)
        output = output.reshape([b, t, 1])
        scores = output.sum(axis=1, keepdim=True)
        return scores
