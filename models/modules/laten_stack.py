"""Latent Conditioning Stack."""

from . import layers
# import tensorflow.compat.v1 as tf

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class LatentCondStack(nn.Layer):
    """Latent Conditioning Stack for the Sampler."""

    def __init__(self, resolution=(256, 256)):
        super().__init__()
        self.resolution = resolution
        self.conv1 = layers.SNConv2D(input_channels=8 ,output_channels=8, kernel_size=3)
        self.lblock1 = LBlock(input_channels=8, output_channels=24)
        self.lblock2 = LBlock(input_channels=24, output_channels=48)
        self.lblock3 = LBlock(input_channels=48, output_channels=192)
        self.mini_attn_block = Attention(num_channels=192)
        self.lblock4 = LBlock(input_channels=192, output_channels=768)

    def forward(self, batch_size, resolution=(256, 256)):

        # Independent draws from a Normal distribution.
        h, w = self.resolution[0] // 32, self.resolution[1] // 32
        # z = paddle.random.normal([batch_size, h, w, 8])
        z = paddle.randn([batch_size, 8, h, w])

        # 3x3 convolution.
        z = self.conv1(z)

        # Three L Blocks to increase the number of channels to 24, 48, 192.
        z = self.lblock1(z)
        z = self.lblock2(z)
        z = self.lblock3(z)

        # Spatial attention module.
        z = self.mini_attn_block(z)

        # L Block to increase the number of channels to 768.
        z = self.lblock4(z)

        return z


class LBlock(nn.Layer):
    """Residual block for the Latent Stack."""

    def __init__(self, input_channels, output_channels, kernel_size=3, conv=layers.Conv2D,
                 activation=F.relu):
        """Constructor for the D blocks of the DVD-GAN.

        Args:
          output_channels: Integer number of channels in convolution operations in
            the main branch, and number of channels in the output of the block.
          kernel_size: Integer kernel size of the convolutions. Default: 3.
          conv: TF module. Default: layers.Conv2D.
          activation: Activation before the conv. layers. Default: tf.nn.relu.
        """
        super().__init__()
        self.output_channels = output_channels
        # self.kernel_size = kernel_size
        self.convh1 = conv(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size)
        self.convh2 = conv(
            input_channels=output_channels,
            output_channels=output_channels,
            kernel_size=kernel_size)
        if input_channels < output_channels:
            self.convout = conv(
                input_channels=input_channels,
                output_channels=output_channels - input_channels,
                kernel_size=1)
        self.activation = activation

    def forward(self, inputs):
        """Build the LBlock.

        Args:
          inputs: a tensor with a complete observation [N 256 256 1]

        Returns:
          A tensor with discriminator loss scalars [B].
        """

        # Stack of two conv. layers and nonlinearities that increase the number of
        # channels.
        h0 = self.activation(inputs)
        # h1 = self.conv(num_channels=self.output_channels,
        #                kernel_size=self.kernel_size)(h0)
        h1 = self.convh1(h0)
        h1 = self.activation(h1)
        # h2 = self.conv(num_channels=self.output_channels,
        #                kernel_size=self.kernel_size)(h1)
        h2 = self.convh2(h1)

        # Prepare the residual connection branch.
        input_channels = h0.shape[1]
        if input_channels < self.output_channels:
            # sc = self.conv(num_channels=self.output_channels - input_channels,
                        #    kernel_size=1)(inputs)
            sc = self.convout(inputs)
            sc = paddle.concat([inputs, sc], axis=1)
        else:
            sc = inputs

        # Residual connection.
        return h2 + sc


def attention_einsum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""

    # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
    # k = tf.reshape(k, [-1, k.shape[-1]])  # [h, w, c] -> [L, c]
    # v = tf.reshape(v, [-1, v.shape[-1]])  # [h, w, c] -> [L, c]
    k = k.reshape([k.shape[0], -1])
    v = v.reshape([v.shape[0], -1])

    # Einstein summation corresponding to the query * key operation.
    # beta = tf.nn.softmax(tf.einsum('hwc, Lc->hwL', q, k), axis=-1)
    beta = F.softmax(paddle.einsum('chw, cL->Lhw', q, k), axis=0)
    # Einstein summation corresponding to the attention * value operation.
    # out = tf.einsum('hwL, Lc->hwc', beta, v)
    out = paddle.einsum('Lhw, cL->chw', beta, v)
    return out


class Attention(nn.Layer):
    """Attention module."""

    def __init__(self, num_channels, ratio_kq=8, ratio_v=8, conv=layers.Conv2D):
        """Constructor."""
        super().__init__()
        self.num_channels = num_channels
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.conv = conv

        self.query_conv = conv(
            input_channels=num_channels,
            output_channels=num_channels // ratio_kq,
            kernel_size=1, padding='VALID', use_bias=False)
        self.key_conv = conv(
            input_channels=num_channels,
            output_channels=num_channels // ratio_kq,
            kernel_size=1, padding='VALID', use_bias=False)
        self.value_conv = conv(
            input_channels=num_channels,
            output_channels=num_channels // ratio_v,
            kernel_size=1, padding='VALID', use_bias=False)
        self.out_conv = conv(
            input_channels=num_channels // ratio_v,
            output_channels=num_channels,
            kernel_size=1, padding='VALID', use_bias=False)

        # Learnable gain parameter
        # self.gamma = tf.get_variable(
        #     'miniattn_gamma', shape=[],
        #     initializer=tf.initializers.zeros(tf.float32))
        self.gamma = self.create_parameter([1], default_initializer=nn.initializer.Constant(0.0))
        self.add_parameter("miniattn_gamma", self.gamma)

    def forward(self, tensor):
        # Compute query, key and value using 1x1 convolutions.
        query = self.query_conv(tensor)
        key = self.key_conv(tensor)
        value = self.value_conv(tensor)

        # Apply the attention operation.
        out = layers.ApplyAlongAxis(
            attention_einsum, axis=0)(query, key, value)
        out = self.gamma * self.out_conv(out)

        # Residual connection.
        return out + tensor
