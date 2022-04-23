"""Generator implementation."""

import functools
from . import discriminators
from .modules import laten_stack
from .modules import layers
# import tensorflow.compat.v1 as tf

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Generator(nn.Layer):
    """Generator for the proposed model."""

    def __init__(self, lead_time=90, time_delta=5):
        """Constructor.

        Args:
          lead_time: last lead time for the generator to predict. Default: 90 min.
          time_delta: time step between predictions. Default: 5 min.
        """
        super().__init__()
        self.cond_stack = ConditioningStack()
        self.sampler = Sampler(lead_time, time_delta)

    def forward(self, inputs):
        """Connect to a graph.

        Args:
          inputs: a batch of inputs on the shape [batch_size, time, h, w, 1].
        Returns:
          predictions: a batch of predictions in the form
            [batch_size, num_lead_times, h, w, 1].
        """
        _, _, _, height, width = inputs.shape  # .as_list()
        initial_states = self.cond_stack(inputs)
        predictions = self.sampler(initial_states, [height, width])
        return predictions

    def get_variables(self):
        """Get all variables of the module."""
        # pass
        raise NotImplementedError


class ConditioningStack(nn.Layer):
    """Conditioning Stack for the Generator."""

    def __init__(self, input_channels):
        super().__init__()
        self.block1 = discriminators.DBlock(input_channels=input_channels,output_channels=48, downsample=True)
        self.conv_mix1 = layers.SNConv2D(input_channels=input_channels,output_channels=48, kernel_size=3)
        self.block2 = discriminators.DBlock(input_channels=input_channels,output_channels=96, downsample=True)
        self.conv_mix2 = layers.SNConv2D(input_channels=input_channels,output_channels=96, kernel_size=3)
        self.block3 = discriminators.DBlock(input_channels=input_channels,
            output_channels=192, downsample=True)
        self.conv_mix3 = layers.SNConv2D(input_channels=input_channels,output_channels=192, kernel_size=3)
        self.block4 = discriminators.DBlock(input_channels=input_channels,
            output_channels=384, downsample=True)
        self.conv_mix4 = layers.SNConv2D(input_channels=input_channels,output_channels=384, kernel_size=3)

    def forward(self, inputs):
        # Space to depth conversion of 256x256x1 radar to 128x128x4 hiddens.
        # TODO: space to depth
        h0 = batch_apply(
            functools.partial(tf.nn.space_to_depth, block_size=2), inputs)

        # Downsampling residual D Blocks.
        h1 = time_apply(self.block1, h0)
        h2 = time_apply(self.block2, h1)
        h3 = time_apply(self.block3, h2)
        h4 = time_apply(self.block4, h3)

        # Spectrally normalized convolutions, followed by rectified linear units.
        init_state_1 = self.mixing_layer(h1, self.conv_mix1)
        init_state_2 = self.mixing_layer(h2, self.conv_mix2)
        init_state_3 = self.mixing_layer(h3, self.conv_mix3)
        init_state_4 = self.mixing_layer(h4, self.conv_mix4)

        # Return a stack of conditioning representations of size 64x64x48, 32x32x96,
        # 16x16x192 and 8x8x384.
        return init_state_1, init_state_2, init_state_3, init_state_4

    def mixing_layer(self, inputs, conv_block):
        # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
        # then perform convolution on the output while preserving number of c.
        stacked_inputs = paddle.concat(paddle.unstack(inputs, axis=1), axis=-1)
        return F.relu(conv_block(stacked_inputs))


class Sampler(object):
    """Sampler for the Generator."""

    def __init__(self, lead_time=90, time_delta=5):
        self.num_predictions = lead_time // time_delta
        self.latent_stack = laten_stack.LatentCondStack()

        self.conv_gru4 = ConvGRU()
        self.conv4 = layers.SNConv2D(kernel_size=1, output_channels=768)
        self.gblock4 = GBlock(output_channels=768)
        self.g_up_block4 = UpsampleGBlock(output_channels=384)

        self.conv_gru3 = ConvGRU()
        self.conv3 = layers.SNConv2D(kernel_size=1, output_channels=384)
        self.gblock3 = GBlock(output_channels=384)
        self.g_up_block3 = UpsampleGBlock(output_channels=192)

        self.conv_gru2 = ConvGRU()
        self.conv2 = layers.SNConv2D(kernel_size=1, output_channels=192)
        self.gblock2 = GBlock(output_channels=192)
        self.g_up_block2 = GBlock(output_channels=96)

        self.conv_gru1 = ConvGRU()
        self.conv1 = layers.SNConv2D(kernel_size=1, output_channels=96)
        self.gblock1 = GBlock(output_channels=96)
        self.g_up_block1 = UpsampleGBlock(output_channels=48)

        self.bn = layers.BatchNorm()
        self.output_conv = layers.SNConv2D(kernel_size=1, output_channels=4)

    def forward(self, initial_states, resolution):
        init_state_1, init_state_2, init_state_3, init_state_4 = initial_states
        batch_size = init_state_1.shape[0]

        # Latent conditioning stack.
        z = self.latent_stack(batch_size, resolution)
        hs = [z] * self.num_predictions

        # Layer 4 (bottom-most).
        # hs, _ = tf.nn.static_rnn(self.conv_gru4, hs, init_state_4)
        hs, _ = self.conv_gru4(hs, init_state_4)
        hs = [self.conv4(h) for h in hs]
        hs = [self.gblock4(h) for h in hs]
        hs = [self.g_up_block4(h) for h in hs]

        # Layer 3.
        # hs, _ = tf.nn.static_rnn(self.conv_gru3, hs, init_state_3)
        hs, _ = self.conv_gru3(hs, init_state_3)
        hs = [self.conv3(h) for h in hs]
        hs = [self.gblock3(h) for h in hs]
        hs = [self.g_up_block3(h) for h in hs]

        # Layer 2.
        # hs, _ = tf.nn.static_rnn(self.conv_gru2, hs, init_state_2)
        hs, _ = self.conv_gru2(hs, init_state_2)
        hs = [self.conv2(h) for h in hs]
        hs = [self.gblock2(h) for h in hs]
        hs = [self.g_up_block2(h) for h in hs]

        # Layer 1 (top-most).
        # hs, _ = tf.nn.static_rnn(self.conv_gru1, hs, init_state_1)
        hs, _ = self.conv_gru1(hs, init_state_1)
        hs = [self.conv1(h) for h in hs]
        hs = [self.gblock1(h) for h in hs]
        hs = [self.g_up_block1(h) for h in hs]

        # Output layer.
        hs = [F.relu(self.bn(h)) for h in hs]
        hs = [self.output_conv(h) for h in hs]
        # TODO: depth to space
        hs = [tf.nn.depth_to_space(h, 2) for h in hs]

        return paddle.stack(hs, axis=1)


class GBlock(nn.Layer):
    """Residual generator block without upsampling."""

    def __init__(self, input_channels, output_channels, sn_eps=0.0001):
        super().__init__()
        self.conv1_3x3 = layers.SNConv2D(
            input_channels=input_channels, output_channels=output_channels, kernel_size=3, sn_eps=sn_eps)
        self.bn1 = layers.BatchNorm(num_channels=input_channels)
        self.conv2_3x3 = layers.SNConv2D(
            input_channels=output_channels,
            output_channels=output_channels, kernel_size=3, sn_eps=sn_eps)
        self.bn2 = layers.BatchNorm(num_channels=output_channels)
        self.output_channels = output_channels
        self.sn_eps = sn_eps

        if input_channels != self.output_channels:
            self.conv_1x1 = layers.SNConv2D(
                input_channels=input_channels,
                output_channels=self.output_channels, kernel_size=1, sn_eps=self.sn_eps)

    def forward(self, inputs):
        input_channels = inputs.shape[-1]

        # Optional spectrally normalized 1x1 convolution.
        if input_channels != self.output_channels:
            sc = self.conv_1x1(inputs)
        else:
            sc = inputs

        # Two-layer residual connection, with batch normalization, nonlinearity and
        # 3x3 spectrally normalized convolution in each layer.
        h = F.relu(self.bn1(inputs))
        h = self.conv1_3x3(h)
        h = F.relu(self.bn2(h))
        h = self.conv2_3x3(h)

        # Residual connection.
        return h + sc


class UpsampleGBlock(nn.Layer):
    """Upsampling residual generator block."""

    def __init__(self, input_channels, output_channels, sn_eps=0.0001):
        super().__init__()
        self.conv_1x1 = layers.SNConv2D(
            input_channels=input_channels,
            output_channels=output_channels, kernel_size=1, sn_eps=sn_eps)
        self.conv1_3x3 = layers.SNConv2D(
            input_channels=input_channels,
            output_channels=output_channels, kernel_size=3, sn_eps=sn_eps)
        self.bn1 = layers.BatchNorm(input_channels)
        self.conv2_3x3 = layers.SNConv2D(
            input_channels=output_channels,
            output_channels=output_channels, kernel_size=3, sn_eps=sn_eps)
        self.bn2 = layers.BatchNorm(output_channels)
        self.output_channels = output_channels

    def forward(self, inputs):
        # x2 upsampling and spectrally normalized 1x1 convolution.
        sc = layers.upsample_nearest_neighbor(inputs, upsample_size=2)
        sc = self.conv_1x1(sc)

        # Two-layer residual connection, with batch normalization, nonlinearity and
        # 3x3 spectrally normalized convolution in each layer, and x2 upsampling in
        # the first layer.
        h = F.relu(self.bn1(inputs))
        h = layers.upsample_nearest_neighbor(h, upsample_size=2)
        h = self.conv1_3x3(h)
        h = F.relu(self.bn2(h))
        h = self.conv2_3x3(h)

        # Residual connection.
        return h + sc


class ConvGRU(nn.Layer):
    """A ConvGRU implementation."""

    def __init__(self, num_channels, state_channels, kernel_size=3, sn_eps=0.0001):
        """Constructor.

        Args:
          kernel_size: kernel size of the convolutions. Default: 3.
          sn_eps: constant for spectral normalization. Default: 1e-4.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sn_eps = sn_eps
        self.read_gate_conv = layers.SNConv2D(
            state_channels+num_channels,state_channels, self.kernel_size, sn_eps=self.sn_eps)
        self.update_gate_conv = layers.SNConv2D(
            num_channels+state_channels,state_channels, self.kernel_size, sn_eps=self.sn_eps)
        self.output_conv = layers.SNConv2D(
            num_channels+state_channels,state_channels, self.kernel_size, sn_eps=self.sn_eps)

    def forward(self, inputs, prev_state):

        # Concatenate the inputs and previous state along the channel axis.
        # num_channels = prev_state.shape[1]
        xh = paddle.concat([inputs, prev_state], axis=1)

        # Read gate of the GRU.

        read_gate = F.sigmoid(self.read_gate_conv(xh))

        # Update gate of the GRU.

        update_gate = F.sigmoid(self.update_gate_conv(xh))

        # Gate the inputs.
        gated_input = paddle.concat([inputs, read_gate * prev_state], axis=1)

        # Gate the cell and state / outputs.

        c = F.relu(self.output_conv(gated_input))
        out = update_gate * prev_state + (1. - update_gate) * c
        new_state = out

        return out, new_state


def time_apply(func, inputs):
    """Apply function func on each element of inputs along the time axis."""
    return layers.ApplyAlongAxis(func, axis=1)(inputs)


def batch_apply(func, inputs):
    """Apply function func on each element of inputs along the batch axis."""
    return layers.ApplyAlongAxis(func, axis=0)(inputs)
