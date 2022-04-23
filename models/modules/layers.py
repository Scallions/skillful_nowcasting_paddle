"""Implementation of miscellaneous DL operations.

Implementation of many of the layers can be found in the Sonnet library.
https://github.com/deepmind/sonnet
"""

# import tensorflow.compat.v1 as tf

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def downsample_avg_pool(x):
    """Utility function for downsampling by 2x2 average pooling."""
    # return tf.layers.average_pooling2d(x, 2, 2, data_format='channels_last')
    return F.avg_pool2d(x, 2, 2, data_format="NCHW")


def downsample_avg_pool3d(x):
    """Utility function for downsampling by 2x2 average pooling."""
    # return tf.layers.average_pooling3d(x, 2, 2, data_format='channels_last')
    return F.avg_pool3d(x, 2, 2, data_format="NCDHW")


def upsample_nearest_neighbor(inputs, upsample_size):
    """Nearest neighbor upsampling.

    Args:
      inputs: inputs of size [b, h, w, c] where b is the batch size, h the height,
        w the width, and c the number of channels.
      upsample_size: upsample size S.
    Returns:
      outputs: nearest neighbor upsampled inputs of size [b, s * h, s * w, c].
    """
    # del inputs
    # del upsample_size
    # # TO BE IMPLEMENTED
    # # One possible implementation could use tf.image.resize.
    # return []
    return F.interpolate(inputs, scale_factor=upsample_size, mode="nearest")


class Conv2D(nn.Layer):
    """2D convolution."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, rate=1,
                 padding='SAME', use_bias=True):
        """Constructor."""
        super().__init__()
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.padding = padding
        # self.initializer = tf.orthogonal_initializer
        self.use_bias = use_bias
        self.conv = nn.Conv2D(
          in_channels=input_channels,
          out_channels=output_channels,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          bias_attr=use_bias,
        )

    def forward(self, tensor):
        # TO BE IMPLEMENTED
        # One possible implementation is provided in the Sonnet library: snt.Conv2D.
        return self.conv(tensor)


class SNConv2D(nn.Layer):
    """2D convolution with spectral normalisation."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, rate=1,
                 padding='SAME', sn_eps=0.0001, use_bias=True):
        """Constructor."""
        super().__init__()
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.padding = padding
        self.sn_eps = sn_eps
        # self.initializer = tf.orthogonal_initializer
        self.use_bias = use_bias
        self.conv = nn.utils.spectral_norm(nn.Conv2D(
          in_channels=input_channels,
          out_channels=output_channels,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          bias_attr=use_bias,
        ),eps=self.sn_eps)

    def forward(self, tensor):
        # TO BE IMPLEMENTED
        # One possible implementation is provided using the Sonnet library as:
        # SNConv2D = snt.wrap_with_spectral_norm(snt.Conv2D, {'eps': 1e-4})
        return self.conv(tensor)


class SNConv3D(nn.Layer):
    """2D convolution with spectral regularisation."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, rate=1,
                 padding='SAME', sn_eps=0.0001, use_bias=True):
        """Constructor."""
        super().__init__()
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.padding = padding
        self.sn_eps = sn_eps
        self.use_bias = use_bias
        self.conv = nn.utils.spectral_norm(nn.Conv3D(
          in_channels=input_channels,
          out_channels=output_channels,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          bias_attr=use_bias,
        ),eps=self.sn_eps)

    def forward(self, tensor):
        # TO BE IMPLEMENTED
        # One possible implementation is provided using the Sonnet library as:
        # SNConv3D = snt.wrap_with_spectral_norm(snt.Conv3D, {'eps': 1e-4})
        return self.conv(tensor)


class Linear(nn.Layer):
    """Simple linear layer.

    Linear map from [batch_size, input_size] -> [batch_size, output_size].
    """

    def __init__(self, input_size, output_size):
        """Constructor."""
        super().__init__()
        # self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, tensor):
        # TO BE IMPLEMENTED
        # One possible implementation is provided in the Sonnet library: snt.Linear.
        # pass
        return self.linear(tensor)


class BatchNorm(nn.Layer):
    """Batch normalization."""

    def __init__(self, num_channels, calc_sigma=True):
        """Constructor."""
        super().__init__()
        self.calc_sigma = calc_sigma
        self.bn = nn.BatchNorm(num_channels=num_channels)

    def forward(self, tensor):
        # TO BE IMPLEMENTED
        # One possible implementation is provided in the Sonnet library:
        # snt.BatchNorm.
        return self.bn(tensor)


class ApplyAlongAxis(nn.Layer):
    """Layer for applying an operation on each element, along a specified axis."""

    def __init__(self, operation, axis=0):
        """Constructor."""
        super().__init__()
        self.operation = operation
        self.axis = axis

    def forward(self, *args):
        """Apply the operation to each element of args along the specified axis."""
        split_inputs = [paddle.unstack(arg, axis=self.axis) for arg in args]
        res = [self.operation(*x) for x in zip(*split_inputs)]
        return paddle.stack(res, axis=self.axis)
