"""Implementation of miscellaneous DL operations.

Implementation of many of the layers can be found in the Sonnet library.
https://github.com/deepmind/sonnet
"""

import tensorflow.compat.v1 as tf


def downsample_avg_pool(x):
  """Utility function for downsampling by 2x2 average pooling."""
  return tf.layers.average_pooling2d(x, 2, 2, data_format='channels_last')


def downsample_avg_pool3d(x):
  """Utility function for downsampling by 2x2 average pooling."""
  return tf.layers.average_pooling3d(x, 2, 2, data_format='channels_last')


def upsample_nearest_neighbor(inputs, upsample_size):
  """Nearest neighbor upsampling.

  Args:
    inputs: inputs of size [b, h, w, c] where b is the batch size, h the height,
      w the width, and c the number of channels.
    upsample_size: upsample size S.
  Returns:
    outputs: nearest neighbor upsampled inputs of size [b, s * h, s * w, c].
  """
  del inputs
  del upsample_size
  # TO BE IMPLEMENTED
  # One possible implementation could use tf.image.resize.
  return []


class Conv2D:
  """2D convolution."""

  def __init__(self, output_channels, kernel_size, stride=1, rate=1,
               padding='SAME', use_bias=True):
    """Constructor."""
    self._output_channels = output_channels
    self._kernel_size = kernel_size
    self._stride = stride
    self._rate = rate
    self._padding = padding
    self._initializer = tf.orthogonal_initializer
    self._use_bias = use_bias

  def __call__(self, tensor):
    # TO BE IMPLEMENTED
    # One possible implementation is provided in the Sonnet library: snt.Conv2D.
    pass


class SNConv2D:
  """2D convolution with spectral normalisation."""

  def __init__(self, output_channels, kernel_size, stride=1, rate=1,
               padding='SAME', sn_eps=0.0001, use_bias=True):
    """Constructor."""
    self._output_channels = output_channels
    self._kernel_size = kernel_size
    self._stride = stride
    self._rate = rate
    self._padding = padding
    self._sn_eps = sn_eps
    self._initializer = tf.orthogonal_initializer
    self._use_bias = use_bias

  def __call__(self, tensor):
    # TO BE IMPLEMENTED
    # One possible implementation is provided using the Sonnet library as:
    # SNConv2D = snt.wrap_with_spectral_norm(snt.Conv2D, {'eps': 1e-4})
    pass


class SNConv3D:
  """2D convolution with spectral regularisation."""

  def __init__(self, output_channels, kernel_size, stride=1, rate=1,
               padding='SAME', sn_eps=0.0001, use_bias=True):
    """Constructor."""
    self._output_channels = output_channels
    self._kernel_size = kernel_size
    self._stride = stride
    self._rate = rate
    self._padding = padding
    self._sn_eps = sn_eps
    self._use_bias = use_bias

  def __call__(self, tensor):
    # TO BE IMPLEMENTED
    # One possible implementation is provided using the Sonnet library as:
    # SNConv3D = snt.wrap_with_spectral_norm(snt.Conv3D, {'eps': 1e-4})
    pass


class Linear:
  """Simple linear layer.

  Linear map from [batch_size, input_size] -> [batch_size, output_size].
  """

  def __init__(self, output_size):
    """Constructor."""
    self._output_size = output_size

  def __call__(self, tensor):
    # TO BE IMPLEMENTED
    # One possible implementation is provided in the Sonnet library: snt.Linear.
    pass


class BatchNorm:
  """Batch normalization."""

  def __init__(self, calc_sigma=True):
    """Constructor."""
    self._calc_sigma = calc_sigma

  def __call__(self, tensor):
    # TO BE IMPLEMENTED
    # One possible implementation is provided in the Sonnet library:
    # snt.BatchNorm.
    pass


class ApplyAlongAxis:
  """Layer for applying an operation on each element, along a specified axis."""

  def __init__(self, operation, axis=0):
    """Constructor."""
    self._operation = operation
    self._axis = axis

  def __call__(self, *args):
    """Apply the operation to each element of args along the specified axis."""
    split_inputs = [tf.unstack(arg, axis=self._axis) for arg in args]
    res = [self._operation(x) for x in zip(*split_inputs)]
    return tf.stack(res, axis=self._axis)
