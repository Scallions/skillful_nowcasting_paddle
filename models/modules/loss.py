import paddle
import paddle.nn.functional as F


def loss_hinge_disc(score_generated, score_real):
  """Discriminator hinge loss."""
  l1 = F.relu(1. - score_real)
  loss = paddle.mean(l1)
  l2 = F.relu(1. + score_generated)
  loss += paddle.mean(l2)
  return loss


def loss_hinge_gen(score_generated):
  """Generator hinge loss."""
  loss = -paddle.mean(score_generated)
  return loss


def grid_cell_regularizer(generated_samples, batch_targets):
  """Grid cell regularizer.

  Args:
    generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
    batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

  Returns:
    loss: A tensor of shape [batch_size].
  """
  gen_mean = paddle.mean(generated_samples, axis=0)
  weights = paddle.clip(batch_targets, 0.0, 24.0)
  loss = paddle.mean(paddle.abs(gen_mean - batch_targets) * weights)
  return loss
