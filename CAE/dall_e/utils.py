import attr
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

logit_laplace_eps: float = 0.1

@attr.s(eq=False)
class Conv2D(nn.Layer):
	n_in:  int = attr.ib(validator=lambda i, a, x: x >= 1)
	n_out: int = attr.ib(validator=lambda i, a, x: x >= 1)
	kw:    int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 2 == 1)

	use_float16:   bool         = attr.ib(default=True)
	requires_grad: bool         = attr.ib(default=False)

	def __attrs_post_init__(self) -> None:
		super().__init__()

		self.w = self.create_parameter([self.n_out, self.n_in, self.kw, self.kw], dtype=paddle.float32,
			default_initializer=nn.initializer.Normal(std=1 / math.sqrt(self.n_in * self.kw ** 2)))
		self.w.stop_gradient = not self.requires_grad

		self.b = self.create_parameter([self.n_out], dtype=paddle.float32,
			default_initializer=nn.initializer.Constant(value=0))
		self.b.stop_gradient = not self.requires_grad

	def forward(self, x: paddle.Tensor) -> paddle.Tensor:
		if self.use_float16:
			if x.dtype != paddle.float16:
				x = x.astype(paddle.float16)

			w, b = self.w.astype(paddle.float16), self.b.astype(paddle.float16)
		else:
			if x.dtype != paddle.float32:
				x = x.astype(paddle.float32)

			w, b = self.w, self.b

		return F.conv2d(x, w, bias=b, padding=(self.kw - 1) // 2)

def map_pixels(x: paddle.Tensor) -> paddle.Tensor:
	if x.dtype != paddle.float32:
		raise ValueError('expected input to have type float')

	return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def unmap_pixels(x: paddle.Tensor) -> paddle.Tensor:
	if len(x.shape) != 4:
		raise ValueError('expected input to be 4d')
	if x.dtype != paddle.float32:
		raise ValueError('expected input to have type float')

	return paddle.clip((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)
