import paddle
import paddle.nn as nn

from dall_e.encoder import Encoder
from dall_e.decoder import Decoder
from dall_e.utils   import map_pixels, unmap_pixels


def load_model(path: str) -> nn.Layer:
    state_dict = paddle.load(path)
    if 'encoder' in path.lower():
        model = Encoder()
    elif 'decoder' in path.lower():
        model = Decoder()
    else:
        raise ValueError
    model.set_state_dict(state_dict)
    return model
