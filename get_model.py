''' Extract encoder and decoder models from dreamer model. '''

from torch import nn
import torch
import mlflow



from distutils.util import strtobool
import argparse
from pydreamer.models import *
import pydreamer.models.encoders as encode
import pydreamer.models.decoders as decode
from pydreamer.tools import *
from pydreamer import tools

import numpy as np

import argparse
import logging
import logging.config
import os
import sys
import time
from distutils.util import strtobool
from collections import defaultdict
from datetime import datetime
from itertools import chain
from logging import critical, debug, error, info, warning
from multiprocessing import Process
from pathlib import Path
from typing import Iterator, Optional

import mlflow
import numpy as np
import scipy.special
import torch
import torch.distributions as D
import torch.nn as nn
from torch import Tensor, tensor
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader

import generator
from pydreamer import tools
from pydreamer.data import DataSequential, MlflowEpisodeRepository
from pydreamer.models import *
from pydreamer.models.functions import map_structure, nanmean
from pydreamer.preprocessing import Preprocessor, WorkerInfoPreprocess
from pydreamer.tools import *


def to_image(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        x = x.astype(np.float32)
        x = x / 255.0 - 0.5  # type: ignore
    else:
        assert 0.0 <= x[0, 0, 0, 0, 0] and x[0, 0, 0, 0, 0] <= 1.0
        x = x.astype(np.float32)
    x = x.transpose(0, 1, 4, 2, 3)  # (T, B, H, W, C) => (T, B, C, H, W)
    return x


def load_encoder(conf, checkpoint):
    # load encoder model

    # these are the names of the weights that the encoder needs
    encoder_keys = ["encoder_image.model.0.weight", "encoder_image.model.0.bias", "encoder_image.model.2.weight", "encoder_image.model.2.bias", "encoder_image.model.4.weight", "encoder_image.model.4.bias", "encoder_image.model.6.weight", "encoder_image.model.6.bias"]
    
    encoder = encode.MultiEncoder(conf)
    
    # Checkpoint has these weights 
    encoder_weights = dict((key[11:], checkpoint['model_state_dict'][key]) for key in checkpoint['model_state_dict'] if key[11:] in encoder_keys)
    
    # print(encoder_weights)
    encoder.load_state_dict(encoder_weights)

    return encoder

def load_decoder(conf, checkpoint):
    # load decoder model

    decoder_keys = ["image.model.0.weight", "image.model.0.bias", "image.model.2.weight", "image.model.2.bias", "image.model.4.weight", "image.model.4.bias", "image.model.6.weight", "image.model.6.bias", "image.model.8.weight", "image.model.8.bias", "image.model.10.weight", "image.model.10.bias", "reward.model.model.0.weight", "reward.model.model.0.bias", "reward.model.model.1.weight", "reward.model.model.1.bias", "reward.model.model.3.weight", "reward.model.model.3.bias", "reward.model.model.4.weight", "reward.model.model.4.bias", "reward.model.model.6.weight", "reward.model.model.6.bias", "reward.model.model.7.weight", "reward.model.model.7.bias", "reward.model.model.9.weight", "reward.model.model.9.bias", "reward.model.model.10.weight", "reward.model.model.10.bias", "reward.model.model.12.weight", "reward.model.model.12.bias", "terminal.model.model.0.weight", "terminal.model.model.0.bias", "terminal.model.model.1.weight", "terminal.model.model.1.bias", "terminal.model.model.3.weight", "terminal.model.model.3.bias", "terminal.model.model.4.weight", "terminal.model.model.4.bias", "terminal.model.model.6.weight", "terminal.model.model.6.bias", "terminal.model.model.7.weight", "terminal.model.model.7.bias", "terminal.model.model.9.weight", "terminal.model.model.9.bias", "terminal.model.model.10.weight", "terminal.model.model.10.bias", "terminal.model.model.12.weight", "terminal.model.model.12.bias"]
    
    decoder = decode.MultiDecoder(2048, conf)
    
    # Checkpoint has these weights 
    decoder_weights = dict((key[11:], checkpoint['model_state_dict'][key]) for key in checkpoint['model_state_dict'] if key[11:] in decoder_keys)
    
    # print(decoder_weights)
    decoder.load_state_dict(decoder_weights)

    return decoder

def run(conf):

    # load encoder/decoder using run
    PATH = 'mlruns/0/d8f95e5302d84d68a3f483e5e3c22882/artifacts/checkpoints/latest.pt'

    model = Dreamer(conf)
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # print([key for key in checkpoint['model_state_dict'].keys() if 'encoder' in key])

    encoder = load_encoder(conf, checkpoint)
    decoder = load_decoder(conf, checkpoint)

    # save models in models folder
    torch.save(encoder.state_dict(), 'models/encoder.pt')
    torch.save(decoder.state_dict(), 'models/decoder.pt')

    # # load encoder/decoder from saved pt (saved using code above)
    # encoder = encode.MultiEncoder(conf)
    # encoder.load_state_dict(torch.load('models/encoder.pt'))

    # decoder = decode.MultiDecoder(2048, conf)
    # decoder.load_state_dict(torch.load('models/decoder.pt'))


    # some unfinished stuff for closed loop image encoding/decoding

    # batch = np.load('image_t.npy')
    # print(batch.shape)

    # preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
    #                           image_key=conf.image_key,
    #                           map_categorical=conf.map_channels if conf.map_categorical else None,
    #                           map_key=conf.map_key,
    #                           action_dim=conf.action_dim,
    #                           clip_rewards=conf.clip_rewards,
    #                           amp=conf.device.startswith('cuda') and conf.amp)


    # batch = preprocess(batch)
    # obs: Dict[str, Tensor] = map_structure(batch, lambda x: x.to('cpu'))


    
    # im = {'reward' : np.array([0]), 'terminal' : np.array([0])}
    # obs = {}
    # obs['image'] = torch.from_numpy(to_image(np.expand_dims(batch, axis=0).transpose(0,4,1,2,3)))
    # print(obs['image'].shape)

    # # im['image'] = torch.from_numpy(np.expand_dims(batch, axis=0).transpose(4,0,3,1,2))

    # print(im['image'].shape)

    # init_state = model.wm.init_state(batch.shape[3])
    # loss, features, states, out_state, metrics, tensors = model.wm.forward(obs, init_state)








if __name__ == "__main__":
    configure_logging(prefix='[TRAIN]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = tools.read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    conf = parser.parse_args(remaining)


    run(conf)