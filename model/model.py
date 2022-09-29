import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Basic transformer model based off of the model specified in the paper Attention is all you need.
This model will pull a lot of inspiration from https://github.com/karpathy/minGPT/tree/master implementation.
"""

class TransformerConfiguaration:
    pass

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super(CasualSelfAttention, self).__init__()
        assert config.n_embed % config.n_heads == 0

        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        

        self.n_heads = config.n_heads
        self.n_embed = config.n_embed


class SelfAttention(nn.Module):
    pass

class EncoderBlock(nn.Module):
    pass

class DecoderBlock(nn.Module):
    pass

class Encoder(nn.Module):
    pass

class Decoder(nn.Module):
    pass

class Transformer(nn.Module):
    pass 

