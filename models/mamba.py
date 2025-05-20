import warnings
from dataclasses import dataclass
from typing import List, Optional

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
#
from models.mamba_nets.attention import Attention
# Mamba
from mamba_ssm import Mamba
from models.mamba_nets.bimamba import Mamba as BiMamba
from models.mamba_nets.mm_bimamba import Mamba as MMBiMamba


class MMMambaEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            d_ffn,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        # if activation == 'Swish':
        #     activation = Swish
        # elif activation == "GELU":
        #     activation = torch.nn.GELU
        # else:
        #     activation = Swish

        bidirectional = mamba_config.pop('bidirectional')

        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )

        mamba_config['bidirectional'] = bidirectional

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(
            self,
            a_x, v_x,
            a_inference_params=None,
            v_inference_params=None
    ):

        a_out1, v_out1 = self.mamba(a_x, v_x, a_inference_params, v_inference_params)
        a_out = a_x + self.norm1(a_out1)
        v_out = v_x + self.norm2(v_out1)

        return a_out, v_out


class MambaEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            d_ffn,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        # if activation == 'Swish':
        #     activation = Swish
        # elif activation == "GELU":
        #     activation = torch.nn.GELU
        # else:
        #     activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(
            self,
            x, inference_params=None
    ):
        out = x + self.norm1(self.mamba(x, inference_params))
        return out


class TCMamba(nn.Module):


    def __init__(
            self,
            num_layers,
            d_model,
            d_ffn=1024,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        at_mamba_list = []
        vt_mamba_list = []
        # print(output_sizes)
        for i in range(num_layers):
            at_mamba_list.append(MMMambaEncoderLayer(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))
            vt_mamba_list.append(MMMambaEncoderLayer(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.at_mamba_layers = torch.nn.ModuleList(at_mamba_list)
        self.vt_mamba_layers = torch.nn.ModuleList(vt_mamba_list)


    def forward(
            self,
            a_x, v_x, t_x,
            a_inference_params=None,
            v_inference_params=None,
            t_inference_params=None
    ):
        a_out = a_x
        v_out = v_x
        t_out = t_x

        for at_mamba_layer, vt_mamba_layer in zip(self.at_mamba_layers, self.vt_mamba_layers):
            a_out, t_out_at = at_mamba_layer(
                a_out, t_out,
                a_inference_params,
                t_inference_params
            )
            v_out, t_out_vt = vt_mamba_layer(
                v_out, t_out,
                v_inference_params,
                t_inference_params
            )
            t_out = (t_out_at+t_out_vt)/2

        return a_out, v_out,t_out


class TQMamba(nn.Module):

    def __init__(
            self,
            num_layers,
            d_model,
            d_ffn=1024,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')

        mamba_list = []
        # print(output_sizes)
        for i in range(num_layers):

            mamba_list.append(MambaEncoderLayer(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)

    def forward(
            self,
            x,
            inference_params=None,
    ):
        out = x

        for mamba_layer in  self.mamba_layers:
            out = mamba_layer(
                out,
                inference_params=inference_params,
            )

        return out

class Crossattn(nn.Module):


    def __init__(
            self,
            num_heads,
            d_model,
    ):
        super().__init__()
        self.cross_attention = Attention(dim=d_model,heads=num_heads)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        # print(output_sizes)


    def forward(
            self,
            x_q, x_kv
    ):
        out_attn = self.cross_attention(x_q,x_kv,x_kv)
        out =  x_q + self.norm(out_attn)

        return out