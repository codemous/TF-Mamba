import torch
from torch import nn
from models.bert import BertTextEncoder
from einops import rearrange, repeat
from models.tmm import EnhanceSubNet
from models.mamba import TCMamba,TQMamba,Crossattn
class TFMamba(nn.Module):
    def __init__(self, args):
        super(TFMamba, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args['model']['feature_extractor']['bert_pretrained'])


        #input seq t a v
        # TME
        self.text_modality_mixup = EnhanceSubNet(
            input_length=args['model']['tmm']['input_length'],
            input_dim=args['model']['tmm']['input_dim'],
            hidden_dim=args['model']['tmm']['hidden_dim'])
        # feature reconstruction
        self.recon_text_low = nn.Sequential(
            nn.Linear(args['model']['tmr']['input_dim_high'], args['model']['tmr']['input_dim_high']),
            nn.ReLU(),
            nn.Dropout(args['model']['tmr']['dropout']),
            nn.Linear(args['model']['tmr']['input_dim_high'], args['model']['tmr']['input_dim_low'])
        )
        #TC-Mamba
        self.text_based_context_mamba = TCMamba(
            num_layers=args['model']['tc_mamba']['num_layers'],
            d_model=args['model']['tc_mamba']['d_model'],
            d_ffn=args['model']['tc_mamba']['d_model'] * 4,
            activation=args['model']['tc_mamba']['activation'],
            dropout=args['model']['tc_mamba']['dropout'],
            causal=args['model']['tc_mamba']['causal'],
            mamba_config=args['model']['tc_mamba']['mamba_config']
        )
        #TQ-Mamba
        self.text_guided_attention = Crossattn(
            num_heads=args['model']['tq_mamba']['attn_heads'],
            d_model=args['model']['tq_mamba']['d_model'],

        )
        self.text_based_query_mamba = TQMamba(
            num_layers=args['model']['tq_mamba']['num_layers'],
            d_model=args['model']['tq_mamba']['d_model'],
            d_ffn=args['model']['tq_mamba']['d_model'] * 4,
            activation=args['model']['tq_mamba']['activation'],
            dropout=args['model']['tq_mamba']['dropout'],
            causal=args['model']['tq_mamba']['causal'],
            mamba_config=args['model']['tq_mamba']['mamba_config']
        )


        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(args['model']['regression']['input_dim'], args['model']['regression']['out_dim'])

    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        b = vision_m.size(0)

        h_0_v = vision_m
        h_0_a = audio_m
        h_0_t = self.bertmodel(language_m)
        # text-aware mixup #t v a
        h_tmm_t, h_tmm_v, h_tmm_a = self.text_modality_mixup(h_0_t,h_0_v,h_0_a)

        # tc-mamba a v t
        h_tc_mamba_a, h_tc_mamba_v, h_tc_mamba_t = self.text_based_context_mamba(h_tmm_a,h_tmm_v,h_tmm_t)

        # tq-mamaba
        h_tm_attn = self.text_guided_attention(h_tc_mamba_t,torch.cat([h_tc_mamba_a,h_tc_mamba_v],dim=1))
        h_tm_mamba = self.text_based_query_mamba(h_tm_attn)

        #regression
        h_m_pool = self.pool(h_tm_mamba.permute(0,2,1)).squeeze(-1)
        output = self.output(h_m_pool)

        rec_text_feats, com_text_feats = None, None
        if (vision is not None) and (audio is not None) and (language is not None):
        #text modal recon
            h_t_o = self.bertmodel(language)
            text_recon_low = self.recon_text_low(h_tmm_t)
            rec_text_feats= [text_recon_low]
            com_text_feats = [h_t_o]

        return {'sentiment_preds': output,
                'rec_text': rec_text_feats,
                'complete_text': com_text_feats}




def build_model(args):
    return TFMamba(args)