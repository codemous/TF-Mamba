import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

__all__ = ['CTCModule', 'EnhanceSubNet']


class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len + 1, num_layers=2,
                                                        batch_first=True)  # 1 denoting blank
        self.out_seq_len = out_seq_len

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank.

        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(
            pred_output_position_inclu_blank)  # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :,
                                    1:]  # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1, 2)  # batch_size x out_seq_len x in_seq_len
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x)  # batch_size x out_seq_len x in_dim

        return pseudo_aligned_out

class EnhanceSubNet(nn.Module): # text-aware modality enhancement
    def __init__(self, input_length, input_dim, hidden_dim):

        super(EnhanceSubNet, self).__init__()

        seq_len_t, seq_len_v, seq_len_a = input_length[0], input_length[1], input_length[2]
        in_dim_t, in_dim_v, in_dim_a = input_dim[0], input_dim[1], input_dim[2]

        self.dst_len = seq_len_t

        self.dst_dim = hidden_dim

        self.eps = 1e-9 #

        self.ctc_vt = CTCModule(in_dim_v, self.dst_len)
        self.logit_scale_vt = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.ctc_at = CTCModule(in_dim_a, self.dst_len)
        self.logit_scale_at = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        self.proj_a = nn.Sequential(
            nn.LayerNorm(in_dim_a, eps=1e-6),
            nn.Linear(in_dim_a, self.dst_dim),
            nn.LayerNorm(self.dst_dim,eps=1e-6),
        )

        self.proj_v = nn.Sequential(
            nn.LayerNorm(in_dim_v, eps=1e-6),
            nn.Linear(in_dim_v, self.dst_dim),
            nn.LayerNorm(self.dst_dim, eps=1e-6),
        )

        self.proj_t = nn.Sequential(
            nn.LayerNorm(in_dim_t, eps=1e-6),
            nn.Linear(in_dim_t, self.dst_dim),
        )

    def get_seq_len(self):
        return self.dst_len

    def forward(self, text_x, video_x, audio_x):
        #
        pseudo_video = self.ctc_vt(video_x) # [B, out_seq_len, in_dim_v]
        pseudo_audio = self.ctc_at(audio_x) # [B, out_seq_len, in_dim_a]

        v_common = self.proj_v(pseudo_video)  # [B, out_seq_len, com_dim]
        v_n = v_common.norm(dim=-1, keepdim=True)
        v_norm = v_common / torch.max(v_n, self.eps * torch.ones_like(v_n))  # [B, out_seq_len, shared_dim]

        t_common = self.proj_t(text_x)  # [B, out_seq_len, shared_dim]
        t_n = t_common.norm(dim=-1, keepdim=True)
        t_norm = t_common / torch.max(t_n, self.eps * torch.ones_like(t_n))  # [B, out_seq_len, shared_dim]

        a_common = self.proj_a(pseudo_audio)  # [B, out_seq_len, shared_dim]
        a_n = a_common.norm(dim=-1, keepdim=True)
        a_norm = a_common / torch.max(a_n, self.eps * torch.ones_like(a_n))  # [B, out_seq_len, shared_dim]

        # vt cosine similarity as logits
        logit_scale_vt = self.logit_scale_vt.exp()
        similarity_matrix_vt = logit_scale_vt * torch.bmm(v_norm, t_norm.permute(0, 2, 1)) #
        logits_vt = similarity_matrix_vt.softmax(dim=-1) #
        mask_vt = (logits_vt > (1 / self.dst_len)).float() #[B, out_seq_len, out_seq_len]

        # at cosine similarity as logits
        logit_scale_at = self.logit_scale_at.exp()
        similarity_matrix_at = logit_scale_at * torch.bmm(a_norm, t_norm.permute(0, 2, 1)) #
        logits_at = similarity_matrix_at.softmax(dim=-1) #
        mask_at = (logits_at > (1 / self.dst_len)).float() #[B, out_seq_len, out_seq_len]

        video_out = v_common + torch.bmm(mask_vt*logits_vt, t_common)
        audio_out = a_common + torch.bmm(mask_at*logits_at, t_common)
        text_out = t_common #
        return text_out, video_out, audio_out
