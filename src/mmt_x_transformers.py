import argparse
import logging
import pathlib
import pprint
import sys
import time

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from x_transformers.autoregressive_wrapper import (
    exists,
    top_a,
    top_k,
    top_p,
)
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Decoder,
    Encoder,
    TokenEmbedding,
    always,
    default,
    exists,
)
sys.path.append('..')
import representation as representation
import utils as utils
import numpy as np
import copy

EPSILON = np.finfo(np.float32).tiny


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def pad(data, maxlen=None, value=0):
    if maxlen is None:
        max_len = max(len(x) for x in data)
    else:
        for x in data:
            assert len(x) <= max_len
    if data[0].ndim == 1:
        padded = [F.pad(x, (0, max_len - len(x)), value=value) for x in data]
        # padded = [np.pad(x, (0, max_len*2 - len(x))) for x in data]
    elif data[0].ndim == 2:
        padded = [F.pad(x, (0, 0, 0, max_len - len(x)), value=value) for x in data]
    else:
        raise ValueError("Got 3D data.")
    return torch.stack(padded)

def get_mask(data):
    max_seq_len = max(len(sample) for sample in data)
    mask = torch.zeros((len(data), max_seq_len), dtype=torch.bool)
    for i, seq in enumerate(data):
        mask[i, : len(seq)] = 1
    return mask


def extract_skyline_with_mask(inp, inD, ix, ich=False, ich_dens=1):
    new_inp = torch.zeros((len(inD['onsetdict'][ix])*2 + 25, inp.shape[-1])).to(device)
    idx = 0
    n_beats = 0
    cur_chord = None
    for onset in sorted(inD['onsetdict'][ix]):
        mask = sorted(inD['onsetdict'][ix][onset])
        cur_chord = inp[mask[0]]

        if onset[0] == 1: # instrument selection, we need all of them
            new_inp[idx:idx+len(mask)] = inp[mask]
            idx += len(mask)
        else:
            if ich and cur_chord[0] == 5 and cur_chord[1] > 0 and (cur_chord[1] - 1) % ich_dens == 0:
                new_inp[idx] = cur_chord
                idx += 1
                n_beats += 1

            if ich and inp[mask[-1]][0] == 5:
                continue
            
            new_inp[idx] = inp[mask[-1]]
            idx += 1

    return new_inp.long()[:idx], n_beats


def extract_skylinemask(inD, ix, ich=False, ich_dens=1):
    all_masks = []
    cur_chord = None
    for osidx, onset in enumerate(sorted(inD['old_onsets'][ix])):
        mask = sorted(inD['onsetdict'][ix][onset])
        if ich:
            cur_chord = inD['chord_seq'][ix, mask[0]]
        if onset[0] == 1: # instrument selection, we need all of them
            all_masks.append(torch.tensor(mask))
        else:
            if ich and cur_chord[0] == 5 and cur_chord[1] > 0 and (cur_chord[1] - 1) % ich_dens == 0:
                all_masks.append(torch.tensor(mask[0]).reshape(1))
            
            if ich and inD['chord_seq'][ix, mask[-1]][0] == 5:
                continue

            all_masks.append(torch.tensor(mask[-1]).reshape(1))
    all_masks = torch.cat(all_masks)
    out = torch.zeros((inD['chord_seq_len'][ix] if ich else inD['seq_len'][ix], 1)).to(device)
    out[all_masks] = 1
    return out


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", 'C3'),
        required=True,
        help="dataset key",
    )
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


class MusicTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        encoding,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_beat=None,
        max_mem_len=0.0,
        shift_mem_down=0,
        emb_dropout=0.0,
        num_memory_tokens=None,
        tie_embedding=False,
        use_abs_pos_emb=True,
        l2norm_embed=False,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        n_tokens = copy.copy(encoding["n_tokens"])
        if max_beat is not None:
            beat_dim = encoding["dimensions"].index("beat")
            n_tokens[beat_dim] = max_beat + 1

        

        self.l2norm_embed = l2norm_embed
        self.token_emb = nn.ModuleList(
            [
                TokenEmbedding(emb_dim, n, l2norm_embed=l2norm_embed)
                for n in n_tokens
            ]
        )
        self.pos_emb = (
            AbsolutePositionalEmbedding(
                emb_dim, max_seq_len, l2norm_embed=l2norm_embed
            )
            if (use_abs_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = (
            nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        )
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = (
            nn.ModuleList([nn.Linear(dim, n) for n in n_tokens[:6]])
        )
        if tie_embedding:
            for emb, to_logits in zip(self.token_emb[:6], self.to_logits):
                to_logits.weight = emb.emb.weight

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(num_memory_tokens, dim)
            )

    def init_(self):
        if self.l2norm_embed:
            for emb in self.token_emb:
                nn.init.normal_(emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return

        for emb in self.token_emb:
            nn.init.kaiming_normal_(emb.emb.weight)

    def forward(
        self,
        x,  # shape : (b, n , d)
        return_embeddings=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        score_in=None,
        **kwargs,
    ):
        num_mem = self.num_memory_tokens
        if type(x) == list:
            b = x[0].shape[0]
            x = sum(
                x[i] @ emb.emb.weight for i, emb in enumerate(self.token_emb)
            )
        else:
            b, _, _ = x.shape
            x = sum(
                emb(x[..., i]) for i, emb in enumerate(self.token_emb)
            )

        x = x + self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
            x = torch.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = (
                mems[: self.shift_mem_down],
                mems[self.shift_mem_down :],
            )
            mems = [*mems_r, *mems_l]

        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, return_hiddens=True, **kwargs
        )
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = (
            [to_logit(x) for to_logit in self.to_logits]
            if not return_embeddings
            else x
        )

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(
                    map(
                        lambda pair: torch.cat(pair, dim=-2),
                        zip(mems, hiddens),
                    )
                )
                if exists(mems)
                else hiddens
            )
            new_mems = list(
                map(
                    lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems
                )
            )
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(
                    lambda t: t.post_softmax_attn,
                    intermediates.attn_intermediates,
                )
            )
            return out, attn_maps

        return out


def sample(logits, kind, threshold, temperature, min_p_pow, min_p_ratio):
    """Sample from the logits with a specific sampling strategy."""
    if logits.shape[-1] == 6:
        logits = logits[..., :5]
    elif logits.shape[-1] == 143:
        logits = logits[..., :130]
    elif logits.shape[-1] == 46:
        logits = logits[..., :33]
    if kind == "top_k":
        probs = F.softmax(top_k(logits, thres=threshold) / temperature, dim=-1)
    elif kind == "top_p":
        probs = F.softmax(top_p(logits, thres=threshold) / temperature, dim=-1)
    elif kind == "top_a":
        probs = F.softmax(
            top_a(logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio)
            / temperature,
            dim=-1,
        )
    elif kind == "entmax":
        probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)
    else:
        raise ValueError(f"Unknown sampling strategy: {kind}")

    out = torch.multinomial(probs, 1)
    return out


class MusicAutoregressiveWrapper(nn.Module):
    def __init__(self, net, encoding, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # Get the type codes
        self.sos_type_code = encoding["type_code_map"]["start-of-song"]
        self.eos_type_code = encoding["type_code_map"]["end-of-song"]
        self.son_type_code = encoding["type_code_map"]["start-of-notes"]
        self.instrument_type_code = encoding["type_code_map"]["instrument"]
        self.note_type_code = encoding["type_code_map"]["note"]

        # Get the dimension indices
        self.dimensions = {
            key: encoding["dimensions"].index(key)
            for key in (
                "type",
                "beat",
                "position",
                "pitch",
                "duration",
                "instrument",
            )
        }
        assert self.dimensions["type"] == 0

    @torch.no_grad()
    def generate(
        self,
        start_tokens,  # shape : (b, n, d)
        seq_len,
        eos_token=None,
        temperature=1.0,  # int or list of int
        filter_logits_fn="top_k",  # str or list of str
        filter_thres=0.8,  # int or list of int
        min_p_pow=2.0,
        min_p_ratio=0.02,
        monotonicity_dim=['type', 'beat'],
        return_attn=False,
        **kwargs,
    ):
        _, t, dim = start_tokens.shape

        if isinstance(temperature, (float, int)):
            temperature = [temperature] * dim
        elif len(temperature) == 1:
            temperature = temperature * dim
        else:
            assert (
                len(temperature) == dim
            ), f"`temperature` must be of length {dim}"

        if isinstance(filter_logits_fn, str):
            filter_logits_fn = [filter_logits_fn] * dim
        elif len(filter_logits_fn) == 1:
            filter_logits_fn = filter_logits_fn * dim
        else:
            assert (
                len(filter_logits_fn) == dim
            ), f"`filter_logits_fn` must be of length {dim}"

        if isinstance(filter_thres, (float, int)):
            filter_thres = [filter_thres] * dim
        elif len(filter_thres) == 1:
            filter_thres = filter_thres * dim
        else:
            assert (
                len(filter_thres) == dim
            ), f"`filter_thres` must be of length {dim}"

        if isinstance(monotonicity_dim, str):
            monotonicity_dim = [self.dimensions[monotonicity_dim]]
        else:
            monotonicity_dim = [self.dimensions[d] for d in monotonicity_dim]

        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 2:
            start_tokens = start_tokens[None, :, :]

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = torch.ones(
                (out.shape[0], out.shape[1]),
                dtype=torch.bool,
                device=out.device,
            )

        if monotonicity_dim is not None:
            current_values = {
                d: torch.max(start_tokens[:, :, d], 1)[0]
                for d in monotonicity_dim
            }
        else:
            current_values = None

        instrument_dim = self.dimensions["instrument"]
        type_dim = self.dimensions["type"]
        start_instrs = start_tokens[:, :, instrument_dim].unique()
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            mask = mask[:, -self.max_seq_len :]

            if return_attn:
                logits, attn = self.net(
                    x, mask=mask, return_attn=True, **kwargs
                )
                logits = [l[:, -1, :] for l in logits]
            else:
                logits = [
                    l[:, -1, :] for l in self.net(x, mask=mask, **kwargs)
                ]

            # Enforce monotonicity
            if monotonicity_dim is not None and 0 in monotonicity_dim:
                for i, v in enumerate(current_values[0]):
                    logits[0][i, :v] = -float("inf")

            # Filter out sos token
            logits[0][type_dim, 0] = -float("inf")

            # Sample from the logits
            sample_type = sample(
                logits[0],
                filter_logits_fn[0],
                filter_thres[0],
                temperature[0],
                min_p_pow,
                min_p_ratio,
            )

            # Update current values
            if monotonicity_dim is not None and 0 in monotonicity_dim:
                current_values[0] = torch.maximum(
                    current_values[0], sample_type.reshape(-1)
                )

            # Iterate after each sample
            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                # A start-of-song, end-of-song or start-of-notes code
                if s_type in (
                    self.sos_type_code,
                    self.eos_type_code,
                    self.son_type_code,
                ):
                    samples[idx] += [torch.zeros_like(s_type)] * (
                        len(logits) - 1
                    )
                # An instrument code
                elif s_type == self.instrument_type_code:
                    samples[idx] += [torch.zeros_like(s_type)] * (
                        len(logits) - 2
                    )
                    logits[instrument_dim][:, 0] = -float("inf")  # avoid none
                    sampled = sample(
                        logits[instrument_dim][idx : idx + 1],
                        filter_logits_fn[instrument_dim],
                        filter_thres[instrument_dim],
                        temperature[instrument_dim],
                        min_p_pow,
                        min_p_ratio,
                    )[0]
                    samples[idx].append(sampled)
                # A note code
                elif s_type == self.note_type_code:
                    for d in range(1, 6):
                        # Enforce monotonicity
                        if (
                            monotonicity_dim is not None
                            and d in monotonicity_dim
                        ):
                            logits[d][idx, : current_values[d][idx]] = -float(
                                "inf"
                            )

                        # Sample from the logits
                        logits[d][:, 0] = -float("inf")  # avoid none
                        if d == 5: # restrict sampling to start_instrs
                            tmp_logits = logits[d].clone()
                            logits[d][:, 1:] = -float("inf")
                            logits[d][:, start_instrs] = tmp_logits[:, start_instrs]

                        sampled = sample(
                            logits[d][idx : idx + 1],
                            filter_logits_fn[d],
                            filter_thres[d],
                            temperature[d],
                            min_p_pow,
                            min_p_ratio,
                        )[0]
                        samples[idx].append(sampled)

                        # Update current values
                        if (
                            monotonicity_dim is not None
                            and d in monotonicity_dim
                        ):
                            current_values[d][idx] = torch.max(
                                current_values[d][idx], sampled
                            )[0]
                else:
                    raise ValueError(f"Unknown event type code: {s_type}")

            stacked = torch.stack(
                [torch.cat(s).expand(1, -1) for s in samples], 0
            )
            out = torch.cat((out, stacked), dim=1)
            mask = F.pad(mask, (0, 1), value=True)

            if exists(eos_token):
                is_eos_tokens = out[..., 0] == eos_token

                # Mask out everything after the eos tokens
                if is_eos_tokens.any(dim=1).all():
                    for i, is_eos_token in enumerate(is_eos_tokens):
                        idx = torch.argmax(is_eos_token.byte())
                        out[i, idx + 1 :] = self.pad_value
                    break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)

        if return_attn:
            return out, attn

        return out

    def forward(self, x, tgt_mask, return_list=False, **kwargs):

        fields = [0,1,2,3,4,5]
        xi = x[:, :-1, fields]
        xo = x[:, 1:, fields]
        use_mask = tgt_mask[:, 1:]

        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs["mask"] = mask

        out = self.net(xi, **kwargs)
        losses = [
            F.cross_entropy(
                out[i].transpose(1, 2),
                xo[..., i],
                ignore_index=self.pad_value,
                reduction="mean",
            )
            for i in fields
        ]
        accs = [
            (out[i].argmax(dim=-1) == xo[..., i]).float() * use_mask.float()
            for i in fields
        ]
        if return_list:
            return losses, losses
        return losses, accs


class MusicXTransformer(nn.Module):
    def __init__(self, *, dim, encoding, encoder, decoder, use_pre=True, ich=False, **kwargs):
        super().__init__()
        assert "dim" not in kwargs, "dimension must be set with `dim` keyword"
        transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_beat": kwargs.pop("max_beat"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        self.encoding = encoding
        if ich:
            encoder_kwargs = copy.copy(transformer_kwargs)
            encoder_kwargs['max_seq_len'] += encoder_kwargs['max_beat']
        else:
            encoder_kwargs = transformer_kwargs
        self.encoder = MusicTransformerWrapper(
            encoding=new_encoding,
            tie_embedding=True,
            attn_layers=Encoder(dim=dim, **kwargs),
            **encoder_kwargs,
        )
        new_encoding = copy.copy(self.encoding)
        new_encoding['n_tokens'] = encoding['n_tokens'][:6]
        self.decoder = MusicTransformerWrapper(
            encoding=new_encoding,
            attn_layers=Decoder(dim=dim,  cross_attend = True, **kwargs),
            tie_embedding=True,
            **transformer_kwargs,
        )
        self.decoder = MusicAutoregressiveWrapper(
            self.decoder, encoding=new_encoding
        )
        
    def forward(self, src, tgt, mask = None, attn_mask = None, tgt_mask=None, score_in=None, **kwargs):
        enc = self.encoder(src, mask = mask, attn_mask = attn_mask, return_embeddings = True, score_in = score_in)
        out = self.decoder(tgt, context = enc, context_mask = mask, tgt_mask = tgt_mask, **kwargs)
        return out
    
    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_lens, mask = None, attn_mask = None, score_in=None, **kwargs):
        encodings = self.encoder(seq_in, mask = mask, attn_mask = attn_mask, score_in=score_in, return_embeddings = True)
        seq_len = seq_lens.max().item()
        return self.decoder.generate(seq_out_start[..., :6], seq_len, context = encodings, context_mask = mask, **kwargs)
    
class MusicXScorerTopK(nn.Module):
    def __init__(self, *, dim, encoding, conv_type, k=1, max_score_density=200, pitch_sep=False, temp=1, ich=False, **kwargs):
        super().__init__()
        assert "dim" not in kwargs, "dimension must be set with `dim` keyword"
        transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_beat": kwargs.pop("max_beat"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        self.conv_type = conv_type
        self.encoding = copy.copy(encoding)
        self.encoder = MusicTransformerWrapper(
            encoding=self.encoding,
            attn_layers=Encoder(dim=dim, **kwargs),
            tie_embedding=True,
            pitch_sep=pitch_sep,
            **transformer_kwargs,
        )
        self.score = nn.Linear(dim, 1)
        self.msd = max_score_density
        self.k = k
        if self.k < 1:
            # frac mode
            self.mode = 'frac'
        else:
            self.mode = 'k'
            self.k = int(self.k)
        self.temp = temp
        self.ich = ich

    @torch.no_grad()
    def generate(self, seq_in, seq_len, **kwargs):
        pass

    def forward(self, src, inD, mask = None, attn_mask = None, temp=None, pitch_mask=None, train=True, kl=0, pretrain=False, st=False, ich=False, **kwargs):
        enc_out = self.encoder(src, mask = mask, attn_mask = attn_mask, return_embeddings = True, **kwargs)
        scores = self.score(enc_out)
        temp = temp if temp is not None else self.temp
        outs = []
        prob = torch.zeros((src.shape[0], max([len(x)+len(list(x.values())[1])-1 for x in inD['onsetdict']]), max([max([len(y) for y in x.values()]) for x in inD['onsetdict']]))).to(device)
        mask_ons = inD['ons_masks'].to(device)
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g * train
        prob[inD['ons_d_k']] = scores.squeeze(-1)[inD['ons_d_v']]
        prob = prob + (1-mask_ons) * -1e10
        probs = torch.zeros_like(prob)
        hard = torch.zeros_like(probs).view(-1, probs.shape[-1])
        onehot_approx = torch.zeros_like(prob)

        has_chord_idxs = [torch.nonzero(torch.tensor([src[ix, x[k][0], 0] == 5 for k in list(x.keys())[3:]]))+len(list(x.values())[1])+2 for ix, x in enumerate(inD['onsetdict'])]

        if self.mode == 'frac':
            frac_lens = (mask_ons.sum(-1) * self.k).ceil().long()
            hc_bound = frac_lens.max().item()
            k_masks = (mask_ons.cumsum(dim=-1) <= frac_lens.unsqueeze(-1)).float()[..., :hc_bound+1]
            
            for ix, x in enumerate(has_chord_idxs):
                k_masks[ix, x, frac_lens[ix]] = 1

        else:
            hc_bound = self.k
            has_chord_mask = torch.zeros((mask_ons.shape[0],  mask_ons.shape[1], hc_bound+1)).to(device)
            for ix, x in enumerate(has_chord_idxs):
                has_chord_mask[ix, :len(inD['onsetdict'][ix])+len(list(inD['onsetdict'][ix].values())[1])-1, :self.k] = 1
                has_chord_mask[ix, x, -1] = 1

        if self.mode == 'frac':
            for i in range(hc_bound+1):
                khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).to(device))
                prob = prob + torch.log(khot_mask)
                onehot_approx = torch.nn.functional.softmax(prob / temp, dim=-1)
                probs = probs + onehot_approx
            
            val, ind = torch.topk(probs, hc_bound+1, dim=-1)
            hard = hard.scatter_(1, ind.view(-1, hc_bound+1), k_masks.view(-1, hc_bound+1))
        else:
            for i in range(self.k+1):
                khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).to(device))
                prob = prob + torch.log(khot_mask)
                onehot_approx = torch.nn.functional.softmax(prob / temp, dim=-1)
                probs = probs + onehot_approx
            val, ind = torch.topk(probs, self.k+1, dim=-1)
            hard = hard.scatter_(1, ind.view(-1, self.k+1), has_chord_mask.view(-1, self.k+1))

        hard = hard.view(*probs.shape) * mask_ons
        hard = hard - probs.detach() + probs
        probs = probs + (1-mask_ons) * 1e-10
        hard = hard * mask_ons
        outs = torch.zeros_like(scores).squeeze(-1)
        outs[inD['ons_d_v']] = hard[inD['ons_d_k']]
        if pretrain:
            return outs, probs * mask_ons
        if ich:
            out, out_mask, n_chords = diff_mask_src(src, [x.emb.weight.shape[0] for x in self.encoder.token_emb], outs.unsqueeze(-1), train=train, st=st, ich=ich)
            return out, out_mask, n_chords
        out, out_mask = diff_mask_src(src, [x.emb.weight.shape[0] for x in self.encoder.token_emb], outs.unsqueeze(-1), train=train, st=st)
        return out, out_mask

class MusicXSkyline(nn.Module):
    '''
    Not an actual transformer, but similar encoder architecture that deterministically extracts melody using skyline algorithm
    '''
    def __init__(self):
        super().__init__()


    @torch.no_grad()
    def generate(self, seq_in, seq_len, **kwargs):
        pass

    def forward(self, src, inD, mask = None, attn_mask = None, temp=1, pitch_mask=None, train=True, st=False, ich=False, **kwargs):
        reds = []
        n_beats = []


        for ix, seq in enumerate(src):
            red, n_beat = extract_skyline_with_mask(seq, inD, ix, ich=ich)
            reds.append(red)
            n_beats.append(n_beat)
        reduct = pad(reds)
        if st:
            notes = (reduct[..., 0] == 3).int()
            reduct[..., 5] = notes * 1 + (1 - notes) * reduct[..., 5]

        red_mask = get_mask(reds)
        return reduct, red_mask, torch.tensor(n_beats).to(device)


class MusicXLeadAE(nn.Module):
    def __init__(self, *, scorer, transformer, **kwargs):
        super().__init__()
        transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_beat": kwargs.pop("max_beat"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        self.hard = kwargs.pop("hard", True)
        self.temp = kwargs.pop("temp", 1)
        self.seq2seq = transformer
        self.scorer = scorer
        self.ind_arr = None

    def forward(self, src, inD, mask = None, attn_mask = None, train=True, encoding=None, st=False, ich=True, **kwargs):
        if train:
            preamble = torch.where(src[:, :, 0] == 2)[1]
            loss = {}
            if type(self.scorer) == MusicXScorerTopK:
                reduct, rmask, n_chords = self.scorer(src if inD['chord_seq'] is None else inD['chord_seq'].to(device), inD, mask = inD['chord_mask'].to(device) if inD['chord_mask'] is not None else mask, attn_mask = attn_mask, temp=self.temp, pitch_mask=pitch_mask, kl=kl, pretrain=False, train=train, st=st, ich=ich, **kwargs)
                recon_loss, accs = self.seq2seq(reduct, src[..., :6], mask = rmask.to(device), attn_mask = attn_mask, tgt_mask=mask, **kwargs)
                loss['sparsity'] = ((rmask.to(device).sum(dim=-1) - n_chords - preamble - 1) / (mask.sum(dim=-1) - preamble - 1)).mean()
                loss['chord_sparsity'] = (n_chords / (src[..., 1].max(dim=1)[0] - preamble - 1)).mean()
            elif type(self.scorer) == MusicXSkyline:
                reduct, rmask, n_beats = self.scorer(src if inD['chord_seq'] is None else inD['chord_seq'].to(device), inD, mask = inD['chord_mask'].to(device) if inD['chord_mask'] is not None else mask, attn_mask = attn_mask, temp=self.temp, pitch_mask=pitch_mask, train=train, kl=kl, frac_lens=frac_lens, pretrain=False, st=st, ich=ich, **kwargs)
                recon_loss, accs = self.seq2seq(reduct, src[..., :6], mask = rmask.to(device), attn_mask = attn_mask, tgt_mask=mask, **kwargs)
                if ich:
                    loss['sparsity'] = ((rmask.to(device).sum(dim=-1) - n_beats)/ (mask.sum(dim=-1))).mean()
                    loss['chord_sparsity'] = (n_beats / src[..., 1].max(dim=1)[0]).mean()
                else:
                    loss['sparsity'] = (rmask.to(device).sum(dim=-1) / mask.sum(dim=-1)).mean()
            else:
                raise NotImplementedError('Only MusicXScorerTopK and MusicXSkyline are supported for training')
            loss['recon'] = recon_loss
            
            return loss, accs
        else:
            preamble = torch.where(src[:, :, 0] == 2)[1]
            # create left pad src_inp for each example in batch such that full preamble is included in input
            src_inp = torch.zeros((src.shape[0], preamble.max().item() + 1, src.shape[2])).to(device).long()
            preamb_offs = preamble.max().item() - preamble
            
            for ix in range(src.shape[0]):
                src_inp[ix, preamb_offs[ix]:, :] = src[ix, :preamble[ix].item()+1, :].long()
            if type(self.scorer) == MusicXScorerTopK:
                reduct, rmask, n_chords = self.scorer(src if inD['chord_seq'] is None else inD['chord_seq'].to(device), inD, mask =inD['chord_mask'].to(device) if inD['chord_mask'] is not None else mask, attn_mask = attn_mask, temp=self.temp, pitch_mask=None, kl=kl, **kwargs, train=False, st=st, ich=ich)
                out = self.seq2seq.generate(reduct, src_inp[..., :6], inD['seq_len'].to(device), mask = rmask.to(device), attn_mask = attn_mask, **kwargs)
            elif type(self.scorer) == MusicXSkyline:
                reduct, rmask, _ = self.scorer(src if inD['chord_seq'] is None else inD['chord_seq'].to(device), inD, mask = inD['chord_mask'].to(device) if inD['chord_mask'] is not None else mask, attn_mask = attn_mask, temp=self.temp, kl=kl, pretrain=False,train=False, encoding=encoding, st=st, ich=ich, **kwargs)
                out = self.seq2seq.generate(reduct, src_inp[..., :6], inD['seq_len'].to(device), mask = rmask.to(device), attn_mask = attn_mask, **kwargs)
            else:
                raise NotImplementedError('Only MusicXScorerTopK and MusicXSkyline are supported for inference')
            return out, reduct
    

def diff_mask_src(inp, encoding, hard, train=True, st=False, ich=False):
    src = inp.clone()
    if st: # get rid of instrument field
        notes = (src[..., 0] == 3).float()
        src[..., 5] = notes * 1 + (1 - notes) * src[..., 5]
    if train: #output will be list of one-hot encoded tensors for each field
        src_onehot = [
            F.one_hot(src[:, :, ix], num_classes=encoding[ix]).float() for ix in range(len(encoding))
        ]

        new_lens = torch.sum(hard, dim=1).view(-1).int()
        out = [
            torch.zeros(src.shape[0], max(new_lens).item(), encoding[ix]).to(device) for ix in range(len(encoding))
        ]
        for i in range(src.shape[0]):
            for ix in range(len(encoding)):
                masked_src = src_onehot[ix][i,:] * hard[i]
                hard_idxs = torch.nonzero(hard[i].view(-1)).squeeze()
                out[ix][i, :new_lens[i], :] = masked_src[hard_idxs, :]
        max_red_len = max(new_lens)
        out = [out[ix][:, :max_red_len, :] for ix in range(len(encoding))]
        # get new mask
        new_mask = torch.zeros((src.shape[0], max_red_len)).to(src.device)
        for i in range(src.shape[0]):
            new_mask[i, :new_lens[i]] = 1
        if ich:
            n_chords = torch.cat([x[..., 5].sum().reshape(1) for x in out[0]])
            return out, new_mask.bool(), n_chords
        return out, new_mask.bool()
    else: #output will be selected rows from src, as does not need to be differentiable
        new_lens = torch.sum(hard, dim=1).view(-1).int()
        out = torch.zeros(src.shape[0], max(new_lens).item(), src.shape[2]).to(device)
        for i in range(src.shape[0]):
            masked_src = src[i,:] * hard[i]
            hard_idxs = torch.nonzero(hard[i].view(-1)).squeeze()
            out[i, :new_lens[i], :] = masked_src[hard_idxs, :]
        max_red_len = max(new_lens)
        out = out[:, :max_red_len, :]
        # get new mask
        new_mask = torch.zeros((src.shape[0], max_red_len)).to(src.device)
        for i in range(src.shape[0]):
            new_mask[i, :new_lens[i]] = 1
        if ich:
            n_chords = (out[..., 0] == 5).sum(dim=1)
            return out.int(), new_mask.bool(), n_chords
        return out.int(), new_mask.bool()

