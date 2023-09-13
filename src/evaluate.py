'''
Evaluate reductions
'''
import  torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utils import *
from models import *
from mmt_dataset import *
import representation
import representation12
import wandb
import time
import math
import os
import argparse
import random
from functools import reduce
from mmt_x_transformers import *
from tqdm import tqdm
import mute
import muspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd", 'hymnal', 'pop909'),
        required=True,
        help="dataset key",
    )
    parser.add_argument(
        "-evm",
        "--eval_mode",
        choices=("all", "ex", 'study'),
        required=True,
        help="evaluation mode, choice between generation metrics for all samples in validation set or generating examples for single sample",
    )
    parser.add_argument(
        "-t", "--train_names", type=pathlib.Path, help="training names"
    )
    parser.add_argument(
        "-v", "--valid_names", type=pathlib.Path, help="validation names"
    )
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    # Data
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=8,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use data augmentation",
    )
    # Model
    parser.add_argument(
        "--max_seq_len",
        default=1024,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--max_beat",
        default=256,
        type=int,
        help="maximum number of beats",
    )
    parser.add_argument("--dim", default=256, type=int, help="model dimension")
    parser.add_argument(
        "-l", "--layers", default=1, type=int, help="number of layers"
    )
    parser.add_argument(
        "--heads", default=4, type=int, help="number of attention heads"
    )
    parser.add_argument(
        "--dropout", default=0.2, type=float, help="dropout rate"
    )
    parser.add_argument(
        "--abs_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use absolute positional embedding",
    )
    parser.add_argument(
        "--rel_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to use relative positional embedding",
    )
    # Training
    parser.add_argument(
        "--steps",
        default=200000,
        type=int,
        help="number of steps",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--n_epochs",
        default=1000,
        type=int,
        help="number of epochs",
    )
    parser.add_argument(
        "--valid_steps",
        default=1000,
        type=int,
        help="validation frequency",
    )
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use early stopping",
    )
    parser.add_argument(
        "-e",
        "--early_stopping_tolerance",
        default=20,
        type=int,
        help="number of extra validation rounds before early stopping",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.0005,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        default=5000,
        type=int,
        help="learning rate warmup steps",
    )
    parser.add_argument(
        "--lr_decay_steps",
        default=100000,
        type=int,
        help="learning rate decay end steps",
    )
    parser.add_argument(
        "--plot_freq",
        default=10,
        type=int,
        help="plot frequency",
    )
    parser.add_argument(
        "-p",
        "--p",
        default=0.40,
        type=float,
        help="top p for top p sampling",
    )
    parser.add_argument(
        "--lr_decay_multiplier",
        default=0.1,
        type=float,
        help="learning rate multiplier at the end",
    )
    parser.add_argument(
        "--grad_norm_clip",
        default=1.0,
        type=float,
        help="gradient norm clipping",
    )
    # Others
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j",
        "--jobs",
        default=4,
        type=int,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    parser.add_argument(
        "-sb", "--strip_beats", action="store_true", help="strip beat tokens in target sequence"
    )
    parser.add_argument(
        "-fd", "--freeze_decoder", action="store_true", help="freeze decoder"
    )
    parser.add_argument(
        "-log", "--logging", type=str, default='True', help="logging"
    )
    parser.add_argument(
        "-ct", "--conv_type", type=str, default='onset', help="convolution type"
    )
    parser.add_argument(
        "-dch", "--decoder_checkpoint", type=str, default=None, help="decoder checkpoint"
    )
    parser.add_argument(
        "-mch", "--model_checkpoint", required=True, type=str, default=None, help="model checkpoint"
    )
    parser.add_argument(
        "-ev", "--eval", action="store_true", help="reduction evaluation, where decoder is trained with fixed encoder"
    )
    parser.add_argument(
        "-mm", "--multi_mask", action="store_true", help="predict duration and instrument in reduction as well as pitch"
    )
    parser.add_argument(
        "-kl", "--kl_term", type=float, default=0, help="kl term"
    )
    parser.add_argument(
        "-k", "--k", type=float, default=1, help="if k greater than 1, use topk scorer"
    )
    parser.add_argument(
        "-wt", "--window_type", type=str, default='onset', help="window type"
    )
    parser.add_argument(
        "-sm", "--scorer_mode", type=str, default='topk', help="topk or topfrac"
    )
    parser.add_argument(
        "-pre", "--pitch-rel-emb",  action="store_true", help="add relative pitch embedding along each beat"
    )
    parser.add_argument(
        "-premd", "--pitch-rel-emb-max-dense",  type=int, default=45, help="maximum number of pitches vertically"
    )
    parser.add_argument(
        "-ch", "--chord",  action="store_true", help="use chords"
    )
    parser.add_argument(
        "-rlch", "--rl-chord",  action="store_true", help="use chords without inversion info"
    )
    parser.add_argument(
        "-upre", "--use-pre",  action="store_true", help="use pitch rel emb in decoder"
    )
    parser.add_argument(
        "-st", "--st",  action="store_true", help="enforce reduction to single track"
    )
    parser.add_argument(
        "-ich", "--inter-chord",  action="store_true", help="use chords interleaved with notes"
    )
    return parser.parse_args(args=args, namespace=namespace)



def init_model(args, device, encoding):
    kwargs = {
        'dim': args.dim,
        'encoding': encoding,
        'depth': args.layers,
        'heads': args.heads,
        'max_seq_len': args.max_seq_len,
        'max_beat': args.max_beat,
        'rotary_pos_emb': args.rel_pos_emb,
        'use_abs_pos_emb': args.abs_pos_emb,
        'emb_dropout': args.dropout,
        'attn_dropout': args.dropout,
        'ff_dropout': args.dropout,
    }
    if args.conv_type in ['onset', 'beat']:
        scorer = MusicXScorerTopK(
            conv_type=args.conv_type,
            k=args.k, 
            **kwargs
        ).to(device)
    elif args.conv_type == 'skyline': # fixed encoder w/skyline algorithm
        scorer = MusicXSkyline().to(device)
    else:
        raise NotImplementedError
    
    s2s = MusicXTransformer(
        encoder=True,
        decoder=True,
        **kwargs
    ).to(device)
    model = MusicXLeadAE(
        scorer=scorer,
        transformer=s2s,
        **kwargs
    ).to(device)
    if args.model_checkpoint is not None:
        state_dict = torch.load(args.model_checkpoint, map_location=device)
        print('loaded model best acc')
        if 'acc' in state_dict:
            print(state_dict['acc'])
        else:
            tmp_sd = {}
            tmp_sd['model'] = state_dict
            state_dict = tmp_sd
        for n, p in model.named_parameters():
            if n in state_dict['model']:
                p.data = state_dict['model'][n].data
                if 'scorer' in n:
                    p.requires_grad = False
    else:
        raise ValueError('model checkpoint required')
    return model.to(device)


ARGDATA2PATH = {
    'sod': 'sod/processed',
    'pop909': 'pop909'
}

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    encoding = representation.get_encoding(True)
    prog_intr_map = {v: k for k, v in encoding['instrument_program_map'].items()}
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    

    if args.inter_chord:
        encoding['n_tokens'][3] += 14
        encoding['n_tokens'][4] += 13
    model = init_model(args, device, encoding)
    print("model loaded")
    model.eval()
    if args.dataset == 'pop909':
        dataset_type = POP909Dataset
    else:
        dataset_type = MusicDataset
    valid_dataset = dataset_type(f"../data/{ARGDATA2PATH[args.dataset]}/test-names.txt", f"../data/{ARGDATA2PATH[args.dataset]}/notes", encoding=encoding, max_seq_len=args.max_seq_len, max_bt_seq_len=args.max_seq_len, max_beat=256, use_augmentation=False, window_type=args.window_type,  pitch_rel_emb=args.pitch_rel_emb, chord=args.chord, rl_chord=args.rl_chord, inter_chord=args.inter_chord)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, args.batch_size if args.eval_mode == 'all' else 1, False, collate_fn=valid_dataset.collate, num_workers=1
    )
    errs = 0
    if args.eval_mode == 'all':
        rec_mutes, rec_pc_mutes, rec_pr_accs, rec_pc_pr_accs = [], [], [], []
        if dataset_type == POP909Dataset:
             red_mutes, red_pc_mutes, red_pr_accs, red_pc_pr_accs = [], [], [], []
    inps = []
    outs = []
    reds = []
    names = []
    name_iou = []
    for ix, batch in enumerate(tqdm(valid_data_loader)):
        inp = batch["seq"].to(device)
        mask = batch["mask"].to(device)
        inp_lens = batch["seq_len"].to(device)
        with torch.no_grad():
            out, reduct = model(inp, batch, mask=mask, train=False, kl=args.kl_term, encoding=encoding,  st=args.st, ich=args.inter_chord)
        
        if args.eval_mode == 'ex':
            break
    for inp, reduct, out, name in zip(inps, reds, outs, names):
        print(name)
        inp_mus = representation.decode(inp[..., :6].reshape(-1, 6), encoding)
        for trck in inp_mus.tracks:
            trck.name = prog_intr_map[trck.program]
        if args.eval_mode == 'ex':
            if not os.path.exists("/".join(args.model_checkpoint.split("/")[:-1]) + f"/valid_{batch['name'][0].split('/')[0]}"):
                os.mkdir("/".join(args.model_checkpoint.split("/")[:-1]) + f"/valid_{batch['name'][0].split('/')[0]}")
            print(batch['name'][0])
            inp_mus.write_midi("/".join(args.model_checkpoint.split("/")[:-1]) + f"/valid_{batch['name'][0]}_original.mid")
            inp_img = muspy.show_pianoroll(inp_mus, preset='frame', ytick='off', xtick='off')
            plt.savefig("/".join(args.model_checkpoint.split("/")[:-1]) + f"/valid_{batch['name'][0]}_original.png", dpi=300)
        
        if args.inter_chord:
            og_red = reduct.clone()
            reduct = reduct[reduct[..., 0] != 5]
        reduct_mus = representation.decode(reduct[..., :6].reshape(-1, 6), encoding)

        out_mus = representation.decode(out[..., :6].reshape(-1, 6), encoding)
        if args.eval_mode == 'ex':
            for trck in reduct_mus.tracks:
                trck.name = prog_intr_map[trck.program]
            reduct_mus.write_midi("/".join(args.model_checkpoint.split("/")[:-1]) + f"/valid_{batch['name'][0]}_reduct.mid")
            # get image of piano roll
            reduct_img = muspy.show_pianoroll(reduct_mus, preset='frame',ytick='off', xtick='off')
            plt.savefig("/".join(args.model_checkpoint.split("/")[:-1]) + f"/valid_{batch['name'][0]}_reduct.png", dpi=300)
        if args.eval_mode == 'ex':
            for trck in out_mus.tracks:
                trck.name = prog_intr_map[trck.program]
            out_mus.write_midi("/".join(args.model_checkpoint.split("/")[:-1]) + f"/valid_{batch['name'][0]}_recons.mid")
            # get image of piano roll
            out_img = muspy.show_pianoroll(out_mus, preset='frame', ytick='off', xtick='off')
            plt.savefig("/".join(args.model_checkpoint.split("/")[:-1]) + f"/valid_{batch['name'][0]}_recons.png", dpi=300)

        inp_proll = inp_mus.to_pianoroll_representation()
        out_proll = out_mus.to_pianoroll_representation()
        if dataset_type == POP909Dataset:
            # load in ground truth reduction
            gt_red = muspy.read_midi(f"../data/{ARGDATA2PATH[args.dataset]}/notes/{name}_mel_dec.mid")
            gt_proll = gt_red.to_pianoroll_representation()
            reduct_proll = reduct_mus.to_pianoroll_representation()
        
        try:
            fail_on = 0
            max_len = max(len(inp_proll), len(out_proll))
            inp_proll = np.pad(inp_proll, ((0, max_len - len(inp_proll)), (0, 0))).astype(bool).astype(int)
            out_proll = np.pad(out_proll, ((0, max_len - len(out_proll)), (0, 0))).astype(bool).astype(int)
            max_piano_len = max(len(inp_piano_proll), len(out_piano_proll))
            inp_piano_proll = np.pad(inp_piano_proll, ((0, max_piano_len - len(inp_piano_proll)), (0, 0))).astype(bool).astype(int)
            out_piano_proll = np.pad(out_piano_proll, ((0, max_piano_len - len(out_piano_proll)), (0, 0))).astype(bool).astype(int)

            recons_mute_score = f1_score(inp_proll, out_proll, average='samples', zero_division=1)
            recons_pc_mute_score = f1_score(mute.to_pitchclass_pianoroll(inp_proll), mute.to_pitchclass_pianoroll(out_proll), average='samples', zero_division=1)
            recons_proll_acc = jaccard_score(inp_proll, out_proll, average='micro', zero_division=0)
            recons_pc_proll_acc = jaccard_score(mute.to_pitchclass_pianoroll(inp_proll), mute.to_pitchclass_pianoroll(out_proll), average='micro', zero_division=0)
            if dataset_type == POP909Dataset:
                max_len = max(len(gt_proll), len(reduct_proll))
                gt_proll = np.pad(gt_proll, ((0, max_len - len(gt_proll)), (0, 0))).astype(bool).astype(int)
                reduct_proll = np.pad(reduct_proll, ((0, max_len - len(reduct_proll)), (0, 0))).astype(bool).astype(int)
                red_mute_score = f1_score(gt_proll, reduct_proll, average='samples', zero_division=np.nan)
                red_pc_mute_score = f1_score(mute.to_pitchclass_pianoroll(gt_proll), mute.to_pitchclass_pianoroll(reduct_proll), average='samples', zero_division=np.nan)
                red_proll_acc = jaccard_score(gt_proll, reduct_proll, average='micro', zero_division=1)
                red_pc_proll_acc = jaccard_score(mute.to_pitchclass_pianoroll(gt_proll), mute.to_pitchclass_pianoroll(reduct_proll), average='micro', zero_division=1)


            if args.eval_mode == 'ex':
                print("Recons Mute Score: ", recons_mute_score)
                print("Recons Pitch Class Mute Score: ", recons_pc_mute_score)
                print("Recons Pianoroll Acc: ", recons_proll_acc)
                print("Recons Piano PC Acc: ", recons_pc_proll_acc)
                exit()
            else:
                rec_mutes.append(recons_mute_score)
                rec_pc_mutes.append(recons_pc_mute_score)
                rec_pr_accs.append(recons_proll_acc)
                rec_pc_pr_accs.append(recons_pc_proll_acc)
                if dataset_type == POP909Dataset:
                    red_mutes.append(red_mute_score)
                    red_pc_mutes.append(red_pc_mute_score)
                    red_pr_accs.append(red_proll_acc)
                    red_pc_pr_accs.append(red_pc_proll_acc)
        except Exception as e:
            print(e)
            print("Failed on task", fail_on)
            print("Error in evaluation")
            errs += 1
            continue
    print("Recons MuTE Score:", np.mean(rec_mutes))
    print("Recons Chroma MuTE Score:", np.mean(rec_pc_mutes))
    print("Recons PRoll Hamming Accuracy", np.mean(rec_pr_accs))
    print("Recons Chroma PRoll Hamming Accuracy", np.mean(rec_pc_pr_accs))
    if dataset_type == POP909Dataset:
        print("Reduction MuTE Score:", np.mean(red_mutes))
        print("Reduction Chroma MuTE Score:", np.mean(red_pc_mutes))
        print("Reduction PRoll Hamming Accuracy", np.mean(red_pr_accs))
        print("Reduction Chroma PRoll Hamming Accuracy", np.mean(red_pc_pr_accs))
    print("Errors: ", errs)
    with open("/".join(args.model_checkpoint.split("/")[:-1]) + f"/test_metrics_{args.seed}.txt", "w") as f:
        f.write(f"Recons MuTE Score: {np.nanmean(rec_mutes)}\n")
        f.write(f"Recons Chroma MuTE Score: {np.nanmean(rec_pc_mutes)}\n")
        f.write(f"Recons PRoll Hamming Accuracy: {np.nanmean(rec_pr_accs)}\n")
        f.write(f"Recons Chroma PRoll Hamming Accuracy: {np.nanmean(rec_pc_pr_accs)}\n")
        if dataset_type == POP909Dataset:
            f.write(f"Reduction MuTE Score: {np.nanmean(red_mutes)}\n")
            f.write(f"Reduction Chroma MuTE Score: {np.nanmean(red_pc_mutes)}\n")
            f.write(f"Reduction PRoll Hamming Accuracy: {np.nanmean(red_pr_accs)}\n")
            f.write(f"Reduction Chroma PRoll Hamming Accuracy: {np.nanmean(red_pc_pr_accs)}")
        f.write(f"Errors: {errs}\n")
