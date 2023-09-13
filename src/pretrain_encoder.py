import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument(
        "-btp",
        "--bt_padding",
        default=0,
        type=int,
        help="maximum number of additional beat tokens to allow"
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
        "-fd", "--freeze_decoder", action="store_true", help="freeze decoder"
    )
    parser.add_argument(
        "-log", "--logging", type=str, default='True', help="logging"
    )
    parser.add_argument(
        "-ct", "--conv_type", type=str, default='onset', help="convolution type"
    )
    parser.add_argument(
        "-wt", "--window_type", type=str, default='onset', help="window type"
    )
    parser.add_argument(
        "-dch", "--decoder_checkpoint", type=str, default=None, help="decoder checkpoint"
    )
    parser.add_argument(
        "-mch", "--model_checkpoint", type=str, default=None, help="model checkpoint"
    )
    parser.add_argument(
        "-ev", "--eval", action="store_true", help="reduction evaluation, where decoder is trained with fixed encoder"
    )
    parser.add_argument(
        "-k", "--k", type=int, default=1, help="if k greater than 1, use topk scorer"
    )
    parser.add_argument(
        "-ich", "--inter-chord",  action="store_true", help="use chords interleaved with notes"
    )
    return parser.parse_args(args=args, namespace=namespace)


def get_lr_multiplier(
    step, warmup_steps, decay_end_steps, decay_end_multiplier
):
    """Return the learning rate multiplier with a warmup and decay schedule.
    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position


def train(data_loader, model, optimizer, scheduler, plot_freq=10, best_acc=None, trstep=0, encoding=None, pitch_sep=False):
    model.train()

    plot_acc, tot_loss = [], []
    n_batch = 0
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        inp = batch["chord_seq"].to(device) if args.inter_chord else batch["seq"].to(device)
        mask = batch['chord_mask'].to(device) if args.inter_chord else batch["mask"].to(device)
        inp_lens = batch["chord_seq_len"].to(device) if args.inter_chord else batch["seq_len"].to(device)
        if args.pretrain_encoder == 'skyline': 
            tgts = pad([extract_skylinemask(batch, ix, args.inter_chord) for ix in range(inp.shape[0])]).to(device)
        else:
            raise NotImplementedError
        out = model(inp, batch, mask = mask, pretrain=True)
        if args.inter_chord:
            hard_out = out[0]
            out = out[1]
            out += 1e-9
            out /= out.sum(-1, keepdim=True)
            ons_tgts = torch.zeros_like(out).to(device)
            ons_tgts[batch['ons_d_k']] = tgts.squeeze(-1)[batch['ons_d_v']]
            ons_tgts /= ons_tgts.sum(-1, keepdim=True)
            loss = loss_fn(torch.log(out).transpose(2, 1), ons_tgts.transpose(2, 1))
            acc = (hard_out == tgts.squeeze(-1)) * mask
        else:
            raise NotImplementedError

        loss = (loss.nansum(-1) / batch['ons_masks'].to(device)[..., 0].sum(1)).mean()
        acc = (acc.nansum(-1) / (inp_lens)).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

        tot_loss.append(loss.item())
        plot_acc.append(acc.item())


        
        n_batch += 1
        if i % plot_freq == 0 or i == len(data_loader) - 1:
            print(f"Acc: {np.mean(plot_acc):.4f}  Total Loss: {np.mean(tot_loss):.4f}")
            if args.logging:
                
                wandb.log({'train/tot_loss': np.mean(tot_loss), 'custom_step': trstep})
                wandb.log({'train/acc': np.mean(plot_acc), 'custom_step': trstep})
            if best_acc is None or np.mean(plot_acc) > best_acc:
                best_acc = np.mean(plot_acc)
                if args.logging:
                    if not os.path.exists(f"../logs/{args.dataset}/{run.name}"):
                        os.mkdir(f"../logs/{args.dataset}/{run.name}")
                    torch.save(model.state_dict(), f"../logs/{args.dataset}/{run.name}/tr_model.pt")
            plot_acc, tot_loss = [], []
            n_batch = 0
        trstep += 1
        
    
            
    return model, best_acc, trstep
        

def evaluate(data_loader, encoding, model, lambd=0.1, trstep=0, epoch=0, best_acc=None, pitch_sep=False):
    model.eval()

    plot_loss = 0
    plot_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inp = batch["chord_seq"].to(device) if args.inter_chord else batch["seq"].to(device)
            mask = batch['chord_mask'].to(device) if args.inter_chord else batch["mask"].to(device)
            inp_lens = batch["chord_seq_len"].to(device) if args.inter_chord else batch["seq_len"].to(device)
            if args.pretrain_encoder == 'skyline': 
                tgts = pad([extract_skylinemask(batch, ix, args.inter_chord) for ix in range(inp.shape[0])]).to(device)
            else:
                raise NotImplementedError
            out = model(inp, batch, mask = mask, pretrain=True)
            hard_out = out[0]
            out = out[1]
            out += 1e-9
            out /= out.sum(-1, keepdim=True)
            ons_tgts = torch.zeros_like(out).to(device)
            ons_tgts[batch['ons_d_k']] = tgts.squeeze(-1)[batch['ons_d_v']]
            ons_tgts /= ons_tgts.sum(-1, keepdim=True)
            loss = loss_fn(torch.log(out).transpose(2, 1), ons_tgts.transpose(2, 1))
            acc = (hard_out == tgts.squeeze(-1)) * mask
            loss = (loss.nansum(-1) / batch['ons_masks'].to(device)[..., 0].sum(1)).mean()
            acc = (acc.nansum(-1) / (inp_lens)).mean()
            plot_loss += loss.item()
            plot_acc += acc.item()
            

        if args.logging:
            wandb.log({'val/loss': plot_loss / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/acc': plot_acc / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
        if best_acc is None or plot_acc / len(data_loader) > best_acc:
            best_acc = plot_acc / len(data_loader)
            if args.logging:
                if not os.path.exists(f"../logs/{args.dataset}/{run.name}"):
                    os.mkdir(f"../logs/{args.dataset}/{run.name}")
                torch.save(model.state_dict(), f"../logs/{args.dataset}/{run.name}/val_model.pt")

    return best_acc, frozen_fields


def init_run(args):
    if not os.path.exists(f"../logs/{args.dataset}"):
        os.mkdir(f"../logs/{args.dataset}")
    run = wandb.init(
        project=f'{args.dataset}-PretrainEncoder',
        config=args,
        mode="disabled" if args.logging == False else "online"
    )
    if args.logging:
        if args.pretrain_encoder == 'skyline':
            run.name = f"SKYPE-{args.learning_rate}-{args.layers}-{args.heads}-{args.dim}-" + run.name.split("-")[-1]
        else:
            run.name = f"PAE-{args.learning_rate}-{args.layers}-{args.heads}-{args.dim}-" + run.name.split("-")[-1]
    return run

def init_model(args, device, encoding):
    kwargs = {
        'dim': args.dim,
        'encoding': encoding,
        'depth': args.layers,
        'heads': args.heads,
        'max_seq_len': args.max_seq_len+args.bt_padding,
        'max_beat': args.max_beat,
        'rotary_pos_emb': args.rel_pos_emb,
        'use_abs_pos_emb': args.abs_pos_emb,
        'emb_dropout': args.dropout,
        'attn_dropout': args.dropout,
        'ff_dropout': args.dropout,
    }
    if args.conv_type == 'onset':
        model = MusicXScorerTopK(
            conv_type=args.conv_type,
            k=args.k,
            **kwargs
        ).to(device)
    elif args.conv_type == 'skyline': # fixed encoder w/skyline algorithm
        model = MusicXSkyline()
    else:
        raise NotImplementedError
    if args.model_checkpoint is not None:
        state_dict = torch.load(args.model_checkpoint)
        for n, p in model.named_parameters():
            if n in state_dict:
                p.data = state_dict[n].data
    
    return model


ARGDATA2PATH = {
    'JSBach': 'JSBach/processed',
    'C3': 'cocochorales_tiny_mmt/main_dataset',
    'NES': 'NES',
    'hymnal': 'hymnal',
    'snd': 'snd',
    'sod': 'sod/processed',
    'pop909': 'pop909',
}


if __name__ == "__main__":
    args = parse_args()
    args.logging = True if args.logging == "True" else False
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    encoding = representation.get_encoding(True)
    if args.inter_chord:
        encoding['n_tokens'][3] += 14
        encoding['n_tokens'][4] += 13
    if args.dataset == 'pop909':
        dataset_type = POP909Dataset
    else:
        dataset_type = MusicDataset

    
    train_dataset = dataset_type(f"../data/{ARGDATA2PATH[args.dataset]}/train-names.txt", f"../data/{ARGDATA2PATH[args.dataset]}/notes", encoding=encoding, max_seq_len=args.max_seq_len, max_bt_seq_len=args.max_seq_len+args.bt_padding,max_beat=256, use_augmentation=True, window_type=args.window_type, inter_chord=args.inter_chord)
    valid_dataset = dataset_type(f"../data/{ARGDATA2PATH[args.dataset]}/valid-names.txt", f"../data/{ARGDATA2PATH[args.dataset]}/notes", encoding=encoding, max_seq_len=args.max_seq_len, max_bt_seq_len=args.max_seq_len+args.bt_padding,max_beat=256, use_augmentation=False, window_type=args.window_type, inter_chord=args.inter_chord)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, args.batch_size, True, collate_fn=train_dataset.collate, num_workers=args.jobs
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, args.batch_size, True, collate_fn=valid_dataset.collate, num_workers=args.jobs
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_model(args, device, encoding)
    run = init_run(args)

    tr_acc = None
    val_acc = None
    step = 0
    frozen_fields = []
    nan_counter = 0
    for epoch in range(args.n_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr_multiplier(
                step,
                args.lr_warmup_steps,
                args.lr_decay_steps, 
                args.lr_decay_multiplier,
            ),
        )
        model, tr_acc, step = train(train_data_loader, model, optimizer, scheduler, plot_freq=args.plot_freq, best_acc=tr_acc, trstep=step, encoding=encoding, pitch_sep=args.pitch_sep)
        val_acc, frozen_fields = evaluate(valid_data_loader, encoding, model, trstep=step, best_acc=val_acc, epoch=epoch, pitch_sep=args.pitch_sep)
        print(f"Epoch {epoch} done")
        print(f"Best Train Acc: {tr_acc}")
        print(f"Best Val Acc: {val_acc}")

