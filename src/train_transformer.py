import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utils import *
from models import *
from mmt_dataset import *
import representation
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
        choices=("sod", 'pop909'),
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
        "-sb", "--strip_beats", action="store_true", help="strip beat tokens in target sequence"
    )
    parser.add_argument(
        "-fd", "--freeze_decoder", action="store_true", help="freeze decoder"
    )
    parser.add_argument(
        "-fe", "--freeze_encoder", action="store_true", help="freeze encoder"
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
        "-ech", "--encoder_checkpoint", type=str, default=None, help="encoder checkpoint"
    )
    parser.add_argument(
        "-mch", "--model_checkpoint", type=str, default=None, help="model checkpoint"
    )
    parser.add_argument(
        "-k", "--k", type=float, default=1, help="if k greater than 1, use topk scorer"
    )
    parser.add_argument(
        "-st", "--st",  action="store_true", help="enforce reduction to single track"
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




def train(data_loader, model, optimizer, scheduler, plot_freq=10, best_acc=None, trstep=0, encoding=None):
    model.train()

    plot_loss, tot_loss, tp_loss, bt_loss, pos_loss, pit_loss, dur_loss, instr_loss, sparsities, chord_sparsities = [], [], [], [], [], [], [], [], [], []
    n_batch = 0
    plot_acc, tp_acc, bt_acc, pos_acc, pit_acc, dur_acc, instr_acc = [], [], [], [], [], [], []
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        inp = batch["seq"].to(device)
        mask = batch["mask"].to(device)
        inp_lens = batch["seq_len"].to(device)
        losses, accs = model(inp, batch, mask=mask, encoding=encoding, st=args.st, ich=args.inter_chord)
        typs_loss, beats_loss, poss_loss, pits_loss, durs_loss, instrs_loss = losses['recon']
        typs_acc, beats_acc, poss_acc, pits_acc, durs_acc, instrs_acc = [(x.sum(-1) / (inp_lens - 1)).mean() for x in accs]
        acc = (reduce(lambda x, y: x * y, accs).sum(dim=-1) / (inp_lens - 1)).mean()
        recon_loss = sum(losses['recon'])
        sparsity = losses['sparsity'] if losses.get('sparsity') is not None else None
        
        loss = recon_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()




        # Update plotting variables
        plot_loss.append(recon_loss.detach().cpu().item())
        tot_loss.append(loss.detach().cpu().item())
        sparsities.append(sparsity.detach().cpu().item() if sparsity is not None else 0)
        chord_sparsities.append(losses['chord_sparsity'].detach().cpu().item() if 'chord_sparsity' in losses else 0)
        tp_loss.append(typs_loss.detach().cpu().item())
        bt_loss.append(beats_loss.detach().cpu().item())
        pos_loss.append(poss_loss.detach().cpu().item())
        pit_loss.append(pits_loss.detach().cpu().item())
        dur_loss.append(durs_loss.detach().cpu().item())
        instr_loss.append(instrs_loss.detach().cpu().item())
        plot_acc.append(acc.detach().cpu().item())
        tp_acc.append(typs_acc.detach().cpu().item())
        bt_acc.append(beats_acc.detach().cpu().item())
        pos_acc.append(poss_acc.detach().cpu().item())
        pit_acc.append(pits_acc.detach().cpu().item())
        dur_acc.append(durs_acc.detach().cpu().item())
        instr_acc.append(instrs_acc.detach().cpu().item())

        
        n_batch += 1
        if i % plot_freq == 0 or i == len(data_loader) - 1:
            print(f"Acc: {np.mean(plot_acc):.4f}  Total Loss: {np.mean(tot_loss):.4f} Sparsity: {np.mean(sparsities):.4f} Chord Sparsity: {np.mean(chord_sparsities):.4f}")
            if args.logging:
                
                wandb.log({'train/loss': np.mean(plot_loss), 'custom_step': trstep})
                wandb.log({'train/tot_loss': np.mean(tot_loss), 'custom_step': trstep})
                wandb.log({'train/avg_latent_sparsity': np.mean(sparsities), 'custom_step': trstep})
                wandb.log({'train/avg_chord_sparsity': np.mean(chord_sparsities), 'custom_step': trstep})
                wandb.log({'train/tp_loss': np.mean(tp_loss), 'custom_step': trstep})
                wandb.log({'train/bt_loss': np.mean(bt_loss), 'custom_step': trstep})
                wandb.log({'train/pos_loss': np.mean(pos_loss), 'custom_step': trstep})
                wandb.log({'train/pit_loss': np.mean(pit_loss), 'custom_step': trstep})
                wandb.log({'train/dur_loss': np.mean(dur_loss), 'custom_step': trstep})
                wandb.log({'train/instr_loss': np.mean(instr_loss), 'custom_step': trstep})
                wandb.log({'train/acc': np.mean(plot_acc), 'custom_step': trstep})
                wandb.log({'train/tp_acc': np.mean(tp_acc), 'custom_step': trstep})
                wandb.log({'train/bt_acc': np.mean(bt_acc), 'custom_step': trstep})
                wandb.log({'train/pos_acc': np.mean(pos_acc), 'custom_step': trstep})
                wandb.log({'train/pit_acc': np.mean(pit_acc), 'custom_step': trstep})
                wandb.log({'train/dur_acc': np.mean(dur_acc), 'custom_step': trstep})
                wandb.log({'train/instr_acc': np.mean(instr_acc), 'custom_step': trstep})
            if best_acc is None or np.mean(plot_acc) > best_acc:
               
                best_acc = np.mean(plot_acc)
                if args.logging:
                    print(f"New best acc: {np.mean(plot_acc):.4f}")
                    if not os.path.exists(f"../logs/{args.dataset}/{run.name}"):
                        os.mkdir(f"../logs/{args.dataset}/{run.name}")
                    sav_d = {
                        'model': model.state_dict(),
                        'acc': best_acc
                    }
                    torch.save(sav_d, f"../logs/{args.dataset}/{run.name}/tr_model.pt")
                    del sav_d
            plot_loss, tot_loss, tp_loss, bt_loss, pos_loss, pit_loss, dur_loss, instr_loss = [], [], [], [], [], [], [], []
            sparsities = []
            chord_sparsities = []
            n_batch = 0
            plot_acc, tp_acc, bt_acc, pos_acc, pit_acc, dur_acc, instr_acc = [], [], [], [], [], [], []
        trstep += 1
        
    
            
    return model, best_acc, trstep
        

def evaluate(data_loader, encoding, model, trstep=0, epoch=0, best_acc=None):
    model.eval()

    plot_loss = 0
    tot_loss = 0
    sparsity = 0
    tp_loss = 0
    bt_loss = 0
    pos_loss = 0
    pit_loss = 0
    dur_loss = 0
    instr_loss = 0
    plot_acc = 0
    tp_acc = 0
    bt_acc = 0
    pos_acc = 0
    pit_acc = 0
    dur_acc = 0
    instr_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inp = batch["seq"].to(device)
            mask = batch["mask"].to(device)
            inp_lens = batch["seq_len"].to(device)
            losses, accs = model(inp, batch, mask=mask, encoding=encoding, st=args.st, ich=args.inter_chord)
            typs_loss, beats_loss, poss_loss, pits_loss, durs_loss, instrs_loss = losses['recon']
            typs_acc, beats_acc, poss_acc, pits_acc, durs_acc, instrs_acc = [(x.sum(-1) / (inp_lens - 1)).mean() for x in accs]
            acc = (reduce(lambda x, y: x * y, accs).sum(dim=-1) / (inp_lens - 1)).mean()
            recon_loss = sum(losses['recon'])
            loss = recon_loss
            plot_loss += recon_loss.detach().cpu().item()
            tot_loss += loss.detach().cpu().item()

            sparsity += losses['sparsity'].detach().cpu().item() if 'sparsity' in losses else 0
            tp_loss += typs_loss.detach().cpu().item()
            bt_loss += beats_loss.detach().cpu().item()
            pos_loss += poss_loss.detach().cpu().item()
            pit_loss += pits_loss.detach().cpu().item()
            dur_loss += durs_loss.detach().cpu().item()
            instr_loss += instrs_loss.detach().cpu().item()
            plot_acc += acc.detach().cpu().item()
            tp_acc += typs_acc.detach().cpu().item()
            bt_acc += beats_acc.detach().cpu().item()
            pos_acc += poss_acc.detach().cpu().item()
            pit_acc += pits_acc.detach().cpu().item()
            dur_acc += durs_acc.detach().cpu().item()
            instr_acc += instrs_acc.detach().cpu().item()
            

        if args.logging:
            wandb.log({'val/loss': plot_loss / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/avg_latent_sparsity': sparsity / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/tp_loss': tp_loss / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/bt_loss': bt_loss / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/pos_loss': pos_loss / len(data_loader), 'custom_step': trstep,'epoch': epoch})
            wandb.log({'val/pit_loss': pit_loss / len(data_loader), 'custom_step': trstep,'epoch': epoch})
            wandb.log({'val/dur_loss': dur_loss / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/instr_loss': instr_loss /len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/acc': plot_acc / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/tp_acc': tp_acc / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/bt_acc': bt_acc / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/pos_acc': pos_acc / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/pit_acc': pit_acc / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/dur_acc': dur_acc / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
            wandb.log({'val/instr_acc': instr_acc / len(data_loader), 'custom_step': trstep, 'epoch': epoch})
        if best_acc is None or plot_acc / len(data_loader) > best_acc:
            
            best_acc = plot_acc / len(data_loader)
            if args.logging:
                print(f"New best acc: {plot_acc / len(data_loader):.4f}")
                if not os.path.exists(f"../logs/{args.dataset}/{run.name}"):
                    os.mkdir(f"../logs/{args.dataset}/{run.name}")
                sav_d = {
                    'model': model.state_dict(),
                    'acc': best_acc
                }
                torch.save(sav_d, f"../logs/{args.dataset}/{run.name}/val_model.pt")
                del sav_d

    return best_acc, frozen_fields


def init_run(args):
    if not os.path.exists(f"../logs/{args.dataset}"):
        os.mkdir(f"../logs/{args.dataset}")
    run = wandb.init(
        project=f'{args.dataset}-Train',
        config=args,
        mode="disabled" if args.logging == False else "online"
    )
    if args.logging:
        if args.conv_type == 'skyline':
            run.name = f"SKY-{args.learning_rate}-{args.layers}-{args.heads}-{args.dim}-" + run.name.split("-")[-1]
        else:
            run.name = f"AE-{args.learning_rate}-{args.layers}-{args.heads}-{args.dim}-" + run.name.split("-")[-1]
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
    if args.inter_chord:
        kwargs['max_seq_len'] += args.max_beat
    if args.conv_type in ['onset', 'beat']:
        scorer = MusicXScorerTopK(
            conv_type=args.conv_type,
            k=args.k,
            temp=1,
            **kwargs
        ).to(device)
    elif args.conv_type == 'skyline': # fixed encoder w/skyline algorithm
        scorer = MusicXSkyline()
    else:
        raise NotImplementedError

    s2s = MusicXTransformer(
        encoder=True,
        decoder=True,
        ich=args.inter_chord,
        **kwargs
    ).to(device)
    if args.freeze_decoder:
        # only train the scorer
        for n, p in s2s.named_parameters():
            p.requires_grad = False
    model = MusicXLeadAE(
        scorer=scorer,
        transformer=s2s,
        **kwargs
    ).to(device)
    if args.encoder_checkpoint is not None and args.decoder_checkpoint is not None: 
        #  loading decoder from decoder checkpoint and encoder from encoder checkpoint
        state_dict = torch.load(args.encoder_checkpoint, map_location=device)
        for n, p in model.scorer.named_parameters():
            if n in state_dict:
                p.data = state_dict[n].data

                if args.freeze_encoder or 'score' not in n:
                    p.requires_grad = False
        state_dict = torch.load(args.decoder_checkpoint,  map_location=device)
        if 'acc' in state_dict:
            print('loaded model best acc')
        else:
            tmp_sd = {}
            tmp_sd['model'] = state_dict
            state_dict = tmp_sd
        for n, p in model.named_parameters():
            if n in state_dict['model'] and 'seq2seq' in n:
                p.data = state_dict['model'][n].data
    elif args.encoder_checkpoint is not None:
        state_dict = torch.load(args.encoder_checkpoint, map_location=device)
        for n, p in model.scorer.named_parameters():
            if n in state_dict:
                p.data = state_dict[n].data
                if args.freeze_encoder:
                    p.requires_grad = False
    elif args.decoder_checkpoint is not None:
        state_dict = torch.load(args.decoder_checkpoint)
        for n, p in model.named_parameters():
            if n in state_dict['model']:
                p.data = state_dict['model'][n].data
                if args.freeze_decoder:
                    p.requires_grad = False
    if args.model_checkpoint is not None:
        state_dict = torch.load(args.model_checkpoint)
        if 'acc' in state_dict:
            print('loaded model best acc')
        else:
            tmp_sd = {}
            tmp_sd['model'] = state_dict
            state_dict = tmp_sd
        
        for n, p in model.named_parameters():
            if n in state_dict['model']:
                p.data = state_dict['model'][n].data
                if 'scorer' in n and args.freeze_encoder:
                    p.requires_grad = False

    return model


ARGDATA2PATH = {
    'sod': 'sod/processed',
    'pop909': 'pop909'
}


if __name__ == "__main__":
    args = parse_args()
    args.logging = True if args.logging == "True" else False
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
    print(len(model.seq2seq.encoder.token_emb), len(model.seq2seq.decoder.net.token_emb))
    print(model.seq2seq.encoder.max_seq_len, model.seq2seq.decoder.net.max_seq_len)
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
        model, tr_acc, step = train(train_data_loader, model, optimizer, scheduler, plot_freq=args.plot_freq, best_acc=tr_acc, trstep=step, encoding=encoding)
        val_acc, frozen_fields = evaluate(valid_data_loader, encoding, model, trstep=step, best_acc=val_acc, epoch=epoch)
        print(f"Epoch {epoch} done")
        print(f"Best Train Acc: {tr_acc}")
        print(f"Best Val Acc: {val_acc}")

