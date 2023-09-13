"""Data loader."""
import argparse
import logging
import pathlib
import pprint
import sys

import numpy as np
import torch
import torch.utils.data
sys.path.append('..')
import src.representation as representation
import src.utils as utils
from collections import OrderedDict
import json


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd", 'JSBach/train', 'JSBach/valid', 'JSBach/test'),
        required=True,
        help="dataset key",
    )
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
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
    # Others
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        help="number of jobs (deafult to `min(batch_size, 8)`)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)

def pad(data, maxlen=None, encoding=None):
    if maxlen is None:
        max_len = max(len(x) for x in data)
    else:
        for x in data:
            assert len(x) <= max_len
    if data[0].ndim == 1:
        padded = [np.pad(x, (0, max_len - len(x))) for x in data]
    elif data[0].ndim == 2:
        max_wid = max([x.shape[1] for x in data])
        padded = [np.pad(x, ((0, max_len - len(x)), (0, max_wid - x.shape[1]))) for x in data]
    else:
        raise ValueError("Got 3D data.")
    return np.stack(padded)

def get_mask(data):
    max_seq_len = max(len(sample) for sample in data)
    mask = torch.zeros((len(data), max_seq_len), dtype=torch.bool)
    for i, seq in enumerate(data):
        mask[i, : len(seq)] = 1
    return mask


class MusicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        data_dir,
        encoding,
        max_seq_len=None,
        max_bt_seq_len=None,
        max_beat=None,
        use_csv=False,
        use_augmentation=False,
        window_type='onset',
        inter_chord=True,
    ):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        with open(filename) as f:
            self.names = [line.strip() for line in f if line]
        if self.names[0].endswith(".json"):
            self.names = [name[:-5] for name in self.names]
        self.encoding = encoding
        self.max_seq_len = max_seq_len
        self.max_bt_seq_len = max_bt_seq_len
        self.max_beat = max_beat
        self.use_csv = use_csv
        self.use_augmentation = use_augmentation
        self.window_type = window_type
        self.inter_chord = inter_chord # directly interleave chords rather than using a separate channel

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # Get the name
        name = self.names[idx]

        # Load data
        if self.use_csv:
            notes = utils.load_csv(self.data_dir / f"{name}.csv")
        else:
            notes = np.load(self.data_dir / f"{name}.npy")

        # Check the shape of the loaded notes
        assert notes.shape[1] == 5

        # Data augmentation
        start_beat = 0
        if self.use_augmentation:
            # Shift all the pitches for k semitones (k~Uniform(-5, 6))
            pitch_shift = np.random.randint(-5, 7)
            notes[:, 2] = np.clip(notes[:, 2] + pitch_shift, 0, 127)
            if self.pitchclass:
                notes[:, 2] = notes[:, 2] % 12

            # Randomly select a starting beat
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                trial = 0
                # Avoid section with too few notes
                while trial < 10:
                    start_beat = np.random.randint(n_beats - self.max_beat)
                    end_beat = start_beat + self.max_beat
                    sliced_notes = notes[
                        (notes[:, 0] >= start_beat) & (notes[:, 0] < end_beat)
                    ]
                    if len(sliced_notes) > 10:
                        break
                    trial += 1
                sliced_notes[:, 0] = sliced_notes[:, 0] - start_beat
                notes = sliced_notes
                

        # Trim sequence to max_beat
        elif self.max_beat is not None:
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                notes = notes[notes[:, 0] < self.max_beat]
        

        # Encode the notes
        seq = representation.encode_notes(notes, self.encoding)
        bt_seq = representation.encode_notes(notes, self.encoding, bt_tok=self.window_type)
        if self.inter_chord:
            chord_seq = representation.encode_notes(notes, self.encoding, bt_tok='beat')
            with open(f"{self.data_dir}/../json/{name}_chords.json", "r") as f:
                chords = json.load(f)
                chords = {int(i)-start_beat: chord for i, chord in chords.items() if int(i) >= start_beat}

                for ix, ch in chords.items():
                    if ch[0] not in (0, 13):
                        assert ch[-1] not in (0, 13), f"{ch} {ix}"
                        if self.use_augmentation:
                            chords[ix] = [(ch[0] - 1 + pitch_shift) % 12 + 1 + self.encoding['n_tokens'][3]-14, ch[1] + self.encoding['n_tokens'][4] - 13]
                        else:
                            chords[ix] = [ch[0] + self.encoding['n_tokens'][3]-14, ch[1] + self.encoding['n_tokens'][4] - 13]
            targ_seq = chord_seq[chord_seq[:, 0] == 5]
            targ_seq[:, 3:5] = torch.tensor([chords.get(x-1, [0,0,0])[:2] for x in chord_seq[chord_seq[:, 0] == 5][:, 1]])
            chord_seq[chord_seq[:, 0] == 5] = targ_seq



        # Add 12-tone note representation
        if self.add12tone:
            notes[:, 2] = notes[:, 2] % 12
            seq12 = representation12.encode_notes(notes, self.encoding)
            seq = np.concatenate((seq, seq12[:, 2].reshape(-1, 1)), axis=1)

        # Trim sequence to max_seq_len
        if (self.max_bt_seq_len is not None and len(bt_seq) > self.max_bt_seq_len):
            bt_seq = np.concatenate((bt_seq[: self.max_bt_seq_len-1], bt_seq[-1:]))
            last_note = bt_seq[-2]
            if last_note[0] == 5:
                bt_seq = np.concatenate([bt_seq[:-2], bt_seq[-1:]])
                last_note = bt_seq[-2]
            matching_in_seq = np.where((seq == last_note).sum(axis=-1) == 6)[0][0]
            seq = np.concatenate((seq[:matching_in_seq+1], seq[-1:]))
            if self.inter_chord:
                matching_in_seq = np.where((chord_seq == last_note).sum(axis=-1) == 6)[0][0]
                chord_seq = np.concatenate((chord_seq[:matching_in_seq+1], chord_seq[-1:]))


        if self.window_type == 'onset':
            wind_ix = 3
            div_ix = 1
        elif self.window_type == 'beat':
            wind_ix = 2
            div_ix = 1
        elif self.window_type == 'bar4':
            wind_ix = 2
            div_ix = 4
        elif self.window_type == 'bar2':
            wind_ix = 2
            div_ix = 2
        ons_seq = chord_seq.copy()
        ons_seq[:, 1] = ons_seq[:, 1] // div_ix
        ons_seq[:, 0] = - np.abs(ons_seq[:, 0] - 4) + 4
        old_onsets = set([tuple(x) for x in np.unique(ons_seq[:, :wind_ix], axis=0)])
        onsetdict = OrderedDict((onset, np.where((ons_seq[:, :wind_ix] == onset).all(axis=1))[0]) for onset in sorted(old_onsets))

        return {"name": name, "seq": seq, "onsetdict": onsetdict, 'old_onsets': old_onsets, 'chord_seq': chord_seq if self.inter_chord else None}

    # @classmethod
    def collate(cls, data):
        seq = [sample["seq"] for sample in data]
        old_onsets = [sample["old_onsets"] for sample in data]
        onsetdict = [sample["onsetdict"] for sample in data]
        chord_seq = [sample["chord_seq"] for sample in data] if cls.inter_chord else None
        ons_masks = [get_ons_mask(sample["onsetdict"]) for sample in data]
        ons_lens = [ons_mask[1] for ons_mask in ons_masks]
        ons_d = [ons_mask[2] for ons_mask in ons_masks]
        ons_d_k = np.concatenate([np.array([(ix, *k) for k in d.keys()]) for ix, d in enumerate(ons_d)])
        ons_d_k = torch.tensor(ons_d_k, dtype=torch.long)
        ons_d_v = np.concatenate([np.array([(ix, v) for v in d.values()]) for ix, d in enumerate(ons_d)])
        ons_d_v = torch.tensor(ons_d_v, dtype=torch.long)
        ons_masks = [ons_mask[0] for ons_mask in ons_masks]
        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(seq, encoding=cls.encoding), dtype=torch.long),
            "seq_len": torch.tensor([len(s) for s in seq], dtype=torch.long),
            "mask": get_mask(seq),
            "chord_seq": torch.tensor(pad(chord_seq, encoding=cls.encoding), dtype=torch.long) if cls.inter_chord else None,
            "chord_seq_len": torch.tensor([len(s) for s in chord_seq], dtype=torch.long) if cls.inter_chord else None,
            "chord_mask": get_mask(chord_seq) if cls.inter_chord else None,
            "ons_masks": torch.tensor(pad(ons_masks), dtype=torch.float),
            "ons_lens": [torch.tensor(x, dtype=torch.long) for x in ons_lens],
            "ons_d_k": [ons_d_k[:, ix] for ix in range(ons_d_k.shape[1])],
            "ons_d_v": [ons_d_v[:, ix] for ix in range(ons_d_v.shape[1])],
            "onsetdict": onsetdict,
            'old_onsets': old_onsets,
        }



class POP909Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        data_dir,
        encoding,
        max_seq_len=None,
        max_bt_seq_len=None,
        max_beat=None,
        use_csv=False,
        use_augmentation=False,
        window_type='onset',
        inter_chord=False,
    ):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        with open(filename) as f:
            self.names = [line.strip() for line in f if line]
        if self.names[0].endswith(".json"):
            self.names = [name[:-5] for name in self.names]
        self.encoding = encoding
        self.max_seq_len = max_seq_len
        self.max_bt_seq_len = max_bt_seq_len
        self.max_beat = max_beat
        self.use_csv = use_csv
        self.use_augmentation = use_augmentation
        self.window_type = window_type
        self.inter_chord = inter_chord # directly interleave chords rather than using a separate channel

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # Get the name
        name = self.names[idx]

        # Load data
        if self.use_csv:
            notes = utils.load_csv(self.data_dir / f"{name}.csv")
        else:
            notes = np.load(self.data_dir / f"{name}_notes.npy")

        # Check the shape of the loaded notes
        assert notes.shape[1] == 5

        # Data augmentation
        start_beat = 0
        if self.use_augmentation:
            # Shift all the pitches for k semitones (k~Uniform(-5, 6))
            pitch_shift = np.random.randint(-5, 7)
            notes[:, 2] = np.clip(notes[:, 2] + pitch_shift, 0, 127)
            if self.pitchclass:
                notes[:, 2] = notes[:, 2] % 12

            # Randomly select a starting beat
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                trial = 0
                # Avoid section with too few notes
                while trial < 10:
                    start_beat = np.random.randint(n_beats - self.max_beat)
                    end_beat = start_beat + self.max_beat
                    sliced_notes = notes[
                        (notes[:, 0] >= start_beat) & (notes[:, 0] < end_beat)
                    ]
                    if len(sliced_notes) > 10:
                        break
                    trial += 1
                sliced_notes[:, 0] = sliced_notes[:, 0] - start_beat
                notes = sliced_notes
                

        # Trim sequence to max_beat
        elif self.max_beat is not None:
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                notes = notes[notes[:, 0] < self.max_beat]
        

        # Encode the notes
        seq = representation.encode_notes(notes, self.encoding)
        bt_seq = representation.encode_notes(notes, self.encoding, bt_tok=self.window_type)
        if self.inter_chord:
            chord_seq = representation.encode_notes(notes, self.encoding, bt_tok='beat')
            with open(f"{self.data_dir}/{name}_chord.json", "r") as f:
                chords = json.load(f)
                chords = {int(i)-start_beat: chord for i, chord in chords.items() if int(i) >= start_beat}

                for ix, ch in chords.items():
                    if ch[0] not in (0, 13):
                        if self.use_augmentation:
                            chords[ix] = [(ch[0] - 1 + pitch_shift) % 12 + 1 + self.encoding['n_tokens'][3]-14, ch[1] + self.encoding['n_tokens'][4] - 13]
                        else:
                            chords[ix] = [ch[0] + self.encoding['n_tokens'][3]-14, ch[1] + self.encoding['n_tokens'][4] - 13]
            targ_seq = chord_seq[chord_seq[:, 0] == 5]
            targ_seq[:, 3:5] = torch.tensor([chords.get(x-1, [0,0,0])[:2] for x in chord_seq[chord_seq[:, 0] == 5][:, 1]])
            chord_seq[chord_seq[:, 0] == 5] = targ_seq
            chord_seq = chord_seq[~np.logical_and(chord_seq[:, 0] == 5, chord_seq[:, 3] == 0)]


        # Trim sequence to max_seq_len
        if (self.max_bt_seq_len is not None and len(bt_seq) > self.max_bt_seq_len):
            bt_seq = np.concatenate((bt_seq[: self.max_bt_seq_len-1], bt_seq[-1:]))
            last_note = bt_seq[-2]
            if last_note[0] == 5:
                bt_seq = np.concatenate([bt_seq[:-2], bt_seq[-1:]])
                last_note = bt_seq[-2]
            matching_in_seq = np.where((seq == last_note).sum(axis=-1) == 6)[0][0]
            seq = np.concatenate((seq[:matching_in_seq+1], seq[-1:]))
            if self.inter_chord:
                matching_in_seq = np.where((chord_seq == last_note).sum(axis=-1) == 6)[0][0]
                chord_seq = np.concatenate((chord_seq[:matching_in_seq+1], chord_seq[-1:]))

        if self.window_type == 'onset':
            wind_ix = 3
            div_ix = 1
        elif self.window_type == 'beat':
            wind_ix = 2
            div_ix = 1
        elif self.window_type == 'bar4':
            wind_ix = 2
            div_ix = 4
        elif self.window_type == 'bar2':
            wind_ix = 2
            div_ix = 2
        ons_seq = chord_seq.copy()
        ons_seq[:, 1] = ons_seq[:, 1] // div_ix
        ons_seq[:, 0] = - np.abs(ons_seq[:, 0] - 4) + 4
        old_onsets = set([tuple(x) for x in np.unique(ons_seq[:, :wind_ix], axis=0)])
        onsetdict = OrderedDict((onset, np.where((ons_seq[:, :wind_ix] == onset).all(axis=1))[0]) for onset in sorted(old_onsets))

        return {"name": name, "seq": seq, "onsetdict": onsetdict, 'old_onsets': old_onsets, 'chord_seq': chord_seq if self.inter_chord else None}

    # @classmethod
    def collate(cls, data):
        seq = [sample["seq"] for sample in data]
        old_onsets = [sample["old_onsets"] for sample in data]
        onsetdict = [sample["onsetdict"] for sample in data]
        chord_seq = [sample["chord_seq"] for sample in data] if cls.inter_chord else None
        ons_masks = [get_ons_mask(sample["onsetdict"]) for sample in data]
        ons_lens = [ons_mask[1] for ons_mask in ons_masks]
        ons_d = [ons_mask[2] for ons_mask in ons_masks]
        ons_d_k = np.concatenate([np.array([(ix, *k) for k in d.keys()]) for ix, d in enumerate(ons_d)])
        ons_d_k = torch.tensor(ons_d_k, dtype=torch.long)
        ons_d_v = np.concatenate([np.array([(ix, v) for v in d.values()]) for ix, d in enumerate(ons_d)])
        ons_d_v = torch.tensor(ons_d_v, dtype=torch.long)
        ons_masks = [ons_mask[0] for ons_mask in ons_masks]
        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(seq, encoding=cls.encoding), dtype=torch.long),
            "seq_len": torch.tensor([len(s) for s in seq], dtype=torch.long),
            "mask": get_mask(seq),
            "chord_seq": torch.tensor(pad(chord_seq, encoding=cls.encoding), dtype=torch.long) if cls.inter_chord else None,
            "chord_seq_len": torch.tensor([len(s) for s in chord_seq], dtype=torch.long) if cls.inter_chord else None,
            "chord_mask": get_mask(chord_seq) if cls.inter_chord else None,
            "ons_masks": torch.tensor(pad(ons_masks), dtype=torch.float),
            "ons_lens": [torch.tensor(x, dtype=torch.long) for x in ons_lens],
            "ons_d_k": [ons_d_k[:, ix] for ix in range(ons_d_k.shape[1])],
            "ons_d_v": [ons_d_v[:, ix] for ix in range(ons_d_v.shape[1])],
            "onsetdict": onsetdict,
            'old_onsets': old_onsets,
        }


def get_ons_mask(onsetdict, max_ons_density=200):
    '''
    For given onset indices, extract a mask of len(onsets) x max_ons_density, with the first density notes in the mask set to 1
    '''
    # ic(list(onsetdict.values())[1])
    mask = np.zeros((len(onsetdict)+len(list(onsetdict.values())[1])-1, max([len(y) for y in onsetdict.values()])))
    out_d = OrderedDict()
    ix = 0
    dix = 0
    lens = []
    for k,v in onsetdict.items():
        if k == (1, 0, 0) or k == (1, 0): # instrument section of preamble, we need to add all of them
            # ic(len(v))
            mask[ix:ix+len(v), 0] = 1
            out_d.update({(ix+i, 0): dix+i for i in range(len(v))})
            ix += len(v)
            lens.extend([1]*len(v))
        else:
            mask[ix, :len(v)] = 1
            out_d.update({(ix, i): dix+i for i in range(len(v))})
            ix += 1
            lens.append(len(v))
        dix += len(v)
    return mask, lens, out_d

pit2octs = {
    1: [1, 13, 25, 37, 49, 61, 73, 85, 97, 109, 121],
    2: [2, 14, 26, 38, 50, 62, 74, 86, 98, 110, 122],
    3: [3, 15, 27, 39, 51, 63, 75, 87, 99, 111, 123],
    4: [4, 16, 28, 40, 52, 64, 76, 88, 100, 112, 124],
    5: [5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125],
    6: [6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126],
    7: [7, 19, 31, 43, 55, 67, 79, 91, 103, 115, 127],
    8: [8, 20, 32, 44, 56, 68, 80, 92, 104, 116, 128],
    9: [9, 21, 33, 45, 57, 69, 81, 93, 105, 117],
    10: [10, 22, 34, 46, 58, 70, 82, 94, 106, 118],
    11: [11, 23, 35, 47, 59, 71, 83, 95, 107, 119],
    0: [12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
}
