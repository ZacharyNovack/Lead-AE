"""Convert MIDI and MusicXML files into music JSON files."""
import argparse
import logging
import pathlib
import pprint
import sys

import joblib
import os
import muspy
import tqdm

import utils
import chorder
import miditoolkit
from icecream import ic
import json
import numpy as np
import representation

qual2idx = {
    'M': 1,
    'm': 2,
    'o': 3,
    '+': 4,
    '7': 5,
    'M7': 6,
    'm7': 7,
    'o7': 8,
    '/o7': 9,
    'sus2': 10,
    'sus4': 11,
    None: 12
}

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert MIDI and MusicXML files into music JSON files."
    )
    parser.add_argument(
        "-n",
        "--names",
        default="data/sod/processed/names.txt",
        type=pathlib.Path,
        help="input names",
    )
    parser.add_argument(
        "-i",
        "--in_dir",
        default="data/sod/processed/notes",
        type=pathlib.Path,
        help="input data directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="data/sod/processed/json/",
        type=pathlib.Path,
        help="output directory",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        default=12,
        type=int,
        help="number of time steps per quarter note",
    )
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="whether to skip existing outputs",
    )
    parser.add_argument(
        "-e",
        "--ignore_exceptions",
        action="store_true",
        help="whether to ignore all exceptions",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="number of jobs"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    parser.add_argument(
        "-p", "--pitchclass", action="store_true", help="12 tone representation"
    )
    return parser.parse_args(args=args, namespace=namespace)


def adjust_resolution(music, resolution):
    """Adjust the resolution of the music."""
    music.adjust_resolution(resolution)
    for track in music:
        for note in track:
            if note.duration == 0:
                note.duration = 1
    music.remove_duplicate()

encoding = representation.get_encoding()


def convert(name, in_dir, out_dir, resolution, skip_existing):
    """Convert MIDI and MusicXML files into MusPy JSON files."""
    # Get output filename
    out_name = name.split(".")[0]
    out_filename = out_dir / f"{out_name}.json"

    # Skip if the output file exists
    if skip_existing and out_filename.is_file():
        return

    # Read the MIDI file
    notes = np.load(in_dir / f"{name}.npy")
    seq = representation.encode_notes(notes, encoding)
    music = representation.decode(seq, encoding)

    # Adjust the resolution
    adjust_resolution(music, resolution)

    # make tmp midi file from music object
    try:
        music.write_midi('tmp.mid', backend='mido')
    except:
        print('writing error for ', name, '...skipping')
        return
    try:
        mid_chords = miditoolkit.midi.parser.MidiFile('tmp.mid')
        chords = chorder.Dechorder.dechord(mid_chords)
        chords = {i: (x.root_pc+1 if x.root_pc is not None else 13, qual2idx[x.quality], x.bass_pc+1 if x.bass_pc is not None else 13) for i,x  in enumerate(chords)}
    except:
        print('read tmp error for ', name, '...skipping')
        os.remove('tmp.mid')
        return

    # delete tmp midi file
    os.remove('tmp.mid')

    # Filter bad files
    end_time = music.get_end_time()
    if end_time > resolution * 4 * 2000 or end_time < resolution * 4 * 10:
        pass

    # Save as a MusPy JSON file
    with open (out_dir / f"{out_name}_chords.json", 'w') as f:
        json.dump(chords, f)

    return out_name


@utils.ignore_exceptions
def convert_ignore_expections(
    name, in_dir, out_dir, resolution, skip_existing
):
    """Convert MIDI files into music JSON files, ignoring all expections."""
    return convert(name, in_dir, out_dir, resolution, skip_existing)


def process(
    name, in_dir, out_dir, resolution, skip_existing, ignore_exceptions=True
):
    """Wrapper for multiprocessing."""
    if ignore_exceptions:
        return convert_ignore_expections(
            name, in_dir, out_dir, resolution, skip_existing
        )
    return convert(name, in_dir, out_dir, resolution, skip_existing)


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()

    # Make sure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    args.out_dir.mkdir(exist_ok=True)

    # Set up the logger
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Get names
    logging.info("Loading names...")
    names = utils.load_txt(args.names)

    # Iterate over names
    logging.info("Iterating over names...")
    if args.jobs == 1:
        converted_names = []
        for name in (pbar := tqdm.tqdm(names)):
            pbar.set_postfix_str(name)
            result = process(
                name,
                args.in_dir,
                args.out_dir,
                args.resolution,
                args.skip_existing,
                args.ignore_exceptions,
            )
            if result is not None:
                converted_names.append(result)
    else:
        results = joblib.Parallel(
            n_jobs=args.jobs, verbose=0 if args.quiet else 5
        )(
            joblib.delayed(process)(
                name,
                args.in_dir,
                args.out_dir,
                args.resolution,
                args.skip_existing,
                args.ignore_exceptions,
            )
            for name in names
        )
        converted_names = [result for result in results if result is not None]
    logging.info(
        f"Converted {len(converted_names)} out of {len(names)} files."
    )

    # Save successfully converted names
    out_filename = args.out_dir.parent / "json-names.txt"
    utils.save_txt(out_filename, converted_names)
    logging.info(f"Saved the converted filenames to: {out_filename}")


if __name__ == "__main__":
    main()