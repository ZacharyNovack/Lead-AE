# Lead-AE
This repository contains the official implementation of [Unsupervised Lead Sheet Generation via Semantic Compression](https://arxiv.org/abs/2310.10772). If you find this repository useful or use this code in your research, please cite the following paper: 

> Zachary Novack, Nikita Srivatsan, Taylor Berg-Kirkpatrick, and Julian McAuley. Unsupervised Lead Sheet Generation via Semantic Compression. 2023.
```
@misc{novack2023unsupervised,
    title={Unsupervised Lead Sheet Generation via Semantic Compression},
    author={Novack, Zachary and Srivatsan, Nikita and Berg-Kirkpatrick, Taylor and McAuley, Julian},
    year={2023},
    eprint={2310.10772},
    archivePrefix={arXiv},
}
```

## Content

- [Prerequisites](#prerequisites)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)

## Prerequisites

We recommend using conda for all dependency management. You can create a conda environment with the relative prerequisites using the following command:

```sh
conda env create -f environment.yml
```

## Preprocessing

Please download the [Symbolic orchestral database (SOD)](https://qsdfo.github.io/LOP/database.html). You may download it via command line as follows.

```sh
wget https://qsdfo.github.io/LOP/database/SOD.zip
```

### Prepare the name list

Get a list of filenames for each dataset.

```sh
find data/sod/SOD -type f -name *.mid -o -name *.xml | cut -c 14- > data/sod/original-names.txt
```

> Note: Change the number in the cut command for different datasets.

### Convert the data

Convert the MIDI and MusicXML files into MusPy files for processing.

```sh
python convert.py -n data/sod/original-names.txt -i data/sod -o data/sod/processed/json
```

> Note: You may enable multiprocessing with the `-j` option, for example, `python convert.py -j 10` for 10 parallel jobs.

Additionally, to extract the chords from each file.

```sh
python convert_chords.py -n data/sod/original-names.txt -i data/sod -o data/sod/processed/json
```

### Extract the note list

Extract a list of notes from the MusPy JSON files.

```sh
python extract.py -d sod
```

### Split training/validation/test sets

Split the processed data into training, validation and test sets.

```sh
python split.py -d sod
```

### (POP909 Specific Setup)

In order to set up the [POP909 Dataset](https://github.com/music-x-lab/POP909-Dataset/tree/master), please clone the repo locally on your machine. Then working through the `notebooks/parse_pop909.ipynb` will process the dataset for training and evaluation.


## Training

To train a baseline Lead-AE model without any warm-starting you can run:

```sh
train_transformer.py -d sod -l 4 --heads 8 --dim 512 -lr 0.01 -ct onset -k 1 -ich -st 
```

Notably, the `-k` field will set the constraint parameter for each onset, with any integer value denoting that number of notes per onset and any fractional value in (0, 1) denoting the fraction of notes in each onset to keep (see `train_transformer.py` for a more detailed description of each field).

In order to warm-start the decoder, we first run the training script with the `-ct` flag set to `skyline` as follows:

```sh
train_transformer.py -d sod -l 4 --heads 8 --dim 512 -lr 0.01 -ct skyline -ich -st 
```

Which will generate the baseline skyline model. We can then run the `pretrain_encoder.py` script to warm start the encoder of Lead-AE (with the skyline algorithm as the target):

```sh
pretrain_encoder.py -d sod -l 4 --heads 8 --dim 512 -lr 0.01 -pe skyline -ct onset -k 1 -ich
```

Using these two pretrained modules, you can then run `train_transformer.py` with setting the `-ech` and `-dch` flags to the paths of the pretrained encoder and decoder respectively.


## Evaluation

To run the evaluation results you can run:

```sh
evaluate.py -d sod -l 4 --heads 8 --dim 512 -lr 0.01 -ct onset -k 1 -ich -st -mch ["PATH-TO-YOUR-TRAINED-MODEL"] -evm all
```
> Notably, the command-line arguments for the model construction should match the arguments from the training script, especially for the value of `-k`. 


---

A special thanks to Hao-Wen Dong, whose repo for [Multitrack Music Transformer](https://github.com/salu133445/mmt/tree/main) is used as the basis for much of this implementation.

