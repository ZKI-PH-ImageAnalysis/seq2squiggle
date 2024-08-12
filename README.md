# seq2squiggle

`seq2squiggle` is a deep learning based-tool for generating artifical nanopore signals from DNA sequence data.

<img src="/img/seq2squiggle_architecture.png" width="750">


Please cite the following publication if you use `seq2squiggle` in your work:
- Beslic,  D., Kucklick, M., Engelmann, S., Fuchs, S., Renards, B.Y., Körber, N. End-to-end simulation of nanopore sequencing signals with feed-forward transformers. bioRxiv (2024).

## Installation 

### Dependencies

`seq2squiggle` runs under Python >= 3.11 and pytorch >= 2.0.1

### Installation with conda/mamba
Proceed as follows to install `seq2squiggle`

```
mamba create -f envs/seq2squiggle.yml
mamba activate seq2squiggle
```

### Download training data and model weights
**Not implemented yet**

When running `seq2squiggle` in prediction mode, `seq2squiggle` needs compatible pretrained model weights to make predictions. The model file can then be specified using the `--model` command-line parameter. To assist users, if no model file is specified `seq2squiggle` will try to download and use a compatible model file automatically.


## Predict signals from FASTA file
`seq2squiggle` simulates artificial signals based on an input FASTX file. By default, the output is in SLOW5/BLOW5 format. Exporting to the new POD5 format is also supported, though BLOW5 is preferred for its stability. You will need to specify the path to the model through the configuration file.

```
# Generate 10,000 reads from a fasta. 
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5 -n 10000

# Generate reads with a coverage of 30 from a fasta
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5 -c 30

# Generate reads with a coverage of 30 and an average read length of 5,000 from a fasta
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5 -c 30 -r 5000

# Simulate signals from basecalled reads (each complete read will be simulated)
./src/seq2squiggle/seq2squiggle.py predict example.fastq -o example.blow5 --read-input

# Export as pod5
./src/seq2squiggle/seq2squiggle.py predict example.fastq -o example.pod5 --read-input

```


## Different noise options
`seq2squiggle` supports different options for generating the signal data.
Per default, the noise sampler and duration sampler are used.

```
# Generate reads using both the noise sampler and duration sampler. 
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5

# Generate reads using the noise sampler with an increased factor and duration sampler
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5 --noise-std 1.5

# Generate reads using a normal distribution for the noise and duration sampler
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5 --noise-std 1.5 --noise-sampling False

# Generate reads using only the noise sampler and a normal distribution for the event length 
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5 --duration-sampling False --ideal-event-length -1

# Generate reads using only the noise sampler and ideal event lengths 
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5 --duration-sampling False --ideal-event-length 10.0

# Generate reads using a normal distribution for the noise and ideal event lengths
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5 --duration-sampling False --ideal-event-length 10.0 --noise-sampling False --noise-std 1.0

# Generate reads using no amplitude noise and ideal event lengths
./src/seq2squiggle/seq2squiggle.py predict example.fasta -o example.blow5 --duration-sampling False --ideal-event-length 10.0 --noise-sampling False --noise-std -1
```

## Train a new model
`seq2squiggle` uses the uncalled4's align output (events.tsv) as training data. 

Run the following commands to generate the data with [uncalled4](https://github.com/skovaka/uncalled4):
```
uncalled4 align REF_FASTA SLOW5 --bam-in INPUT_BAM --eventalign-out OUTPUT_TSV --eventalign-flags print-read-names,signal-index,samples --pore-model dna_r10.4.1_400bps_9mer --flowcell FLO-MIN114 --kit SQK-LSK114
```

Additionally, we use a small script to standardize the event_noise column:
```
./src/seq2squiggle/standardize-events.py INPUT_TSV OUTPUT_TSV
```

To preprocess and train a model from scratch:
```
./src/seq2squiggle/seq2squiggle.py preprocess events.tsv train_dir --max-chunks -1 --config my_config.yml
./src/seq2squiggle/seq2squiggle.py preprocess events_valid.tsv valid_dir --max-chunks -1 --config my_config.yml
./src/seq2squiggle/seq2squiggle.py train train_dir valid_dir --config my_config.yml --model last.ckpt
```

## Acknowledgement
The model is based on [xcmyz's implementation of FastSpeech](https://github.com/xcmyz/FastSpeech). Some code snippets for preprocessing DNA-signal chunks have been taken from [bonito](https://github.com/nanoporetech/bonito). 
