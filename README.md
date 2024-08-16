# seq2squiggle

`seq2squiggle` is a deep learning-based tool for generating artifical nanopore signals from DNA sequence data.

<img src="/img/seq2squiggle_architecture.png" width="750">


Please cite the following publication if you use `seq2squiggle` in your work:
- Beslic,  D., Kucklick, M., Engelmann, S., Fuchs, S., Renards, B.Y., Körber, N. End-to-end simulation of nanopore sequencing signals with feed-forward transformers. bioRxiv (2024).

## Installation 

### Dependencies

`seq2squiggle` requires Python >= 3.10. 

We recommend to run `seq2squiggle` in a separate conda / mamba environment. This keeps the tool and its dependencies isolated from your other Python environments.

```
conda create -n seq2squiggle-env python=3.10
conda activate seq2squiggle-env 
```

### Install with pip
```
pip install seq2squiggle 
```

### Install from source
```
git clone https://github.com/ZKI-PH-ImageAnalysis/seq2squiggle.git
cd seq2squiggle
pip install . 
```

### Download training data and model weights

`seq2squiggle` requires compatible pretrained model weights to make predictions, which can be specified using the `--model` command-line parameter.

If you do not provide a model file, `seq2squiggle` will automatically attempt to download a compatible model file to ensure predictions can be made. 

## Predict signals from FASTA file
`seq2squiggle` simulates artificial signals based on an input FASTX file. By default, the output is in SLOW5/BLOW5 format. Exporting to the new POD5 format is also supported, though BLOW5 is preferred for its stability. You will need to specify the path to the model through the configuration file.

For optimal performance, running `seq2squiggle` on a GPU is recommended, especially to speed up inference. However, the tool also works on CPU-only systems, though at a slower inference speed.

### Examples 

Generate 10,000 reads from a fasta file:
```
seq2squiggle predict example.fasta -o example.blow5 -n 10000
```
Generate reads with a coverage of 30:
```
seq2squiggle predict example.fasta -o example.blow5 -c 30
```
Generate reads with a coverage of 30 and an average read length of 5,000:
```
seq2squiggle predict example.fasta -o example.blow5 -c 30 -r 5000
```
Simulate signals from basecalled reads (each single read will be simulated):
```
seq2squiggle predict example.fastq -o example.blow5 --read-input
```
Export as pod5:
```
seq2squiggle predict example.fastq -o example.pod5 --read-input
```



## Different noise options
`seq2squiggle` supports different options for generating the signal data.
Per default, the noise sampler and duration sampler are used.

### Examples

Generate reads using both the noise sampler and duration sampler: 
```
seq2squiggle predict example.fasta -o example.blow5
```
Generate reads using the noise sampler with an increased factor and duration sampler:
```
seq2squiggle predict example.fasta -o example.blow5 --noise-std 1.5
```
Generate reads using a static normal distribution for the noise and duration sampler:
```
seq2squiggle predict example.fasta -o example.blow5 --noise-std 1.5 --noise-sampling False
```
Generate reads using only the noise sampler and a static normal distribution for the event length:
```
seq2squiggle predict example.fasta -o example.blow5 --duration-sampling False --ideal-event-length -1
```
Generate reads using only the noise sampler and ideal event lengths:
```
seq2squiggle predict example.fasta -o example.blow5 --duration-sampling False --ideal-event-length 10.0
```
Generate reads using a static normal distribution for the amplitude noise and ideal event lengths:
```
seq2squiggle predict example.fasta -o example.blow5 --duration-sampling False --ideal-event-length 10.0 --noise-sampling False --noise-std 1.0
```
Generate reads using no amplitude noise and ideal event lengths:
```
seq2squiggle predict example.fasta -o example.blow5 --duration-sampling False --ideal-event-length 10.0 --noise-sampling False --noise-std -1
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
seq2squiggle preprocess events.tsv train_dir --max-chunks -1 --config my_config.yml
seq2squiggle preprocess events_valid.tsv valid_dir --max-chunks -1 --config my_config.yml
seq2squiggle train train_dir valid_dir --config my_config.yml --model last.ckpt
```

## Acknowledgement
The model is based on [xcmyz's implementation of FastSpeech](https://github.com/xcmyz/FastSpeech). Some code snippets for preprocessing DNA-signal chunks have been taken from [bonito](https://github.com/nanoporetech/bonito). 
