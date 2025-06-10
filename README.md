# seq2squiggle

`seq2squiggle` is a deep learning-based tool for generating artifical nanopore signals from DNA sequence data.

<img src="/img/seq2squiggle.png" width="750">


Please cite the following publication if you use `seq2squiggle` in your work:
- Beslic D, Kucklick M, Engelmann S, Fuchs S, Renard BY, Körber N. End-to-end simulation of nanopore sequencing signals with feed-forward transformers. *Bioinformatics*. 2024; btae744. [doi:10.1093/bioinformatics/btae744](https://doi.org/10.1093/bioinformatics/btae744) 

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

### Install with conda or mamba
```
mamba install seq2squiggle -c bioconda
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

## Predict signals from FASTA/Q file
`seq2squiggle` simulates artificial signals based on an input FASTA or FASTQ file. It supports two main simulation modes.

### Reference mode (default)

**Input**: FASTA file (e.g. genome or transcriptome reference)

**Goal**: Generate synthetic reads _and_ simulate corresponding signals.

#### How it works

- Synthetic reads are sampled from the reference sequence.
- Each read is used to simulate a raw signal.
- Use `-n`, `-c`, or `-r` to control number, coverage, or average read length.
- Use `--distr` to specify the read length distribution.

Example:

```bash
# Generate 10,000 reads & signals from a fasta reference file
seq2squiggle predict example.fasta -o example.blow5 -n 10000

# Generate 10,000 reads & signals using R9.4.1 chemistry
seq2squiggle predict example.fasta -o example.blow5 -n 10000 --profile dna_r9_min

# Export as pod5
seq2squiggle predict example.fastq -o example.pod5 -n 10000

# Generate reads with a coverage of 30 and an average read length of 5,000:
seq2squiggle predict example.fasta -o example.blow5 -c 30 -r 5000
```

### Read mode (--read-input)

**Input**: FASTA or FASTQ file containing pre-existing reads (e.g. basecalled reads)

**Goal**: Simulates signals directly from input reads - no additional fragmentation

There are two behaviors within Read Mode:
#### One signal per input read
```bash
seq2squiggle predict example.fastq -o example.blow5 --read-input
```
This generates exactly one signal per input read (1:1 mapping).

#### Multiple signals based on input reads (-n)
```bash
seq2squiggle predict example.fastq -o example.blow5 --read-input -n 1000
```
This samples n total signals by selecting input reads without splitting them.
It's not simulating n signals per read, but rather simulating n signals across the input reads.

This hybrid approach is useful when:
- You want to simulate larger datasets from real reads without generating synthetic sequences.
- You want to preserve real read characteristics while exploring noise or signal variability.

### Notes

#### Output format

By default, the output is in SLOW5/BLOW5 format. Exporting to the new POD5 format is also supported, though BLOW5 is preferred for its stability. You will need to specify the path to the model through the configuration file.


#### Performance tips

For optimal performance, running `seq2squiggle` on a GPU is recommended, especially to speed up inference for a high number of samples. However, the tool also works on CPU-only systems, though at a slower inference speed.


## Noise options
Signal generation can be customized with:
- **Noise Sampler**: predicts per-k-mer signal amplitude variation
- **Duration Sampler**: predicts per-k-mer dwell time (event length) variation

You can disable either or both to use static distributions instead.

### Examples using different noise options

```bash
# Default configuration (noise sampler and duration sampler enabled)
seq2squiggle predict example.fasta -o example.blow5

# Using the noise sampler with increased noise standard deviation and the duration sampler
seq2squiggle predict example.fasta -o example.blow5 --noise-std 1.5

# Using a static normal distribution for the amplitude noise and the duration sampler
seq2squiggle predict example.fasta -o example.blow5 --noise-std 1.0 --noise-sampling False

# Using the noise sampler and a static normal distribution for event durations
seq2squiggle predict example.fasta -o example.blow5 --duration-sampling False --dwell-std 4.0

# Using the noise sampler with ideal event lengths (each k-mer event will have a length of 10):
seq2squiggle predict example.fasta -o example.blow5 --duration-sampling False --dwell-mean 10.0 --dwell-std 0.0

# Using a static normal distribution for amplitude noise and ideal event lengths:
seq2squiggle predict example.fasta -o example.blow5 --duration-sampling False --dwell-mean 10.0 --dwell-std 0.0 --noise-sampling False --noise-std 1.0

# Generating reads with no amplitude noise and ideal event lengths:
seq2squiggle predict example.fasta -o example.blow5 --duration-sampling False --dwell-mean 10.0 --dwell-std 0.0 --noise-sampling False --noise-std 0.0 
```

## Train a new model

`seq2squiggle` can be trained using segmented sequence and signal data obtained from a signal-to-reference alignment. These alignments can be generated using [uncalled4](https://github.com/skovaka/uncalled4) or [f5c](https://github.com/hasindu2008/f5c), which produce an eventalign table mapping raw signals to their corresponding reference sequences.

### 1. Generate training data with `uncalled4` or `f5c`  
You need event-level signal alignments as training data. These can be generated using either `uncalled4` or `f5c`.  

#### Using [uncalled4](https://github.com/skovaka/uncalled4) 
Run the following command to align reads and extract event-level signals:  

```bash
uncalled4 align input.ref input.slow5 \
    --bam-in input.bam \
    --eventalign-out events.tsv \
    --eventalign-flags print-read-names,signal-index,samples \
    --pore-model dna_r10.4.1_400bps_9mer \
    --flowcell FLO-MIN114 \
    --kit SQK-LSK114  
```

#### Using [f5c](https://github.com/hasindu2008/f5c)
Alternatively, you can use f5c to perform the alignment:

```bash
f5c eventalign \
    -r input.fastq \
    -b input.bam \
    -g input.ref \
    --slow5 input.slow5 \
    -o events.tsv \
    --samples \
    --signal-index \
    --print-read-names \
    --collapse-events \
    --pore r10
```

### 2. Filter and transform into pA values (optional)

Since `uncalled4` version 4.1.0, it no longer outputs scaled pA values directly. Instead, it outputs the normalized pore model values, which have a mean of 0 and a standard deviation of 1. `seq2squiggle` requires the signals to be in pA, so they must be converted back. 

To denormalize signals, you first need to compute the average mean and average standard deviation of the pA values from your data. This can be done using [sigtk](https://github.com/hasindu2008/sigtk):

```bash
sigtk pa -n input.slow5 | \
    cut -f3 | \
    sed 's/,/\n/g' | \  
    datamash mean 1 sstdev 1 | \
    awk '{print "Mean pa_mean:", $1, "Mean pa_std:", $2}' > sigtk.txt
```

The sigtk.txt will look like this
```
Mean pa_mean: 80.646167196571
Mean pa_std: 17.826207455846
```

Once you have the sigtk.txt file, you can run the `standardize-events.py` script to denormalize the signal values:
```bash
./src/seq2squiggle/standardize-events.py events.tsv events_norm.tsv --sigtk sigtk.txt
```

### 3. Preprocess the event table to .npy chunks
Before training, the eventalign table needs to be preprocessed into .npy chunks. It is recommended to first split the data into training and validation sets. This can be done based on chromosome regions (e.g., using chromosome 1 for training and chromosome 21 for validation) or by applying a random split with a specified fraction.

Once the data is split, preprocess each subset:
```bash
seq2squiggle preprocess events_norm_train.tsv train_dir --max-chunks -1 --config my_config.yml
seq2squiggle preprocess events_norm_valid.tsv valid_dir --max-chunks -1 --config my_config.yml
```


### 4. Train
Once preprocessing is complete, you can start training:
```bash
seq2squiggle train train_dir valid_dir --config my_config.yml --model last.ckpt
```

For a detailed explanation of the configuration parameters, see [CONFIG.md](CONFIG.md)

## Acknowledgement
The model is based on [xcmyz's implementation of FastSpeech](https://github.com/xcmyz/FastSpeech). Some code snippets for preprocessing DNA-signal chunks have been taken from [bonito](https://github.com/nanoporetech/bonito). We also incorporated code snippets from [Casanovo](https://github.com/Noble-Lab/casanovo) for different functionalities, including downloading weights, logging, and the design of the main function. 
Additionally, we used parameter profiles from squigulator for various chemistries to set digitisation, sample-rate, range, median_before, and other signal parameters. These profiles are detailed in [squigulator's documentation](https://hasindu2008.github.io/squigulator/docs/profile.html).
