# **seq2squiggle Configuration File**

This guide explains all the available parameters in the `config.yaml` file used to configure training in seq2squiggle. The parameters are grouped into different sections based on their purpose, such as logging, preprocessing, model architecture, and training.

---

## **Logging Parameters**

- **`log_name`**  
  The name of the logging directory where logs will be saved.  
  *Example:* `"Human-R1041-4khz"`

- **`wandb_logger_state`**  
  The state of the wandb logger, which is used for experiment tracking and visualization.  
  Options:  
  - `disabled`: Completely disable wandb logging.  
  - `online`: Enable online logging, which requires logging into wandb via `wandb login` or using other authentication methods (API key, environment variable, etc.). More details can be found on the [wandb website](https://wandb.ai/site).  
  - `offline`: Store logs locally without syncing to the wandb cloud. Use this if you prefer not to log in or need offline access to logs.  

---

## **Preprocessing Parameters**

A **chunk** consists of a DNA sequence of length `max_dna_len` and its corresponding mapped signal, determined by the segmentation tool (e.g., f5c or uncalled4). The chunk size is defined by `max_dna_len` (DNA sequence length) and `max_signal_len` (maximum signal length), both of which are set in the config file.  

💡 **Important:** The values of `max_dna_len` and `max_signal_len` **must be the same** for both preprocessing and training. Using different values will cause mismatches and likely lead to errors.

- **`max_chunks_train`**  
  The maximum number of chunks for training data.  
  *Default:* `210000000`  
  This limit applies during both preprocessing and training. If your dataset contains fewer chunks, only that number will be used.

- **`max_chunks_valid`**  
  The maximum number of chunks for validation data.  
  *Default:* `100000`  

- **`scaling_max_value`**  
  The value for scaling input signals.  
  *Default:* `165.0`  
  This is used to normalize pA (picoampere) values before feeding them into the model.  

- **`train_valid_split`**  
  The fraction of data used for training. The remaining portion is used for validation.  
  *Default:* `0.9`  
  If you don't provide a separate validation dataset during preprocessing, seq2squiggle will automatically split your data based on this ratio.

- **`max_dna_len`**  
  The maximum length of the DNA sequence in each chunk.  
  *Default:* `16`  

- **`max_signal_len`**  
  The maximum signal length for each DNA sequence.  
  *Default:* `250`  
  - If the corresponding signal is shorter, it will be **zero-padded** to reach this length.  
  - If it's longer, it will be **excluded** from training after preprocessing.

- **`allowed_chars`**  
  The allowed characters in the DNA sequence.  
  *Default:* `"_ACGT"`  
  - The `_` represents an empty symbol used when splitting input sequences into chunks of `max_dna_len`.  
  - Example: `"ACGTACGT__"` (when padding is needed).

- **`seq_kmer`**  
  The k-mer size used in the model.  
  *Default:* `9`  
  - For **R10.4.1 data**, set this to **9**.  
  - For **R9.4.1 data**, set this to **6**.  
  Using the wrong value for your dataset may cause unexpected behavior.

---

## **Model Parameters**

seq2squiggle uses a **Feed-Forward Transformer** (based on FastSpeech) to predict nanopore signals from DNA data.

- **`pre_layers`**  
  The number of **pre-processing layers** before the main transformer model.  
  *Default:* `1`  
  - Input DNA k-mers are one-hot encoded and passed through a **dense layer**.  
  - Then, `pre_layers` specifies the number of additional dense layers before the feed-forward transformer block.

- **`dmodel`**  
  The dimension of the model (hidden layer size).  
  *Default:* `64`  

- **`dff`**  
  The size of the feed-forward layer.  
  *Default:* `256`  

- **`encoder_layers`**  
  The number of layers in the **encoder**.  
  *Default:* `2`  

- **`encoder_heads`**  
  The number of **attention heads** in the encoder.  
  *Default:* `8`  

- **`decoder_layers`**  
  The number of layers in the **decoder**.  
  *Default:* `2`  

- **`decoder_heads`**  
  The number of **attention heads** in the decoder.  
  *Default:* `8`  

- **`encoder_dropout`**  
  The dropout rate for the encoder layers.  
  *Default:* `0.2`  

- **`decoder_dropout`**  
  The dropout rate for the decoder layers.  
  *Default:* `0.2`  

- **`duration_dropout`**  
  The dropout rate for **duration sampling** (used in signal prediction).  
  *Default:* `0.2`  

---

## **Training Parameters**

- **`train_batch_size`**  
  The batch size for training.  
  *Default:* `512`  

- **`max_epochs`**  
  The maximum number of epochs for training.  
  *Default:* `25`  

- **`save_model`**  
  Whether to save the trained model.  
  *Options:* `True` or `False`  

- **`optimizer`**  
  The optimizer used for training.  
  *Options:*  
  - `Adam`  
  - `AdamW`  
  - `RAdam`  
  - `AdaFactor`  
  - `RMSProp`  
  *Default:* `"Adam"`  

- **`warmup_ratio`**  
  The fraction of training steps used for **learning rate warmup**.  
  *Default:* `0.01`  

- **`lr`**  
  The learning rate.  
  *Default:* `0.0005`  

- **`weight_decay`**  
  The weight decay for regularization.  
  *Default:* `0.0`  

- **`lr_schedule`**  
  The learning rate schedule. Options:  
  - `warmup_cosine`: Cosine annealing with warmup.  
  - `warmup_constant`: Constant learning rate with warmup.  
  - `constant`: Fixed learning rate.  
  - `warmup_cosine_restarts`: Cosine annealing with periodic restarts.  
  - `one_cycle`: One-cycle learning rate schedule.  

- **`gradient_clip_val`**  
  The **gradient clipping** value to prevent exploding gradients.  
  *Default:* `1.0`  
  - This is necessary for stabilizing training, especially for the **Feed-Forward Transformer** model.  

---

## **Final Notes**
- Always ensure your **preprocessing settings match your training settings** (especially `max_dna_len`, `max_signal_len`, and `seq_kmer`)
- If you encounter unexpected training behavior, check whether **your optimizer and learning rate schedule** are suitable for your dataset
- For logging and experiment tracking, **wandb integration is recommended** (but can be disabled if not needed)
- If you are working with different **nanopore sequencing chemistries**, adjust `seq_kmer` accordingly
