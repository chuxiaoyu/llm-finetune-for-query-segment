# Fine-tune LLM for query segmentation

## Code
```
query-segment/
├── hf_cache/         # [REMOVED] Cached tokenizer and models from HuggingFace
├── models/           # [REMOVED] The fine-tuned models and log history
├── query_segment.py  # Finetune code
├── evaluate.ipynb    # [REMOVED] Evaluation results
├── README.md         # Methods, findings, improvements
└── requirements.txt  # Required packages
```

## Setup
```
conda create -n qner
conda activate qner
pip install -r requirements.txt

python query_segment.py  # for fine-tuning
```

## Methods
The raw dataset was transformed into HuggingFace NER format (token sequence + BIO labels). A key step is the `tokenize_and_align_labels` function, because transformer tokenizers operate at the subword level. A single word-level token in the dataset may be split into multiple subword tokens by the model tokenizer. For example, a token like "Station" may be tokenized into ["Sta", "##tion"]. Since labels are provided at the original token level, we align them by assigning the label to the first subword and ignoring the remaining subwords during training (using label `-100`).

Model Selections and Performance Summary:
|ID|Model|Size|Train Time [min]|Precision|Recall|F1|
|-|-|-|-|-|-|-|
|1|[distilbert-base-multilingual-cased](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)|100M params|23.63|0.8252|0.8385|0.8318|
|2|[distilbert-NER](https://huggingface.co/dslim/distilbert-NER)|66M params|25.46|0.8024|0.8339|0.8179|

Hyper Parameters and Enviroment Settings:
|Epoch|LearningRate|BatchSize|Hardware|Python|PyTorch|
|-|-|-|-|-|-|
|7|2e-5|16|Apple Silicon (arm64)| x86_64 via Anaconda | 2.6.0, CPU-only|


## Future Improvements

1. Training acceleration. Use GPU or PyTorch MPS to speed up model training and fine-tuning, reducing overall computation time.

2. Data imbalance. Address the unequal distribution of labels, which can negatively affect model performance on underrepresented classes. We can use techniques such as class weighting, oversampling, or data augmentation.

3. Overfitting. During training, the test loss plateaus after several epochs, indicating that the model has reached its optimal performance. To prevent overfitting and save computation, we can use techniques such as early stopping.

4. Hyper parameters optimization. We can systematically search for the best hyperparameter combanitions (e.g., learning rate, batch size, number of epochs) using methods such as grid search.
