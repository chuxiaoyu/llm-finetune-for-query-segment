Model Selections and Performance Summary:
|ID|Model|Size|Train Time [min]|Precision|Recall|F1|
|-|-|-|-|-|-|-|
|1|[distilbert-base-multilingual-cased](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)|100M params|23.63|0.8252|0.8385|0.8318|
|2|[distilbert-NER](https://huggingface.co/dslim/distilbert-NER)|66M params|25.46|0.8024|0.8339|0.8179|

Hyper Parameters and Enviroment Settings:
|Epoch|LearningRate|BatchSize|Hardware|Python|PyTorch|
|-|-|-|-|-|-|
|7|2e-5|16|Apple Silicon (arm64)| x86_64 via Anaconda | 2.6.0, CPU-only|