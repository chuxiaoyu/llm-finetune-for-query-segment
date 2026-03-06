import numpy as np
import pandas as pd
import ast
import os
import time
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, TrainerCallback
from sklearn.metrics import confusion_matrix
import evaluate

# MODEL CONFIG: SELECT THE MODEL TO TRAIN/FINETUNE
MODEL_NAME = "distilbert-base-multilingual-cased"
# MODEL_NAME = "dslim/distilbert-NER"
# MODEL_NAME = "bert-base-multilingual-cased"

OUTPUT_DIR = f"./models/{MODEL_NAME.replace('/', '_')}"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_csv_dataset(path):
    df = pd.read_csv(path, index_col=0)

    # Convert string -> Python list
    def parse_seg(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        x = x.strip()
        return ast.literal_eval(x)

    df["query_segmentation"] = df["query_segmentation"].apply(parse_seg)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['tags'] = df['tags'].apply(ast.literal_eval)

    return df


def df_to_samples(df):
    examples = []
    for _, row in df.iterrows():
        examples.append({
            "searched_query": row["searched_query"],
            "tokens": row["tokens"],
            "labels": row["tags"]
        })
    return examples


def to_hf_dataset(samples):
    return Dataset.from_list(samples)


def build_label_set(samples):
    label_set = set()
    for ex in samples:
        label_set.update(ex["labels"])
    label_list = sorted(label_set)
    label_count = {l:0 for l in label_list}
    for ex in samples:
        for l in ex["labels"]:
            label_count[l] += 1
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}
    return label_list, label_count, label2id, id2label


def tokenize_and_align_labels(batch, label2id):
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True
    )

    aligned_labels = []
    for i in range(len(batch["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        ex_labels = batch["labels"][i]

        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[ex_labels[word_id]])
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def make_splits(dataset, test_size=0.2, seed=42):
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({
        "train": split["train"],
        "test": split["test"]
    })


def compute_metrics(p, id2label):
    seqeval = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = []
    true_labels = []

    for pred, lab in zip(predictions, labels):
        cur_preds = []
        cur_labels = []
        for p_i, l_i in zip(pred, lab):
            if l_i == -100:
                continue
            cur_preds.append(id2label[p_i])
            cur_labels.append(id2label[l_i])
        true_preds.append(cur_preds)
        true_labels.append(cur_labels)

    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class PerEpochTrainTimeCallback(TrainerCallback):
    """Log train runtime (seconds) for each epoch."""
    def __init__(self):
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.perf_counter()

    def on_evaluate(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None and state.log_history:
            elapsed = time.perf_counter() - self.epoch_start_time
            # last entry is the eval one just completed; epoch and eval_runtime are in there
            last = state.log_history[-1]
            epoch = last.get("epoch")
            eval_runtime = last.get("eval_runtime")
            train_runtime = elapsed
            if eval_runtime is not None:
                train_runtime = max(elapsed - eval_runtime, 0.0)
            if epoch is not None:
                # Print and also store into log_history so it ends up in trainer_state.json
                if eval_runtime is not None:
                    print(
                        f"  [Epoch {epoch:.1f}] train runtime: {train_runtime:.1f}s "
                        f"(eval: {eval_runtime:.1f}s)"
                    )
                else:
                    print(f"  [Epoch {epoch:.1f}] train runtime: {train_runtime:.1f}s")
                state.log_history.append({"epoch": epoch, "train_runtime": train_runtime})
        self.epoch_start_time = None


def train_ner_model(dataset_dict, label_list, label2id, id2label):
    # ignore_mismatched_sizes=True: load pretrained backbone, reinit classifier head for our num_labels
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    tokenized = dataset_dict.map(
        lambda x: tokenize_and_align_labels(x, label2id),
        batched=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[PerEpochTrainTimeCallback()],
    )

    trainer.train()
    return trainer


def evaluate_model(trainer):
    metrics = trainer.evaluate()
    print(metrics)
    return metrics


def show_confusion_matrix(trainer, save_dir=None):
    """Print confusion matrix (rows=true, cols=pred) on the eval (test) dataset. Optionally save CSV under save_dir."""
    pred_out = trainer.predict(trainer.eval_dataset)
    pred_ids = np.argmax(pred_out.predictions, axis=2)
    label_ids = pred_out.label_ids

    y_true = []
    y_pred = []
    id2label = trainer.model.config.id2label
    for pred_row, label_row in zip(pred_ids, label_ids):
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            y_true.append(id2label.get(int(l), str(l)))
            y_pred.append(id2label.get(int(p), str(p)))

    n = len(id2label)
    labels = [id2label[i] if i in id2label else id2label[str(i)] for i in range(n)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm_df)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "confusion_matrix.csv")
        cm_df.to_csv(path)
        print(f"Saved to {path}")
    return cm_df


def main():
    path = "./tomtom/segmentation_sample.csv"
    df = load_csv_dataset(path)
    
    samples = df_to_samples(df)
    dataset = to_hf_dataset(samples)
    dataset_dict = make_splits(dataset)

    label_list, label_count, label2id, id2label = build_label_set(samples)

    trainer = train_ner_model(dataset_dict, label_list, label2id, id2label)
    evaluate_model(trainer)
    show_confusion_matrix(trainer, save_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()