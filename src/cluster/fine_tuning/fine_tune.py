import argparse
import datasets
from transformers import BertTokenizer
from transformers import EvaluationStrategy
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--logging-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertForSequenceClassification.from_pretrained(args.model)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    print(dataset)
    print()

    for part in dataset:
        print()
        print(part)
        for x in dataset[part]:
            print(x)
            break

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=args.logging_dir,
        save_total_limit=5,
        seed=args.seed,
        learning_rate=3e-5,
        label_names=['label'],
        logging_first_step=True,
        evaluation_strategy=EvaluationStrategy.STEPS,
        eval_steps=500,
        logging_steps=500,
        save_steps=500,
    )

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    trainer.train()
    trainer.evaluate()

