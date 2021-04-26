import argparse
import datasets
from transformers import BertTokenizer
from transformers import EvaluationStrategy
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertForSequenceClassification.from_pretrained(args.model)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    print(dataset)
    print()

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        eval_dataset=dataset['test'],
    )

    pred = trainer.predict(test_dataset=dataset['test'])
    df = pd.DataFrame.from_dict({
        'predictions': pred.predictions,
        'label_ids': pred.label_ids,
        args.column: dataset['test'][args.column]
    })
    df.to_csv(args.save, index=False)
