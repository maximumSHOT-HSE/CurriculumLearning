import argparse
import datasets
from datasets import Dataset
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers import EvaluationStrategy
from transformers import BertForMaskedLM, BertForPreTraining, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


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
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertForMaskedLM.from_pretrained(args.model)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])

    model.eval()

    DEVICE = 'cuda'  if torch.cuda.is_available() else 'cpu'
    model = model.to(DEVICE)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    print(dataset)
    print()

    def calc_mlm_loss(x):
        collated = data_collator([x])
        loss = model(
            input_ids=collated['input_ids'].to(DEVICE),
            attention_mask=collated['attention_mask'].to(DEVICE),
            token_type_ids=collated['token_type_ids'].to(DEVICE),
            labels=collated['labels'].to(DEVICE)
        ).loss.detach().cpu().item()
        return loss

    for x in dataset['train']:
        calc_mlm_loss(x)
        break

    with torch.set_grad_enabled(False):
        dataset = dataset.map(lambda x: {'mlm_loss': calc_mlm_loss(x)})
        dataset.save_to_disk(args.save)

