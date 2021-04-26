import argparse
import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers import EvaluationStrategy
from transformers import BertForMaskedLM, BertForPreTraining, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn.functional as F
from tqdm import tqdm


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


def calculate_mlm_for_dataset(
    model,
    ds: Dataset,
    batch_size: int = 16,
):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    result = []
    with torch.set_grad_enabled(False):
        for batch in tqdm(dl):
            x = [
                {
                    'input_ids': a,
                    'attention_mask': b,
                    'token_type_ids': c
                }
                for a, b, c in zip(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
            ]
            collated = data_collator(x)
            logits = model(
                input_ids=collated['input_ids'],
                attention_mask=collated['attention_mask'],
                token_type_ids=collated['token_type_ids'],
                labels=collated['labels']
            ).logits
            for logit, label in zip(logits, collated['labels']):
                loss = F.cross_entropy(logit, label)
                result.append(loss.item())
    return result


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertForMaskedLM.from_pretrained(args.model)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    print(dataset)
    print()

    train_mlm_losses = calculate_mlm_for_dataset(model, dataset['train'])
    test_mlm_losses = calculate_mlm_for_dataset(model, dataset['test'])

    dataset['train'] = dataset['train'].map(lambda ex, idx: {'mlm_loss': train_mlm_losses[idx]})
    dataset['test'] = dataset['test'].map(lambda ex, idx: {'mlm_loss': test_mlm_losses[idx]})

    dataset.save_to_disk(args.save)

    # for x in dataset['train']:
    #     calc_mlm_loss(x)
    #     break

    # with torch.set_grad_enabled(False):
    #     dataset = dataset.map(lambda x: {'mlm_loss': calc_mlm_loss(x)})
    #     dataset.save_to_disk(args.save)

