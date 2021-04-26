import argparse
import datasets
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers import EvaluationStrategy
from transformers import BertForMaskedLM, BertForPreTraining, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from custom_trainers import (
    SequentialTrainer,
    CurriculumTrainerHyperbole,
    CurriculumTrainerDifficultyBiased,
    CurriculumTrainerCompetenceBased,
    ReverseSequentialTrainer,
    FromFileTrainer
)


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


TRAINERS = {
    'default': Trainer,
    'sequential': SequentialTrainer,
    'hyperbole': CurriculumTrainerHyperbole,
    'difficulty-based': CurriculumTrainerDifficultyBiased,
    'competence-based': CurriculumTrainerCompetenceBased,
    'reverse-sequential': ReverseSequentialTrainer,
    'from-file': FromFileTrainer
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--logging-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--trainer', type=str, default='default', choices=list(TRAINERS.keys()))
    parser.add_argument('--from-file', type=str, default=None)
    parser.add_argument('--mlm-prob', type=float, default=0.15)
    parser.add_argument('--reverse', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = datasets.load_from_disk(args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertForMaskedLM.from_pretrained(args.model)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask'])

    print(dataset)
    print()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir=args.logging_dir,
        save_total_limit=1,
        seed=args.seed,
        logging_first_step=True,
        evaluation_strategy=EvaluationStrategy.STEPS,
        eval_steps=500,
        logging_steps=500,
        save_steps=500,
    )

    trainer = TRAINERS[args.trainer](
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
    )

    if args.trainer != 'default':
        trainer.reverse = bool(int(args.reverse) != 0)

    if args.trainer == 'from-file':
        trainer.file = args.from_file

    trainer.train()
    print(trainer.evaluate())

