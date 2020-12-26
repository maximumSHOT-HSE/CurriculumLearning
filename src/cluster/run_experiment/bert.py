from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
import datasets
from transformers import EvaluationStrategy
from transformers import set_seed
import os

SEED = 42


def get_experiment_num():
    folder = 'Logs'
    if not os.path.exists(folder):
        return 1

    files = os.listdir('Logs')
    return max([int(file.split('BertLogs')[1]) for file in files]) + 1


def train(model, data_collator, dataset_train, dataset_eval, tokenizer):
    training_args = TrainingArguments(
        output_dir=f'./Logs/BertLogs{get_experiment_num()}',
        evaluation_strategy=EvaluationStrategy.STEPS,
        eval_steps=20000,
        seed=SEED,
        do_eval=True,
        do_train=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval
    )

    trainer.train()


# TODO dismember data collator
def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )


# TODO change to something more clever
def create_dataset(path_to_dump='/home/aomelchenko/datasets/wiki40b_en_encoded_cased'):
    return datasets.load_from_disk(path_to_dump)


# TODO maybe some experiments with tokenizer
def load_tokenizer():
    return BertTokenizer.from_pretrained("/home/aomelchenko/tokenizer_cased")


# TODO maybe some hyper parameters tuning
def create_model():
    return BertForMaskedLM(config=BertConfig.from_pretrained('../BertLargeConfig'))


def run():
    set_seed(SEED)

    model = create_model()
    tokenizer = load_tokenizer()

    dataset = create_dataset()
    dataset_train = dataset['train']
    dataset_eval = dataset['validation']

    dataset_train.set_format('torch')
    dataset_eval.set_format('torch')

    data_collator = create_data_collator(tokenizer)
    train(model=model, data_collator=data_collator, dataset_eval=dataset_eval,
          dataset_train=dataset_train, tokenizer=tokenizer)


if __name__ == '__main__':
    run()

