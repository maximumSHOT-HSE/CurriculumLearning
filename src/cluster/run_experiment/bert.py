from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
import datasets
from transformers import EvaluationStrategy
from transformers import set_seed
import os
from curriculum_utils import CurriculumTrainerHyperbole, CurriculumTrainerDifficultyBiased
from argparse import ArgumentParser
import sys


SEED = 42


PATHS_TO_DATASET = {
    'base': '/home/aomelchenko/datasets/wiki40b_en_encoded_cased',
    'ee': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_by_column/wiki40b_encoded_cased_sorted_by_ee',
    'tse': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_by_column/wiki40b_encoded_cased_sorted_by_tse',
    'len': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_with_map/wiki40b_encoded_cased_sorted_by_len',
    'tse_div_len': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_with_map/wiki40b_encoded_cased_sorted_by_tse_div_len',
    'tse_difficult': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_by_column/wiki40b_encoded_cased_sorted_by_tse'
}

NUM_EPOCHS = {
    'base': 3,
    'ee': 1,
    'tse': 1,
    'len': 1,
    'tse_div_len': 1,
    'tse_difficult': 1
}


def get_experiment_num():
    folder = 'Logs'
    if not os.path.exists(folder):
        return 1

    files = os.listdir('Logs')
    return max([int(file.split('BertLogs')[1]) for file in files]) + 1


def get_trainer_class(experiment_type):
    if experiment_type == 'base':
        print('Base trainer chosen')
        return Trainer
    elif experiment_type == 'tse_difficult':
        print('Difficulty based trainer chosen')
        return CurriculumTrainerDifficultyBiased

    print('Hyperbole trainer chosen')
    return CurriculumTrainerHyperbole


def train(model, data_collator, dataset_train, dataset_eval, tokenizer, experiment_type):
    training_args = TrainingArguments(
        output_dir=f'/home/aomelchenko/Bachelor-s-Degree/Logs/{get_experiment_name(experiment_type)}',
        evaluation_strategy=EvaluationStrategy.STEPS,
        eval_steps=5000,
        save_steps=5000,
        num_train_epochs=NUM_EPOCHS[experiment_type],
        logging_steps=5000,
        seed=SEED,
        do_eval=True,
        do_train=True
    )

    trainer_class = get_trainer_class(experiment_type)

    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval
    )

    trainer.train()


def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )


def create_dataset(experiment_type):
    return datasets.load_from_disk(PATHS_TO_DATASET[experiment_type])


def load_tokenizer():
    return BertTokenizer.from_pretrained("/home/aomelchenko/tokenizer_cased")


def create_model():
    return BertForMaskedLM(config=BertConfig.from_pretrained('/home/aomelchenko/BertLargeConfig'))


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="base, ee or tse")

    parsed_args = parser.parse_args()

    return parsed_args.dataset


def get_experiment_name(experiment_type):
    return experiment_type + '_bert'


def run():
    set_seed(SEED)

    model = create_model()
    tokenizer = load_tokenizer()

    experiment_type = parse_argument()
    dataset = create_dataset(experiment_type)

    dataset_train = dataset['train']
    dataset_eval = dataset['validation']

    dataset_train.set_format('torch')
    dataset_eval.set_format('torch')

    data_collator = create_data_collator(tokenizer)
    train(model=model, data_collator=data_collator, dataset_eval=dataset_eval,
          dataset_train=dataset_train, tokenizer=tokenizer,
          experiment_type=experiment_type)


if __name__ == '__main__':
    run()

