from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
import datasets
from transformers import EvaluationStrategy
from transformers import set_seed
import os
from curriculum_utils import CurriculumTrainerHyperbole, CurriculumTrainerDifficultyBiased, CurriculumTrainerCompetenceBased
from argparse import ArgumentParser
import sys


SEED = 42


PATHS_TO_DATASET = {
    'base': '/home/aomelchenko/datasets/wiki40b_en_3M_tokenized',
    'ee': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_by_column/wiki40b_en_3M_sorted_by_ee',
    'tse': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_by_column/wiki40b_en_3M_sorted_by_tse',
    'len': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_with_map/wiki40b_en_3M_tokenized_sorted_by_len',
    'tse_div_len': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_with_map/wiki40b_en_3M_sorted_by_tse_div_len',
    'tse_difficult': '/home/aomelchenko/Bachelor-s-Degree/src/cluster/sort_dataset_by_column/wiki40b_en_3M_sorted_by_tse'
}

NUM_EPOCHS = {
    'base': 1,
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


def get_trainer_class(experiment_type, curriculum_type):
    if experiment_type == 'base':
        print('Base trainer chosen')
        return Trainer
    elif experiment_type == 'tse_difficult':
        print('Difficulty based trainer chosen')
        if curriculum_type == 'hyperbole':
            return CurriculumTrainerDifficultyBiased
        else:
            return CurriculumTrainerCompetenceBased

    print('Hyperbole trainer chosen')
    return CurriculumTrainerHyperbole


def train(model, data_collator, dataset_train, dataset_eval, tokenizer, experiment_type, curriculum_type):
    training_args = TrainingArguments(
        output_dir=f'/home/aomelchenko/Bachelor-s-Degree/Logs/{curriculum_type}_{get_experiment_name(experiment_type)}',
        evaluation_strategy=EvaluationStrategy.STEPS,
        eval_steps=500,
        save_steps=500,
        num_train_epochs=NUM_EPOCHS[experiment_type],
        logging_steps=500,
        seed=SEED,
        per_device_eval_batch_size=128,
        per_device_train_batch_size=128,
        logging_first_step=True
    )

    print("train")
    trainer_class = get_trainer_class(experiment_type, curriculum_type)

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
    return BertTokenizer.from_pretrained("/home/aomelchenko/tokenizers/BertTokenizerBase")


def create_model():
    return BertForMaskedLM(config=BertConfig.from_pretrained('/home/aomelchenko/BertBaseConfigReduced'))


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="base, ee or tse")
    parser.add_argument("--curriculum", type=str, help="base, ee or tse")

    parsed_args = parser.parse_args()

    return parsed_args.dataset, parsed_args.curriculum


def get_experiment_name(experiment_type):
    return experiment_type + '_bert'


def run():
    set_seed(SEED)

    model = create_model()
    tokenizer = load_tokenizer()
    model.resize_token_embeddings(len(tokenizer))

    experiment_type, curriculum_type = parse_argument()
    dataset = create_dataset(experiment_type)

    dataset_train = dataset['train']
    dataset_eval = dataset['validation']

    dataset_train.set_format('torch', columns=['input_ids', 'attention_mask'])
    dataset_eval.set_format('torch', columns=['input_ids', 'attention_mask'])

    data_collator = create_data_collator(tokenizer)
    train(model=model, data_collator=data_collator, dataset_eval=dataset_eval,
          dataset_train=dataset_train, tokenizer=tokenizer,
          experiment_type=experiment_type, curriculum_type=curriculum_type)


if __name__ == '__main__':
    run()

