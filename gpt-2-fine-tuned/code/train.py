import os
import time
import json
import argparse

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers.trainer_utils import set_seed

from athena.dataset_utils import ADDITIONAL_TOKENS
from athena.dataset_utils import get_dataloaders
from athena.trainer import CustomExplicitTrainer


def read_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def write_config(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  


def main(args):

    # create output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save current config file
    config_file_current = os.path.join(args.output_dir, 'config_train_exp.json')
    write_config(vars(args), config_file_current)

    # create empty log file
    log_file = os.path.join(args.output_dir, 'log.txt')
    with open(log_file, "w") as f:
        pass


    # set seed at the onset
    set_seed(args.seed)

    # init tokenizer
    time_tok_start = time.time()
    print('Init tokenizer ...', flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', 
                                              bos_token=ADDITIONAL_TOKENS['START'], 
                                              eos_token=ADDITIONAL_TOKENS['END'], 
                                              pad_token=ADDITIONAL_TOKENS['PAD']) 
    tokenizer.add_special_tokens({'additional_special_tokens': 
                                  [ADDITIONAL_TOKENS[t] for t in ADDITIONAL_TOKENS.keys() if t not in ['START', 'END', 'PAD']]})
    print(f'initialization of tokenizer is done, [time elapsed] = {time.time() - time_tok_start} .', flush=True)

    # get dataset
    time_ds_start = time.time()
    print('Get datasets ...', flush=True)
    train_dataloader, val_dataloader = get_dataloaders(args, tokenizer)
    print(f'initialization of dataset is done, [time elapsed] = {time.time() - time_ds_start} .', flush=True)

    # get model
    print('Init model & trainer ...', flush=True)
    time_model_start = time.time()
    configuration = GPT2Config.from_pretrained(args.model_name_or_path, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=configuration)

    # to accomodate all new tokens
    model.resize_token_embeddings(len(tokenizer))

    # load pretrained model (if available)
    if args.model_pretrained_path is not None and len(args.model_pretrained_path) > 0:
        model = model.from_pretrained(args.model_pretrained_path)

    # init Trainer
    trainer = CustomExplicitTrainer(
        model=model,
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        log_txt=log_file
    )
    print(f'initialization of model & trainer is done, [time elapsed] = {time.time() - time_model_start} .', flush=True)

    # training
    print('Start training ...', flush=True)
    time_train_start = time.time()
    trainer.train()
    print(f'training is done, [time elapsed] = {time.time() - time_train_start} .', flush=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parse first config file
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
    args, remaining_args = parser.parse_known_args()
    config_params = read_config(args.config)

    # parse the rest
    for k,v in config_params.items():
        parser.add_argument('--' + k, type=type(v), default=config_params.get(k))
    args = parser.parse_args()

    main(args)
