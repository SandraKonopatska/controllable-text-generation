import os
from typing import Dict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


ADDITIONAL_TOKENS = {
    'WORDS': '<COND_LENGTH>',
    'POS': '<COND_POS>',
    'SENTIMENT': '<COND_SENT>',
    'TENSE': '<COND_TENSE>',    
    'START': '<START>',
    'END': '<END>',
    'PAD': '<PAD>'
    }  
PAD_VALUES = {'input_ids': 1, 'attention_mask': 0}


def get_dataloaders(data_args: dataclass, tokenizer: PreTrainedTokenizerBase):

    # load datasets
    dataset = load_dataset("csv",
                            data_files={"train": os.path.join(data_args.data_path, 'train.csv'), 
                                        "val": os.path.join(data_args.data_path, 'val.csv')},
                            )
    # datasets (conditional)
    train_dataset = NumWordsDataset(dataset['train'], tokenizer, conditions=data_args.condition)
    valid_dataset = NumWordsDataset(dataset['val'], tokenizer, conditions=data_args.condition)

    # dataloaders
    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset, num_samples=data_args.num_samples), # Select batches randomly
                batch_size = data_args.per_device_train_batch_size, # Trains with this batch size.
                num_workers=data_args.num_workers
            )
    validation_dataloader = DataLoader(
                valid_dataset, # The validation samples.
                sampler = SequentialSampler(valid_dataset), # Pull out batches sequentially.
                batch_size = data_args.per_device_eval_batch_size, # Evaluate with this batch size.
                num_workers=data_args.num_workers
            )

    return train_dataloader, validation_dataloader


def create_prompt(conditions: Dict[str, str], train_txt: str = None) -> str:

      # check that we know every condition
      assert all([(cond_name.upper() in ADDITIONAL_TOKENS) for cond_name in conditions.keys()])
      
      # convert every value to str
      conditions = {cond_name:str(cond_value) for cond_name, cond_value in conditions.items()}

      prompt = ''
      for cond_name, cond_value in conditions.items():
         prompt += ADDITIONAL_TOKENS[cond_name.upper()] + ' ' + cond_value + ' '
      prompt += ADDITIONAL_TOKENS['START'] 

      if train_txt is not None:
           prompt += ' ' + train_txt + ' ' + ADDITIONAL_TOKENS['END']
           
      return prompt 


class NumWordsDataset(Dataset):

  def __init__(self, ds, tokenizer, conditions = 'words', max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for sample in ds:

      txt = sample['text']

      cond2prompt = {}
      for cond in conditions.split('|'):
         cond2prompt[cond] = sample[cond]

      prompt = create_prompt(cond2prompt, train_txt=txt)
      encodings_dict = tokenizer(prompt,
                                truncation=True,
                                max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]
