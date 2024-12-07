import json

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np


class DOMEDataset(Dataset):
    def __init__(self, tokenizer, dataset, mode, max_token_inline=20, max_line_num=10, max_comment_len=15):
        # Initialize the dataset object with necessary attributes
        self.ids = []
        self.stat = []
        self.comment = []
        self.exemplar = []
        self.objectives = []
        self.bos_id = []
        self.max_token_instat = []

        # Store mode and tokenizer, and define the objective-to-id and objective-to-BOS ID mappings
        self.mode = mode
        self.tokenizer = tokenizer
        self.objective2id = {'what': 0, 'why': 1, 'usage': 2, 'done': 3, 'property': 4}
        self.objective2bos_id = {'what': "[WHAT/]", 'why': "[WHY/]", 'usage': "[USAGE/]", 'done': "[DONE/]",
                              'property': "[PROP/]"}
        self.objective2cls_id = {'what': "[/WHAT]", 'why': "[/WHY]", 'usage': "[/USAGE]", 'done': "[/DONE]",
                              'property': "[/PROP]"}

        # Open and read the dataset files
        with open(rf'./dataset/{dataset}/{mode}/code_split.{mode}', 'r') as f:
            code_stat_lines = f.readlines()
        with open(rf'./dataset/{dataset}/{mode}/comment.{mode}', 'r') as f:
            comment_lines = f.readlines()
        with open(rf'./dataset/{dataset}/{mode}/comment.similar_{mode}', 'r') as f:
            exemplar_lines = f.readlines()
        with open(rf'./dataset/{dataset}/{mode}/label.{mode}', 'r') as f:
            label_lines = f.readlines()

        # Iterate over the dataset and process each line
        count_id = 0
        for code_stat_line, comment_line, exemplar_line, label_line in tqdm(
                zip(code_stat_lines, comment_lines, exemplar_lines, label_lines)):
            
            # Parse the code statement
            statement_line = json.loads(code_stat_line.strip())
            if not statement_line['code']:
                continue
            count_id += 1
            self.ids.append(count_id)
             # Process the objective and map it to an ID
            objective = label_line.strip()
            self.objectives.append(self.objective2id[objective])

            # Process the code statements (split into lines, tokenize, and pad)
            temp_code = [[self.tokenizer.token_to_id(self.objective2cls_id[objective])]]
            temp_max_token = 0
            for stat_idx, stat in enumerate(statement_line['code'][:max_line_num]):
                temp_code.append(self.tokenizer.encode(stat).ids[:max_token_inline])
                temp_max_token = max(temp_max_token, len(temp_code[-1]))
            self.stat.append(temp_code)
            self.max_token_instat.append(temp_max_token)

            # Process the exemplar (example comments)
            exemplar_line = json.loads(exemplar_line.strip())[objective]
            self.exemplar.append(self.tokenizer.encode(exemplar_line).ids[:max_comment_len])
            
            # Process the comment (target comment for training/testing)
            if 'test' not in mode:
                self.comment.append(self.tokenizer.encode(comment_line.strip()).ids[:max_comment_len] +
                                    [self.tokenizer.token_to_id('[EOS]')])
            else:
                comment_token_list = comment_line.strip().split(' ')
                self.comment.append(comment_token_list)
            
            # Process the BOS token ID for the objective
            self.bos_id.append(self.tokenizer.token_to_id(self.objective2bos_id[objective]))

        # Ensure consistency between the lengths of all lists
        assert len(self.bos_id) == len(self.stat) == len(self.ids) == len(self.objectives)

    def __getitem__(self, index):
        '''
        Given an index, return the corresponding data sample, including:
        '''
        return self.stat[index], \
               len(self.stat[index]) - 1, \
               self.max_token_instat[index], \
               self.exemplar[index], \
               self.comment[index], \
               len(self.exemplar[index]), \
               len(self.comment[index]), \
               self.objectives[index], \
               self.bos_id[index], \
               self.ids[index]

    def __len__(self):
        '''
        Returns the total number of samples in the dataset.
        '''
        return len(self.ids)

    def padding_in_minibatch(self, code_list, stat_num_list, max_token_list):
        '''
        Given a list of code statements, apply padding so that all sequences in the batch have the same length.
        '''

        padded_code_list = []
        max_stat_num = max(stat_num_list)
        max_token_num = max(max_token_list)
        for code in code_list:
            # Pad the remaining code lines to match the max token length
            temp_code = code[0]
            for code_row in code[1:]:
                temp_code += code_row + [self.tokenizer.token_to_id('[PAD]')] * (max_token_num - len(code_row))
            # Pad the entire code to match the max number of statements
            temp_code += [self.tokenizer.token_to_id('[PAD]')] * max_token_num * (max_stat_num - len(code))
            padded_code_list.append(torch.tensor(temp_code))
        padded_seq = pad_sequence(padded_code_list, batch_first=True)
        stat_valid_num = torch.tensor(stat_num_list)
        return padded_seq, stat_valid_num

    def collate_fn(self, data):
        '''
        Collate function to combine a list of data samples into a batch.
        This function handles padding and ensures each input is ready for training.
        '''
        data_df = pd.DataFrame(data)
        return_list = []
        # Process the code (statements) in the batch
        stat_list, stat_num_list, stat_max_token_list = data_df[0].tolist(), data_df[1].tolist(), data_df[2].tolist()
        return_list += self.padding_in_minibatch(stat_list, stat_num_list, stat_max_token_list)

        # Process the other fields in the batch (exemplars, comments, objectives, etc.)
        for i in data_df:
            if i == 3:
                return_list.append(pad_sequence([torch.tensor(x, dtype=torch.int64) for x in data_df[i].tolist()], True))
            elif i == 4:
                if 'test' in self.mode:
                    return_list.append(data_df[i].tolist())
                else:
                    return_list.append(pad_sequence([torch.tensor(x) for x in data_df[i].tolist()], True))
            elif 4 < i < 9:
                return_list.append(torch.tensor(data_df[i].tolist()))
            elif i == 9:
                return_list.append(data_df[i].tolist())
        return return_list
