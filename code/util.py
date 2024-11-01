import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support, \
    classification_report, auc
import math
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import logging
import os
import re

logger = logging.getLogger(__name__)


def preprocess_code_line(code, remove_python_common_tokens=False):
    # code = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ') \
    #     .replace('[', ' ').replace(']', ' ').replace('.', ' ').replace(':', ' ') \
    #     .replace(';', ' ').replace(',', ' ').replace(' _ ', '_')

    code = re.sub('``.*``', '<STR>', code)
    code = re.sub("'.*'", '<STR>', code)
    code = re.sub('".*"', '<STR>', code)
    code = re.sub('\d+', '<NUM>', code)

    code = code.split()
    code = ' '.join(code)
    if remove_python_common_tokens:
        new_code = ''
        python_common_tokens = []
        for tok in code.split():
            if tok not in [python_common_tokens]:
                new_code = new_code + tok + ' '

        return new_code.strip()

    else:
        return code.strip()


def remove_punctuation(line):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('', line)
    return line


def add_code_line_identify_token(code_lines, code_line_labels, type_code=None):
    temp_lines = []
    temp_labels = []
    for idx, line in enumerate(code_lines):
        if len(line):
            if type_code == 'added':
                temp_lines.append('[ADD] ' + line)
            elif type_code == 'deleted':
                temp_lines.append('[DEL] ' + line)

            temp_labels.append(code_line_labels[idx])

    return temp_lines, temp_labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, commit_id, commit_idx, commit_ids, commit_tokens,
                 input_ids, input_mask, input_tokens, manual_features, label,
                 line_ids, line_tokens, line_label):
        self.commit_id = commit_id
        self.commit_idx = commit_idx
        self.commit_ids = commit_ids
        self.commit_tokens = commit_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_tokens = input_tokens
        self.manual_features = manual_features
        self.label = label
        self.line_ids = line_ids
        self.line_tokens = line_tokens
        self.line_label = line_label


class InputMixFeatures(object):
    """A single set of features of data."""

    def __init__(self, commit_id, commit_idx, input_type, input_ids, input_tokens, label):
        self.commit_id = commit_id
        self.commit_idx = commit_idx
        self.input_type = input_type  # 0 for commit, 1 for code line
        self.input_ids = input_ids
        self.input_tokens = input_tokens
        self.label = label


class InputCommitFeatures(object):
    """A single set of features of data."""

    def __init__(self, commit_id, commit_idx, commit_ids, commit_tokens, commit_label):
        self.commit_id = commit_id
        self.commit_idx = commit_idx
        self.commit_ids = commit_ids
        self.commit_tokens = commit_tokens
        self.commit_label = commit_label


class InputLineFeatures(object):
    """A single set of features of data."""

    def __init__(self, commit_id, line_idx, line_ids, line_tokens, line_label):
        self.commit_id = commit_id
        self.line_idx = line_idx
        self.line_ids = line_ids
        self.line_tokens = line_tokens
        self.line_label = line_label


def convert_examples_to_features(item, mask_padding_with_zero=True, buggy_commit_lines_df=None, commit_idx=0):
    # source
    commit_id, files, msg, label, tokenizer, args, manual_features = item

    # prepare code line data
    old_add_code_lines = list(files['added_code'])  # added code lines
    old_delete_code_lines = list(files['removed_code'])  # deleted code lines

    if commit_id in buggy_commit_lines_df['commit hash'].to_list():
        commit_info_df = buggy_commit_lines_df[buggy_commit_lines_df['commit hash'] == commit_id]
        commit_info_df = commit_info_df.reset_index(drop=True)

        commit_info_df['code line'] = commit_info_df['code line'].apply(lambda x: remove_punctuation(x))

        add_code_lines_dict = {remove_punctuation(line): line for line in old_add_code_lines}
        delete_code_lines_dict = {remove_punctuation(line): line for line in old_delete_code_lines}

        commit_info_df_added = commit_info_df[commit_info_df['change type'] == 'added']
        commit_info_df_added = commit_info_df_added.reset_index(drop=True)
        add_code_lines = []
        add_code_line_labels = []
        for i in range(len(commit_info_df_added)):
            if commit_info_df_added['code line'][i] in add_code_lines_dict.keys():
                add_code_lines.append(add_code_lines_dict[commit_info_df_added['code line'][i]])
                add_code_line_labels.append(commit_info_df_added['label'][i])

            else:
                print('added lines matching exists wrong')

        commit_info_df_deleted = commit_info_df[commit_info_df['change type'] == 'deleted']
        commit_info_df_deleted = commit_info_df_deleted.reset_index(drop=True)
        delete_code_lines = []
        delete_code_line_labels = []
        for i in range(len(commit_info_df_deleted)):
            if commit_info_df_deleted['code line'][i] in delete_code_lines_dict.keys():
                delete_code_lines.append(delete_code_lines_dict[commit_info_df_deleted['code line'][i]])
                delete_code_line_labels.append(commit_info_df_deleted['label'][i])
            else:
                print('deleted lines matching exists wrong')
    else:
        add_code_lines = old_add_code_lines
        delete_code_lines = old_delete_code_lines
        add_code_line_labels = [0] * len(add_code_lines)
        delete_code_line_labels = [0] * len(delete_code_lines)

    assert len(add_code_lines) == len(add_code_line_labels)
    assert len(delete_code_lines) == len(delete_code_line_labels)

    add_code_lines = [preprocess_code_line(line, False) for line in add_code_lines]
    delete_code_lines = [preprocess_code_line(line, False) for line in delete_code_lines]

    add_code_lines, add_code_line_labels = add_code_line_identify_token(add_code_lines,
                                                                        add_code_line_labels,
                                                                        type_code='added')
    delete_code_lines, delete_code_line_labels = add_code_line_identify_token(delete_code_lines,
                                                                              delete_code_line_labels,
                                                                              type_code='deleted')

    assert len(add_code_lines) == len(add_code_line_labels)
    assert len(delete_code_lines) == len(delete_code_line_labels)

    total_code_lines = add_code_lines + delete_code_lines
    total_code_line_labels = add_code_line_labels + delete_code_line_labels
    total_code_line_tokens = [tokenizer.tokenize(line) for line in total_code_lines]

    assert len(total_code_lines) == len(total_code_line_labels)

    # prepare commit data
    added_tokens = []
    removed_tokens = []
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]
    files['added_code'] = add_code_lines[:]
    files['removed_code'] = delete_code_lines[:]
    file_codes = files
    added_codes = file_codes['added_code']
    codes = ''.join([line for line in added_codes])
    added_tokens.extend(tokenizer.tokenize(codes))
    removed_codes = file_codes['removed_code']
    codes = ''.join([line for line in removed_codes])
    removed_tokens.extend(tokenizer.tokenize(codes))
    commit_tokens = msg_tokens + added_tokens + removed_tokens
    commit_ids = tokenizer.convert_tokens_to_ids(commit_tokens)
    # construct "commit + code line" item
    ret_list = []
    for code_line_tokens, code_line, code_line_label in \
            zip(total_code_line_tokens, total_code_lines, total_code_line_labels):

        if len(code_line_tokens) >= args.max_codeline_token_length:
            code_line_tokens = (code_line_tokens[:args.max_codeline_token_length])

        input_tokens = commit_tokens[:args.max_input_token_length - 3 - len(code_line_tokens)]
        input_tokens = [tokenizer.cls_token] + input_tokens + \
                       [tokenizer.sep_token] + code_line_tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = args.max_input_token_length - len(input_ids)
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        assert len(input_ids) == args.max_input_token_length
        assert len(input_mask) == args.max_input_token_length

        code_line_id = tokenizer.convert_tokens_to_ids(code_line_tokens)
        code_line_item = InputFeatures(commit_id=commit_id,
                                       commit_idx=commit_idx,
                                       commit_ids=commit_ids,
                                       commit_tokens=commit_tokens,
                                       input_ids=input_ids,
                                       input_mask=input_mask,
                                       input_tokens=input_tokens,
                                       manual_features=manual_features,
                                       label=int(label),
                                       line_ids=code_line_id,
                                       line_tokens=code_line_tokens,
                                       line_label=int(code_line_label))
        ret_list.append(code_line_item)
    return ret_list


def convert_examples_to_mix_features(item, buggy_commit_lines_df=None, commit_idx=0):
    # source
    commit_id, files, msg, label, tokenizer, args, manual_features = item

    # prepare code line data
    old_add_code_lines = list(files['added_code'])  # added code lines
    old_delete_code_lines = list(files['removed_code'])  # deleted code lines

    if commit_id in buggy_commit_lines_df['commit hash'].to_list():
        commit_info_df = buggy_commit_lines_df[buggy_commit_lines_df['commit hash'] == commit_id]
        commit_info_df = commit_info_df.reset_index(drop=True)

        commit_info_df['code line'] = commit_info_df['code line'].apply(lambda x: remove_punctuation(x))

        add_code_lines_dict = {remove_punctuation(line): line for line in old_add_code_lines}
        delete_code_lines_dict = {remove_punctuation(line): line for line in old_delete_code_lines}

        commit_info_df_added = commit_info_df[commit_info_df['change type'] == 'added']
        commit_info_df_added = commit_info_df_added.reset_index(drop=True)
        add_code_lines = []
        add_code_line_labels = []
        for i in range(len(commit_info_df_added)):
            if commit_info_df_added['code line'][i] in add_code_lines_dict.keys():
                add_code_lines.append(add_code_lines_dict[commit_info_df_added['code line'][i]])
                add_code_line_labels.append(commit_info_df_added['label'][i])

            else:
                print('added lines matching exists wrong')

        commit_info_df_deleted = commit_info_df[commit_info_df['change type'] == 'deleted']
        commit_info_df_deleted = commit_info_df_deleted.reset_index(drop=True)
        delete_code_lines = []
        delete_code_line_labels = []
        for i in range(len(commit_info_df_deleted)):
            if commit_info_df_deleted['code line'][i] in delete_code_lines_dict.keys():
                delete_code_lines.append(delete_code_lines_dict[commit_info_df_deleted['code line'][i]])
                delete_code_line_labels.append(commit_info_df_deleted['label'][i])
            else:
                print('deleted lines matching exists wrong')
    else:
        add_code_lines = old_add_code_lines
        delete_code_lines = old_delete_code_lines
        add_code_line_labels = [0] * len(add_code_lines)
        delete_code_line_labels = [0] * len(delete_code_lines)

    add_code_lines = [preprocess_code_line(line, False) for line in add_code_lines]
    delete_code_lines = [preprocess_code_line(line, False) for line in delete_code_lines]

    add_code_lines, add_code_line_labels = add_code_line_identify_token(add_code_lines,
                                                                        add_code_line_labels,
                                                                        type_code='added')
    delete_code_lines, delete_code_line_labels = add_code_line_identify_token(delete_code_lines,
                                                                              delete_code_line_labels,
                                                                              type_code='deleted')

    assert len(add_code_lines) == len(add_code_line_labels)
    assert len(delete_code_lines) == len(delete_code_line_labels)

    total_code_lines = add_code_lines + delete_code_lines
    total_code_line_labels = add_code_line_labels + delete_code_line_labels
    total_code_line_tokens = [tokenizer.tokenize(line) for line in total_code_lines]

    # prepare commit data
    added_tokens = []
    removed_tokens = []
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]
    files['added_code'] = add_code_lines[:]
    files['removed_code'] = delete_code_lines[:]
    file_codes = files
    added_codes = file_codes['added_code']
    codes = ''.join([line for line in added_codes])
    added_tokens.extend(tokenizer.tokenize(codes))
    removed_codes = file_codes['removed_code']
    codes = ''.join([line for line in removed_codes])
    removed_tokens.extend(tokenizer.tokenize(codes))
    commit_tokens = msg_tokens + added_tokens + removed_tokens
    commit_tokens = commit_tokens[: args.max_input_token_length - 2]
    commit_tokens = [tokenizer.cls_token] + commit_tokens + [tokenizer.sep_token]
    commit_ids = tokenizer.convert_tokens_to_ids(commit_tokens)
    padding_length = args.max_input_token_length - len(commit_ids)
    commit_ids = commit_ids + ([tokenizer.pad_token_id] * padding_length)

    ret_list = [InputMixFeatures(commit_id, commit_idx, 0, commit_ids, commit_tokens, int(label))]
    for code_line_tokens, code_line, code_line_label in \
            zip(total_code_line_tokens, total_code_lines, total_code_line_labels):
        code_line_tokens = [tokenizer.cls_token] + code_line_tokens + [tokenizer.sep_token]
        code_line_tokens = code_line_tokens[: args.max_input_token_length - 2]
        code_line_id = tokenizer.convert_tokens_to_ids(code_line_tokens)
        padding_length = args.max_input_token_length - len(code_line_id)
        code_line_id = code_line_id + ([tokenizer.pad_token_id] * padding_length)
        code_line_item = InputMixFeatures(commit_id, commit_idx, 1, code_line_id, code_line_tokens,
                                          int(code_line_label))
        ret_list.append(code_line_item)
    return ret_list


def convert_examples_to_commit_features(item, buggy_commit_lines_df=None, commit_idx=0):
    # source
    commit_id, files, msg, label, tokenizer, args, manual_features = item

    # prepare code line data
    old_add_code_lines = list(files['added_code'])  # added code lines
    old_delete_code_lines = list(files['removed_code'])  # deleted code lines

    if commit_id in buggy_commit_lines_df['commit hash'].to_list():
        commit_info_df = buggy_commit_lines_df[buggy_commit_lines_df['commit hash'] == commit_id]
        commit_info_df = commit_info_df.reset_index(drop=True)

        commit_info_df['code line'] = commit_info_df['code line'].apply(lambda x: remove_punctuation(x))

        add_code_lines_dict = {remove_punctuation(line): line for line in old_add_code_lines}
        delete_code_lines_dict = {remove_punctuation(line): line for line in old_delete_code_lines}

        commit_info_df_added = commit_info_df[commit_info_df['change type'] == 'added']
        commit_info_df_added = commit_info_df_added.reset_index(drop=True)
        add_code_lines = []
        add_code_line_labels = []
        for i in range(len(commit_info_df_added)):
            if commit_info_df_added['code line'][i] in add_code_lines_dict.keys():
                add_code_lines.append(add_code_lines_dict[commit_info_df_added['code line'][i]])
                add_code_line_labels.append(commit_info_df_added['label'][i])

            else:
                print('added lines matching exists wrong')

        commit_info_df_deleted = commit_info_df[commit_info_df['change type'] == 'deleted']
        commit_info_df_deleted = commit_info_df_deleted.reset_index(drop=True)
        delete_code_lines = []
        delete_code_line_labels = []
        for i in range(len(commit_info_df_deleted)):
            if commit_info_df_deleted['code line'][i] in delete_code_lines_dict.keys():
                delete_code_lines.append(delete_code_lines_dict[commit_info_df_deleted['code line'][i]])
                delete_code_line_labels.append(commit_info_df_deleted['label'][i])
            else:
                print('deleted lines matching exists wrong')
    else:
        add_code_lines = old_add_code_lines
        delete_code_lines = old_delete_code_lines
        add_code_line_labels = [0] * len(add_code_lines)
        delete_code_line_labels = [0] * len(delete_code_lines)

    add_code_lines = [preprocess_code_line(line, False) for line in add_code_lines]
    delete_code_lines = [preprocess_code_line(line, False) for line in delete_code_lines]

    add_code_lines, add_code_line_labels = add_code_line_identify_token(add_code_lines,
                                                                        add_code_line_labels,
                                                                        type_code='added')
    delete_code_lines, delete_code_line_labels = add_code_line_identify_token(delete_code_lines,
                                                                              delete_code_line_labels,
                                                                              type_code='deleted')
    assert len(add_code_lines) == len(add_code_line_labels)
    assert len(delete_code_lines) == len(delete_code_line_labels)

    # prepare commit data
    added_tokens = []
    removed_tokens = []
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(args.max_msg_length, len(msg_tokens))]
    files['added_code'] = add_code_lines[:]
    files['removed_code'] = delete_code_lines[:]
    file_codes = files
    added_codes = file_codes['added_code']
    codes = ''.join([line for line in added_codes])
    added_tokens.extend(tokenizer.tokenize(codes))
    removed_codes = file_codes['removed_code']
    codes = ''.join([line for line in removed_codes])
    removed_tokens.extend(tokenizer.tokenize(codes))
    commit_tokens = msg_tokens + added_tokens + removed_tokens
    commit_tokens = commit_tokens[: args.max_input_token_length - 2]
    commit_tokens = [tokenizer.cls_token] + commit_tokens + [tokenizer.sep_token]
    commit_ids = tokenizer.convert_tokens_to_ids(commit_tokens)
    padding_length = args.max_input_token_length - len(commit_ids)
    commit_ids = commit_ids + ([tokenizer.pad_token_id] * padding_length)
    ret_list = [InputCommitFeatures(commit_id, commit_idx, commit_ids, commit_tokens, int(label))]
    return ret_list


def convert_examples_to_line_features(item, buggy_commit_lines_df=None):
    # source
    commit_id, files, msg, label, tokenizer, args, manual_features = item

    # prepare code line data
    old_add_code_lines = list(files['added_code'])  # added code lines
    old_delete_code_lines = list(files['removed_code'])  # deleted code lines

    if commit_id in buggy_commit_lines_df['commit hash'].to_list():
        commit_info_df = buggy_commit_lines_df[buggy_commit_lines_df['commit hash'] == commit_id]
        commit_info_df = commit_info_df.reset_index(drop=True)

        commit_info_df['code line'] = commit_info_df['code line'].apply(lambda x: remove_punctuation(x))

        add_code_lines_dict = {remove_punctuation(line): line for line in old_add_code_lines}
        delete_code_lines_dict = {remove_punctuation(line): line for line in old_delete_code_lines}

        commit_info_df_added = commit_info_df[commit_info_df['change type'] == 'added']
        commit_info_df_added = commit_info_df_added.reset_index(drop=True)
        add_code_lines = []
        add_code_line_labels = []
        for i in range(len(commit_info_df_added)):
            if commit_info_df_added['code line'][i] in add_code_lines_dict.keys():
                add_code_lines.append(add_code_lines_dict[commit_info_df_added['code line'][i]])
                add_code_line_labels.append(commit_info_df_added['label'][i])

            else:
                print('added lines matching exists wrong')

        commit_info_df_deleted = commit_info_df[commit_info_df['change type'] == 'deleted']
        commit_info_df_deleted = commit_info_df_deleted.reset_index(drop=True)
        delete_code_lines = []
        delete_code_line_labels = []
        for i in range(len(commit_info_df_deleted)):
            if commit_info_df_deleted['code line'][i] in delete_code_lines_dict.keys():
                delete_code_lines.append(delete_code_lines_dict[commit_info_df_deleted['code line'][i]])
                delete_code_line_labels.append(commit_info_df_deleted['label'][i])
            else:
                print('deleted lines matching exists wrong')
    else:
        add_code_lines = old_add_code_lines
        delete_code_lines = old_delete_code_lines
        add_code_line_labels = [0] * len(add_code_lines)
        delete_code_line_labels = [0] * len(delete_code_lines)

    add_code_lines = [preprocess_code_line(line, False) for line in add_code_lines]
    delete_code_lines = [preprocess_code_line(line, False) for line in delete_code_lines]

    add_code_lines, add_code_line_labels = add_code_line_identify_token(add_code_lines,
                                                                        add_code_line_labels,
                                                                        type_code='added')
    delete_code_lines, delete_code_line_labels = add_code_line_identify_token(delete_code_lines,
                                                                              delete_code_line_labels,
                                                                              type_code='deleted')

    assert len(add_code_lines) == len(add_code_line_labels)
    assert len(delete_code_lines) == len(delete_code_line_labels)

    total_code_lines = add_code_lines + delete_code_lines
    total_code_line_labels = add_code_line_labels + delete_code_line_labels
    total_code_line_tokens = [tokenizer.tokenize(line) for line in total_code_lines]

    ret_list = []
    line_idx = 0
    for code_line_tokens, code_line, code_line_label in \
            zip(total_code_line_tokens, total_code_lines, total_code_line_labels):
        code_line_tokens = [tokenizer.cls_token] + code_line_tokens + [tokenizer.sep_token]
        code_line_tokens = code_line_tokens[: args.max_input_token_length - 2]
        code_line_id = tokenizer.convert_tokens_to_ids(code_line_tokens)
        padding_length = args.max_input_token_length - len(code_line_id)
        code_line_id = code_line_id + ([tokenizer.pad_token_id] * padding_length)
        code_line_item = InputLineFeatures(commit_id, line_idx, code_line_id, code_line_tokens, int(code_line_label))
        line_idx += 1
        ret_list.append(code_line_item)
    return ret_list


manual_features_columns = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
                           'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']


def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, mode='train', buggy_lines_file_path=None):
        # prepare buggy code lines
        assert mode in ['train', 'valid', 'test'], "buggy line file path error"
        if mode == 'train':
            buggy_commit_lines_df_path, _, _ = buggy_lines_file_path
        elif mode == 'valid':
            _, buggy_commit_lines_df_path, _ = buggy_lines_file_path
        else:
            _, _, buggy_commit_lines_df_path = buggy_lines_file_path

        buggy_commit_lines_df = pd.read_pickle(buggy_commit_lines_df_path)

        # jitfine
        self.examples = []
        self.args = args
        changes_filename, features_filename = file_path

        data = []
        ddata = pd.read_pickle(changes_filename)

        features_data = pd.read_pickle(features_filename)
        features_data = convert_dtype_dataframe(features_data, manual_features_columns)

        features_data = features_data[['commit_hash'] + manual_features_columns]

        manual_features = preprocessing.scale(features_data[manual_features_columns].to_numpy())
        features_data[manual_features_columns] = manual_features

        commit_ids, labels, msgs, codes = ddata

        for commit_id, label, msg, code in zip(commit_ids, labels, msgs, codes):
            manual_features = features_data[features_data['commit_hash'] == commit_id][
                manual_features_columns].to_numpy().squeeze()
            data.append((commit_id, code, msg, label, tokenizer, args, manual_features))

        commit_idx = 0
        for item in tqdm(data):
            converted_item = convert_examples_to_features(item,
                                                          buggy_commit_lines_df=buggy_commit_lines_df,
                                                          commit_idx=commit_idx)
            commit_idx += 1
            self.examples.extend(converted_item)

        if mode == 'train':
            for _, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("commit_idx: {}".format(example.commit_idx))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("commit_tokens: {}".format(example.commit_tokens))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("label: {}".format(example.label))
                logger.info("line_ids: {}".format(' '.join(map(str, example.line_ids))))
                logger.info("line_label: {}".format(example.line_label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].commit_idx),
                torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].manual_features),
                torch.tensor(self.examples[item].label),
                torch.tensor(self.examples[item].line_label))


class MixDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, mode='train', buggy_lines_file_path=None):
        # prepare buggy code lines
        assert mode in ['train', 'valid', 'test'], "buggy line file path error"
        if mode == 'train':
            buggy_commit_lines_df_path, _, _ = buggy_lines_file_path
        elif mode == 'valid':
            _, buggy_commit_lines_df_path, _ = buggy_lines_file_path
        else:
            _, _, buggy_commit_lines_df_path = buggy_lines_file_path

        buggy_commit_lines_df = pd.read_pickle(buggy_commit_lines_df_path)

        # jitfine
        self.examples = []
        self.args = args
        changes_filename, features_filename = file_path

        data = []
        ddata = pd.read_pickle(changes_filename)

        features_data = pd.read_pickle(features_filename)
        features_data = convert_dtype_dataframe(features_data, manual_features_columns)

        features_data = features_data[['commit_hash'] + manual_features_columns]

        manual_features = preprocessing.scale(features_data[manual_features_columns].to_numpy())
        features_data[manual_features_columns] = manual_features

        commit_ids, labels, msgs, codes = ddata

        for commit_id, label, msg, code in zip(commit_ids, labels, msgs, codes):
            manual_features = features_data[features_data['commit_hash'] == commit_id][
                manual_features_columns].to_numpy().squeeze()
            data.append((commit_id, code, msg, label, tokenizer, args, manual_features))

        commit_idx = 0
        for item in tqdm(data):
            converted_item = convert_examples_to_mix_features(item,
                                                              buggy_commit_lines_df=buggy_commit_lines_df,
                                                              commit_idx=commit_idx)
            self.examples.extend(converted_item)
            commit_idx += 1

        if mode == 'train':
            for _, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("input_type: {}".format('commit' if int(example.input_type) == 0 else 'code line'))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("label: {}".format(example.label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].commit_idx),
                torch.tensor(self.examples[item].input_type),
                torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].label))


class CommitDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, mode='train', buggy_lines_file_path=None):
        # prepare buggy code lines
        assert mode in ['train', 'valid', 'test'], "buggy line file path error"
        if mode == 'train':
            buggy_commit_lines_df_path, _, _ = buggy_lines_file_path
        elif mode == 'valid':
            _, buggy_commit_lines_df_path, _ = buggy_lines_file_path
        else:
            _, _, buggy_commit_lines_df_path = buggy_lines_file_path

        buggy_commit_lines_df = pd.read_pickle(buggy_commit_lines_df_path)

        # jitfine
        self.examples = []
        self.args = args
        changes_filename, features_filename = file_path

        data = []
        ddata = pd.read_pickle(changes_filename)

        features_data = pd.read_pickle(features_filename)
        features_data = convert_dtype_dataframe(features_data, manual_features_columns)

        features_data = features_data[['commit_hash'] + manual_features_columns]

        manual_features = preprocessing.scale(features_data[manual_features_columns].to_numpy())
        features_data[manual_features_columns] = manual_features

        commit_ids, labels, msgs, codes = ddata

        for commit_id, label, msg, code in zip(commit_ids, labels, msgs, codes):
            manual_features = features_data[features_data['commit_hash'] == commit_id][
                manual_features_columns].to_numpy().squeeze()
            data.append((commit_id, code, msg, label, tokenizer, args, manual_features))

        commit_idx = 0
        for item in tqdm(data):
            converted_item = convert_examples_to_commit_features(item,
                                                                 buggy_commit_lines_df=buggy_commit_lines_df,
                                                                 commit_idx=commit_idx)
            self.examples.extend(converted_item)
            commit_idx += 1

        if mode == 'train':
            for _, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("input_ids: {}".format(' '.join(map(str, example.commit_ids))))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.commit_tokens]))
                logger.info("label: {}".format(example.commit_label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].commit_idx),
                torch.tensor(self.examples[item].commit_ids),
                torch.tensor(self.examples[item].commit_label))


class LineDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, mode='train', buggy_lines_file_path=None):
        # prepare buggy code lines
        assert mode in ['train', 'valid', 'test'], "buggy line file path error"
        if mode == 'train':
            buggy_commit_lines_df_path, _, _ = buggy_lines_file_path
        elif mode == 'valid':
            _, buggy_commit_lines_df_path, _ = buggy_lines_file_path
        else:
            _, _, buggy_commit_lines_df_path = buggy_lines_file_path

        buggy_commit_lines_df = pd.read_pickle(buggy_commit_lines_df_path)

        # jitfine
        self.examples = []
        self.args = args
        changes_filename, features_filename = file_path

        data = []
        ddata = pd.read_pickle(changes_filename)

        features_data = pd.read_pickle(features_filename)
        features_data = convert_dtype_dataframe(features_data, manual_features_columns)

        features_data = features_data[['commit_hash'] + manual_features_columns]

        manual_features = preprocessing.scale(features_data[manual_features_columns].to_numpy())
        features_data[manual_features_columns] = manual_features

        commit_ids, labels, msgs, codes = ddata

        for commit_id, label, msg, code in zip(commit_ids, labels, msgs, codes):
            manual_features = features_data[features_data['commit_hash'] == commit_id][
                manual_features_columns].to_numpy().squeeze()
            data.append((commit_id, code, msg, label, tokenizer, args, manual_features))

        for item in tqdm(data):
            converted_item = convert_examples_to_line_features(item, buggy_commit_lines_df=buggy_commit_lines_df)
            self.examples.extend(converted_item)

        if mode == 'train':
            for _, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("input_ids: {}".format(' '.join(map(str, example.line_ids))))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.line_tokens]))
                logger.info("label: {}".format(example.line_label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].line_idx),
                torch.tensor(self.examples[item].line_ids),
                torch.tensor(self.examples[item].line_label))


def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
    cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
    buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
    buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
    recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    return recall_k_percent_effort


def eval_metrics(result_df):
    pred = result_df['defective_commit_pred']
    y_test = result_df['label']

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary')  # at threshold = 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    #     rec = tp/(tp+fn)

    FAR = fp / (fp + tn)  # false alarm rate
    dist_heaven = math.sqrt((pow(1 - rec, 2) + pow(0 - FAR, 2)) / 2.0)  # distance to heaven

    AUC = roc_auc_score(y_test, result_df['defective_commit_prob'])

    result_df['defect_density'] = result_df['defective_commit_prob'] / result_df['LOC']  # predicted defect density
    result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density

    result_df = result_df.sort_values(by='defect_density', ascending=False)
    actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
    actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

    result_df['cum_LOC'] = result_df['LOC'].cumsum()
    actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

    real_buggy_commits = result_df[result_df['label'] == 1]

    label_list = list(result_df['label'])

    all_rows = len(label_list)

    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * result_df.iloc[-1]['cum_LOC']
    buggy_line_20_percent = result_df[result_df['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

    # find P_opt
    percent_effort_list = []
    predicted_recall_at_percent_effort_list = []
    actual_recall_at_percent_effort_list = []
    actual_worst_recall_at_percent_effort_list = []

    for percent_effort in np.arange(10, 101, 10):
        predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df,
                                                                           real_buggy_commits)
        actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df,
                                                                        real_buggy_commits)
        actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_worst_result_df,
                                                                              real_buggy_commits)

        percent_effort_list.append(percent_effort / 100)

        predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
        actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
        actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

    p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                 (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                  auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

    return f1, AUC, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt


def load_change_metrics_df(data_dir, mode='train'):
    change_metrics = pd.read_pickle(data_dir)
    feature_name = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
    change_metrics = convert_dtype_dataframe(change_metrics, feature_name)

    return change_metrics[['commit_hash'] + feature_name]


def eval_result(result_path, features_path):
    RF_result = pd.read_csv(result_path, sep='\t')

    RF_result.columns = ['test_commit', 'defective_commit_prob', 'defective_commit_pred', 'label']  # for new result

    test_commit_metrics = load_change_metrics_df(features_path, 'test')[['commit_hash', 'la', 'ld']]
    RF_df = pd.DataFrame()
    RF_df['commit_id'] = RF_result['test_commit']
    RF_df = pd.merge(RF_df, test_commit_metrics, left_on='commit_id', right_on='commit_hash', how='inner')
    RF_df = RF_df.drop('commit_hash', axis=1)
    RF_df['LOC'] = RF_df['la'] + RF_df['ld']
    RF_result = pd.merge(RF_df, RF_result, how='inner', left_on='commit_id', right_on='test_commit')
    f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt = eval_metrics(
        RF_result)
    logging.info(
        'F1: {:.4f}, AUC: {:.4f}, Recall@20%Effort: {:.4f}, Effort@20%Recall: {:.4f}, POpt: {:.4f}'.format(
            f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt))


def get_line_level_metrics(label, line_score):
    han_scaler = MinMaxScaler()
    line_score = han_scaler.fit_transform(np.array(line_score).reshape(-1, 1))
    line_score = [float(val) for val in list(line_score)]

    line_df = pd.DataFrame()
    line_df['pred'] = [float(val) for val in list(line_score)]
    line_df['label'] = label
    line_df = line_df.sort_values(by='pred', ascending=False)
    line_df['row'] = np.arange(1, len(line_df) + 1)

    real_buggy_lines = line_df[line_df['label'] == 1]

    top_10_acc = 0
    top_5_acc = 0

    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2 * len(line_df))

    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
        label_list = list(line_df['label'])

        all_rows = len(label_list)

        # find top-10 accuracy
        if all_rows < 10:
            top_10_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_10_acc = np.sum(label_list[:10]) / len(label_list[:10])

        # find top-5 accuracy
        if all_rows < 5:
            top_5_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_5_acc = np.sum(label_list[:5]) / len(label_list[:5])

        # find recall
        LOC_20_percent = line_df.head(int(0.2 * len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num)) / float(len(real_buggy_lines))

        # find effort @20% LOC recall

        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc


def create_path_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
