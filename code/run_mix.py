from __future__ import absolute_import, division, print_function
import os
import argparse
import logging

import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer, RobertaModel, T5Config,
                          T5ForConditionalGeneration)
from tqdm import tqdm
from defect_represent_model import Model
from util import MixDataset, eval_result, preprocess_code_line, \
    get_line_level_metrics, create_path_if_not_exist, CommitDataset, LineDataset

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}


def set_seed(args):
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def train(args, train_dataset, model, eval_dataset, tokenizer):
    """ Train the model """

    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 5
    args.warmup_steps = 0
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    best_f1 = 0
    global_step = 0
    model.zero_grad()
    patience = 0

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            if args.run_line or args.run_commit:
                (_, input_ids, label) = [x.to(args.device) for x in batch]
            else:
                (_, _, input_ids, label) = [x.to(args.device) for x in batch]
            model.train()
            loss, _ = model(input_ids, label)
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            avg_loss = round(tr_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.save_steps == 0:

                results = evaluate(args, model, eval_when_training=True, eval_dataset=eval_dataset)

                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    logger.info("  " + "*" * 20)
                    logger.info("  Best f1:%s", round(best_f1, 4))
                    logger.info("  " + "*" * 20)

                    checkpoint_prefix = 'checkpoint-best-f1'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                    patience = 0
                    torch.save({
                        'epoch': idx,
                        'step': step,
                        'patience': patience,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                else:
                    patience += 1
                    if patience > args.patience * 5:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return


def evaluate(args, model, eval_when_training=False, eval_dataset=None):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    y_trues = []
    logits = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        (commit_idx, input_ids, label) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit = model(input_ids, label)
            torch.cuda.empty_cache()
            eval_loss += loss.mean().item()
            y_trues.append(label.cpu().numpy())
            logits.append(logit.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    y_preds = logits[:, 1] > best_threshold

    # calculate scores
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, best_threshold=0.5, commit_dataset=None, line_dataset=None):
    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    model.eval()

    # code commit results
    commit_eval_sampler = SequentialSampler(commit_dataset)
    commit_eval_dataloader = DataLoader(commit_dataset, sampler=commit_eval_sampler, batch_size=args.eval_batch_size)

    # Eval
    logger.info("***** Running Commit Test *****")
    logger.info("  Num examples = %d", len(commit_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    commit_idxs = []
    commit_logits = []
    y_trues = []
    # set_seed(args)

    for batch in tqdm(commit_eval_dataloader, total=len(commit_eval_dataloader)):
        (commit_idx, input_ids, label) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit = model(input_ids, label)
            commit_idxs.append(commit_idx.cpu().numpy())
            commit_logits.append(logit.cpu().numpy())
            y_trues.append(label.cpu().numpy())

    commit_idxs = np.concatenate(commit_idxs, 0)
    commit_logits = np.concatenate(commit_logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_prob = commit_logits[:, 1]
    y_preds = commit_logits[:, 1] > best_threshold

    commit_result_df = pd.DataFrame({'commit_idxs': commit_idxs, 'y_preds': y_preds, 'y_prob': y_prob,
                                     'y_trues': y_trues})
    commit_ids = []
    commit_tokens = []
    exam_commit_idxs = []
    for example in commit_dataset.examples:
        if len(commit_ids) == 0 or example.commit_id not in commit_ids:
            commit_ids.append(example.commit_id)
            commit_tokens.append(example.commit_tokens)
            exam_commit_idxs.append(example.commit_idx)

    commit_result_df['commit_ids'] = commit_ids

    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')

    result = {
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold": best_threshold,
    }
    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    commit_result = pd.DataFrame(commit_result_df, columns=['commit_ids', 'y_prob', 'y_preds', 'y_trues'])
    commit_result.to_csv(os.path.join(args.output_dir, "predictions.csv"), sep='\t', index=None)

    # code line results
    line_eval_sampler = SequentialSampler(line_dataset)
    line_eval_dataloader = DataLoader(line_dataset, sampler=line_eval_sampler, batch_size=args.eval_batch_size)

    # Eval
    logger.info("***** Running Code Line Test *****")
    logger.info("  Num examples = %d", len(line_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    line_idxs = []
    line_logits = []
    line_labels = []
    # set_seed(args)

    for batch in tqdm(line_eval_dataloader, total=len(line_eval_dataloader)):
        (line_idx, input_ids, label) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit = model(input_ids, label)
            line_idxs.append(line_idx.cpu().numpy())
            line_logits.append(logit.cpu().numpy())
            line_labels.append(label.cpu().numpy())

    line_idxs = np.concatenate(line_idxs, 0)
    line_logits = np.concatenate(line_logits, 0)
    line_labels = np.concatenate(line_labels, 0)
    line_probs = line_logits[:, 1]
    line_preds = line_logits[:, 1] > best_threshold

    line_result_df = pd.DataFrame({'line_idxs': line_idxs, 'line_preds': line_preds, 'line_probs': line_probs,
                                   'line_labels': line_labels})
    line_commit_ids = []
    for example in line_dataset.examples:
        line_commit_ids.append(example.commit_id)
    line_result_df['commit_ids'] = line_commit_ids

    commit2codes, idx2label = commit_with_codes(args.buggy_line_filepath, tokenizer)
    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []

    for commit_id, pred, label, commit_token in zip(commit_ids, y_preds, y_trues, commit_tokens):
        # calculate
        if int(label) == 1 and int(pred) == 1 and '[ADD]' in commit_token:
            cur_codes = commit2codes[commit2codes['commit_id'] == commit_id]
            cur_labels = idx2label[idx2label['commit_id'] == commit_id]
            code_line_nums = len(set(cur_labels['idx'].to_list()))
            cur_line_sc = line_result_df[line_result_df['commit_ids'] == commit_id]['line_probs']
            cur_line_sc = cur_line_sc.tolist()

            assert code_line_nums == len(cur_line_sc), \
                'The number of code lines does not match, commit_idxs {}, idx {}'.format(len(cur_line_sc),
                                                                                         code_line_nums)
            line_idx = list(range(len(cur_line_sc)))
            cur_line_sc = pd.DataFrame({'idx': line_idx, 'line_score': cur_line_sc})

            cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, \
            cur_top_10_acc, cur_top_5_acc = calc_line_level_metrics(cur_codes,
                                                                    cur_labels,
                                                                    cur_line_sc,
                                                                    args.only_adds)
            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
            top_10_acc.append(cur_top_10_acc)
            top_5_acc.append(cur_top_5_acc)

    logger.info(
        'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}, Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
            round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
            round(np.mean(top_20_percent_LOC_recall), 4),
            round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
    )

    return result


def commit_with_codes(filepath, tokenizer):
    data = pd.read_pickle(filepath)
    data['code line'] = data['code line'].apply(lambda x: preprocess_code_line(x))

    commit2codes = []
    idx2label = []
    for _, item in data.iterrows():
        # commit_id, idx, changed_type, label, raw_changed_line, changed_line = item

        prj, file, changed_type, commit_id, idx, changed_line, label = item

        line_tokens = [token.replace('\u0120', '') for token in tokenizer.tokenize(changed_line)]
        for token in line_tokens:
            commit2codes.append([commit_id, idx, changed_type, token])
        idx2label.append([commit_id, idx, label])
    # each token will corresponding to a commit_id and a idx
    commit2codes = pd.DataFrame(commit2codes, columns=['commit_id', 'idx', 'changed_type', 'token'])
    idx2label = pd.DataFrame(idx2label, columns=['commit_id', 'idx', 'label'])
    return commit2codes, idx2label


def calc_line_level_metrics(commit2codes, idx2label, idx2linescore, only_adds=False):
    # calculate score for each line in commit
    if only_adds:
        commit2codes = commit2codes[commit2codes['changed_type'] == 'added']  # only count for added lines
    commit2codes = commit2codes.drop('commit_id', axis=1)
    commit2codes = commit2codes.drop('changed_type', axis=1)

    result_df = pd.merge(commit2codes, idx2label, how='inner', on='idx')
    result_df = pd.merge(result_df, idx2linescore, how='left', on='idx')
    result_df['line_score'].fillna(0, inplace=True)

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = \
        get_line_level_metrics(result_df['label'].tolist(), result_df['line_score'].tolist())

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", nargs=2, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_seed', type=int, default=123456,
                        help="random seed for data order initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    parser.add_argument('--feature_size', type=int, default=14,
                        help="Number of features")
    parser.add_argument('--num_labels', type=int, default=2,
                        help="Number of labels")
    parser.add_argument('--semantic_checkpoint', type=str, default=None,
                        help="Best checkpoint for semantic feature")
    parser.add_argument('--manual_checkpoint', type=str, default=None,
                        help="Best checkpoint for manual feature")
    parser.add_argument('--max_msg_length', type=int, default=64,
                        help="Number of labels")
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stop')
    parser.add_argument("--only_adds", action='store_true',
                        help="Whether to run eval on the only added lines.")
    parser.add_argument("--buggy_line_filepath", type=str,
                        help="complete buggy line-level  data file for RQ3")
    parser.add_argument('--head_dropout_prob', type=float, default=0.1,
                        help="Number of labels")
    parser.add_argument('--no_abstraction', type=float,
                        help="no_abstraction")

    # new args
    parser.add_argument('--max_codeline_length', type=int, default=256,
                        help="max_codeline_length")
    parser.add_argument('--max_codeline_token_length', type=int, default=64,
                        help="max_codeline_token_length")
    parser.add_argument("--buggy_lines_file", nargs=3, type=str, required=True,
                        help="The input xxx_buggy_commit_lines_df.pkl (a .pkl file).")

    parser.add_argument('--dp_loss_weight', type=float, default=0.3,
                        help="The loss weight of the defect prediction task")
    parser.add_argument('--dl_loss_weight', type=float, default=0.7,
                        help="The loss weight of the defect localization task")

    parser.add_argument('--max_input_token_length', type=int, default=512,
                        help="max input token length")
    parser.add_argument("--model_type", default="robert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--mtl_checkpoint", action='store_true',
                        help="Whether to load mtl representation.")
    parser.add_argument("--mtl_model_checkpoint", default=None, type=str,
                        help="The mtl model checkpoint to be fine-tuned.")

    parser.add_argument("--run_line", action='store_true',
                        help="Whether to run line-level defect code representation.")
    parser.add_argument("--run_commit", action='store_true',
                        help="Whether to run commit-level defect code representation.")

    args = parser.parse_args()
    return args


def main(args):
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.num_labels
    config.feature_size = args.feature_size
    config.hidden_dropout_prob = args.head_dropout_prob
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
    else:
        model = model_class(config)

    model.resize_token_embeddings(len(tokenizer))
    logger.info("Training/evaluation parameters %s", args)

    model = Model(model, config, tokenizer, args)

    # Training
    if args.do_train:
        if args.semantic_checkpoint:
            semantic_checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.semantic_checkpoint, '{}'.format(semantic_checkpoint_prefix))
            logger.info("Loading semantic checkpoint from {}".format(output_dir))
            checkpoint = torch.load(output_dir)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if args.manual_checkpoint:
            manual_checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.manual_checkpoint, '{}'.format(manual_checkpoint_prefix))
            logger.info("Loading manual checkpoint from {}".format(output_dir))
            checkpoint = torch.load(output_dir)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if args.mtl_checkpoint:
            output_dir = args.mtl_model_checkpoint
            logger.info("Loading mtl checkpoint from {}".format(output_dir))
            checkpoint = torch.load(output_dir)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(args.device)

        if args.run_line:
            train_dataset = LineDataset(tokenizer, args, file_path=args.train_data_file, mode='train',
                                        buggy_lines_file_path=args.buggy_lines_file)
        elif args.run_commit:
            train_dataset = CommitDataset(tokenizer, args, file_path=args.train_data_file, mode='train',
                                          buggy_lines_file_path=args.buggy_lines_file)
        else:
            train_dataset = MixDataset(tokenizer, args, file_path=args.train_data_file, mode='train',
                                       buggy_lines_file_path=args.buggy_lines_file)

        eval_dataset = CommitDataset(tokenizer, args, file_path=args.eval_data_file, mode='valid',
                                     buggy_lines_file_path=args.buggy_lines_file)

        train(args, train_dataset, model, eval_dataset, tokenizer)

    # Evaluation
    results = {}

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        checkpoint = torch.load(output_dir, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
        model.to(args.device)
        commit_dataset = CommitDataset(tokenizer, args, file_path=args.test_data_file, mode='test',
                                       buggy_lines_file_path=args.buggy_lines_file)
        line_dataset = LineDataset(tokenizer, args, file_path=args.test_data_file, mode='test',
                                   buggy_lines_file_path=args.buggy_lines_file)

        test(args, model, tokenizer, best_threshold=0.5, commit_dataset=commit_dataset, line_dataset=line_dataset)
        eval_result(os.path.join(args.output_dir, "predictions.csv"), args.test_data_file[-1])

    return results


if __name__ == "__main__":
    cur_args = parse_args()
    create_path_if_not_exist(cur_args.output_dir)
    # Set seed
    set_seed(cur_args)
    main(cur_args)
