# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import torch.nn as nn
import random
from sklearn.metrics import f1_score
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers.data.processors.utils import InputExample
from longformer.longformer import Longformer
from collections import OrderedDict
import codecs
from load_data import load_harsh_data#load_train_data, load_dev_data, load_test_data
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_roberta import RobertaClassificationHead

import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)


from transformers import (
    # MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    # get_linear_schedule_with_warmup,
    WarmupLinearSchedule,
)

# MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
#     [
#         (DistilBertConfig, DistilBertForSequenceClassification),
#         (AlbertConfig, AlbertForSequenceClassification),
#         (CamembertConfig, CamembertForSequenceClassification),
#         (XLMRobertaConfig, XLMRobertaForSequenceClassification),
#         (BartConfig, BartForSequenceClassification),
#         (RobertaConfig, RobertaForSequenceClassification),
#         (BertConfig, BertForSequenceClassification),
#         (XLNetConfig, XLNetForSequenceClassification),
#         (FlaubertConfig, FlaubertForSequenceClassification),
#         (XLMConfig, XLMForSequenceClassification),
#     ]
# )

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

# MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
# print('MODEL_TYPES:', MODEL_TYPES)

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)
# print('ALL_MODELS:', ALL_MODELS)
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, dev_dataloader, test_dataloader, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    max_dev_f1 = 0.0
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to global_step of last saved checkpoint from model path
    #     try:
    #         global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     except ValueError:
    #         global_step = 0
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #
    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if global_step % 5000 == 0:
                # if global_step % 5 == 0:
                    dev_f1 = evaluate(args, model, tokenizer, dev_dataloader, prefix='dev set')
                    if dev_f1 > max_dev_f1:
                        max_dev_f1 = dev_f1
                        test_f1 = evaluate(args, model, tokenizer, test_dataloader, prefix='test set')



                        '''# Save model checkpoint'''
                        raw_output_dir = '/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/longformer_hypo_only/'
                        output_dir = os.path.join(raw_output_dir, "f1.dev.{dev}.test{test}".format(dev = max_dev_f1, test = test_f1))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    print('>>dev_f1:', dev_f1, ' max_dev_f1:', max_dev_f1)



            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, eval_dataloader, pair_len2size, hypo_len2size, prefix="test set"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    for eval_task in eval_task_names:


        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        # logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        print('preds:', sum(preds), len(preds))
        print('out_label_ids:', sum(out_label_ids), len(out_label_ids))
        f1_pos = f1_score(list(out_label_ids), list(preds), pos_label= 0, average='binary')
        # f1_neg = f1_score(list(out_label_ids), list(preds), pos_label= 1, average='binary')
        print('>>test_f1_pos:', f1_pos)

        assert len(pair_len2size) == len(list(preds))
        assert len(hypo_len2size) == len(list(preds))

        out_label_ids = list(out_label_ids)
        preds = list(preds)
        '''pair f1'''
        pair_group_size = len(set(pair_len2size))
        pair_f1_list = []
        for i in range(pair_group_size):
            gold_list_i = []
            pred_list_i = []
            for id, group_id in enumerate(pair_len2size):
                if group_id == i: #this example belongs to this group
                    gold_list_i.append(out_label_ids[id])
                    pred_list_i.append(preds[id])
            f1_pos_i = f1_score(gold_list_i, pred_list_i, pos_label= 0, average='binary')
            pair_f1_list.append(f1_pos_i)
        print('pair_f1_list:', pair_f1_list)
        '''hypo f1'''
        hypo_group_size = len(set(hypo_len2size))
        hypo_f1_list = []
        for i in range(hypo_group_size):
            gold_list_i = []
            pred_list_i = []
            for id, group_id in enumerate(hypo_len2size):
                if group_id == i: #this example belongs to this group
                    gold_list_i.append(out_label_ids[id])
                    pred_list_i.append(preds[id])
            f1_pos_i = f1_score(gold_list_i, pred_list_i, pos_label= 0, average='binary')
            hypo_f1_list.append(f1_pos_i)

        print('hypo_f1_list:', hypo_f1_list)

    return f1_pos



def load_and_cache_examples(args, task, filename, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    # cached_features_file = os.path.join(
    #     args.data_dir,
    #     "cached_{}_{}_{}_{}".format(
    #         "dev" if evaluate else "train",
    #         list(filter(None, args.model_name_or_path.split("/"))).pop(),
    #         str(args.max_seq_length),
    #         str(task),
    #     ),
    # )
    # if os.path.exists(cached_features_file) and not args.overwrite_cache:
    #     logger.info("Loading features from cached file %s", cached_features_file)
    #     features = torch.load(cached_features_file)
    # else:
    # logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    '''label_list: ['entailment', 'not_entailment']'''

    if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]
    # examples = (
    #     processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
    # )
    # examples = get_DUC_examples(filename)
    if filename == 'train':
        examples = load_harsh_data('train', hypo_only=True)
    elif filename == 'dev':
        examples = load_harsh_data('dev', hypo_only=True)
    else:
        examples = load_harsh_data('test', hypo_only=True)



    '''count the length of (premise, hypothesis) and hypothesis only'''
    pair_len2size = []# defaultdict(int)
    hypo_len2size = []#defaultdict(int)
    for ex in examples:
        hypo_len = len(ex.text_b.strip().split())
        prem_len = len(ex.text_a.strip().split())
        pair_len = prem_len+hypo_len

        pair_id = int(pair_len/150)
        if pair_id > 11:
            pair_id = 11
        pair_len2size.append(pair_id)

        hypo_id = int(hypo_len/50)
        if hypo_id > 4:
            hypo_id = 4
        hypo_len2size.append(hypo_id)#[hypo_id] = hypo_len2size.get(hypo_id, 0)+=1


    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.pad_token_id,
        # pad_token_segment_id=tokenizer.pad_token_type_id,
    )
    # if args.local_rank in [-1, 0]:
    #     logger.info("Saving features into cached file %s", cached_features_file)
    #     torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset, pair_len2size, hypo_len2size

def get_DUC_examples(filename):
    #/export/home/Dataset/para_entail_datasets/DUC/train_in_entail.txt
    print('loading...', filename)
    readfile = codecs.open(filename, 'r', 'utf-8')
    start = False
    examples = []
    guid_id = -1
    pos_size = 0
    neg_size = 0
    for line in readfile:
        if len(line.strip()) == 0:
            start = False
        else:
            parts = line.strip().split('\t')
            if parts[0] == 'document>>':
                start = True
                premise = parts[1].strip()
            elif parts[0] == 'positive>>':
                guid_id+=1
                pos_hypo = parts[1].strip()
                examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=pos_hypo, label='entailment'))
                pos_size+=1
            elif parts[0] == 'negative>>' and parts[1] != '#ShuffleWord#>>' and parts[1] != '#RemoveWord#>>':
                guid_id+=1
                neg_hypo = parts[2].strip()

                if filename.find('train_in_entail') > -1:
                    examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                    neg_size+=1
                else:
                    rand_prob = random.uniform(0, 1)
                    if rand_prob > 3/4:
                        examples.append(InputExample(guid=str(guid_id), text_a=premise, text_b=neg_hypo, label='not_entailment'))
                        neg_size+=1

    print('>>pos:neg: ', pos_size, neg_size)
    return examples


class LongformerForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # self.roberta = RobertaModel(config)
        self.roberta = Longformer(config)
        self.classifier = RobertaClassificationHead(config)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        # inputs_embeds=None, #the BertModel in Longformer does not have this parameter
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss().cuda()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



def main():

    parser = argparse.ArgumentParser()

    # Required parameters
    # parser.add_argument(
    #     "--data_dir",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    # )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        # help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        # help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    # parser.add_argument(
    #     "--output_dir",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="The output directory where the model predictions and checkpoints will be written.",
    # )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=500,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    # parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    # parser.add_argument(
    #     "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    # )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    # parser.add_argument(
    #     "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    # )
    # parser.add_argument(
    #     "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    # )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # if (
    #     os.path.exists(args.output_dir)
    #     and os.listdir(args.output_dir)
    #     and args.do_train
    #     and not args.overwrite_output_dir
    # ):
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
    #             args.output_dir
    #         )
    #     )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    longformer_path = '/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/longformer_full_train/roberta-f1.dev.0.779148541443071.test0.7792291259985912'
    '''config file and model should load from longformer-large-4096; tokenizer from roberta-large'''
    config = AutoConfig.from_pretrained(
        longformer_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = LongformerForSequenceClassification.from_pretrained(
        longformer_path,
        from_tf=bool(".ckpt" in longformer_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer.max_len = model.config.max_position_embeddings



    # longformer_path = 'longformer-large-4096'

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    # if args.do_train:
    # train_filename = '/export/home/Dataset/para_entail_datasets/DUC/test_in_entail.txt'
    # train_filename = '/export/home/Dataset/para_entail_datasets/CNN_DailyMail/test_in_entail.txt'
    # train_filename = 'train'
    # train_dataset = load_and_cache_examples(args, args.task_name, train_filename, tokenizer, evaluate=False)


    # dev_filename = 'dev'
    # dev_dataset = load_and_cache_examples(args, args.task_name, dev_filename, tokenizer, evaluate=True)
    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # dev_sampler = SequentialSampler(dev_dataset)
    # dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.eval_batch_size)


    test_filename = 'test'
    test_dataset, pair_len2size, hypo_len2size = load_and_cache_examples(args, args.task_name, test_filename, tokenizer, evaluate=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    test_f1 = evaluate(args, model, tokenizer, test_dataloader, pair_len2size, hypo_len2size, prefix='test set')
    print('test_f1:', test_f1)
    # global_step, tr_loss = train(args, None, None, test_dataloader, model, tokenizer)
    # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)



if __name__ == "__main__":
    main()
    '''
    ERROR: longformer 0.1 has requirement torch==1.2.0, but you'll have torch 1.4.0 which is incompatible.
    ERROR: longformer 0.1 has requirement transformers==2.0.0, but you'll have transformers 2.8.0 which is incompatible.
    '''
    '''
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u test_longformer.py --model_type roberta --model_name_or_path roberta-base --task_name rte
    '''
