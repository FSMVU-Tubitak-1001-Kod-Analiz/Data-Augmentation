import argparse
import json
import logging
import multiprocessing
import pickle

import torch
import numpy as np
import torch.utils.data

import os
import sys

from tqdm import tqdm
from transformers import RobertaTokenizer

from .parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from .parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser
from pathlib import Path

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}
import os
# load parsers
parsers = {}
for lang in dfg_function:

    LANGUAGE = Language(str(Path(__file__).parent / './parser/my-languages.so'), lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def get_parsers():
    return parsers


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except Exception as e:
        print(e)
        dfg = []

    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        # self.nl_tokens = nl_tokens
        # self.nl_ids = nl_ids
        # self.url = url


def convert_examples_to_features(item):
    js, tokenizer, args = item
    # code

    parser = parsers[args.lang]
    # extract data flow
    # code_tokens, dfg = extract_dataflow(js['original_string'], parser, args.lang)
    code_tokens, dfg = extract_dataflow(js, parser, args.lang)
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]
    # truncating
    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg = dfg[:args.code_length + args.data_flow_length - len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length
    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    # nl
    # nl = ' '.join(js['docstring_tokens'])
    # nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
    # nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    # nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    # padding_length = args.nl_length - len(nl_ids)
    # nl_ids += [tokenizer.pad_token_id] * padding_length

    # return InputFeatures(code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg, nl_tokens, nl_ids, js['url'])
    return InputFeatures(code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path=None, pool=None):
        prefix = file_path.split('/')[-1][:-6]

        self.args = dotdict()
        self.args.code_length = 256
        self.args.data_flow_length = 64
        self.args.lang = 'java'
        self.args.nl_length = 128
        # cache_file = args.output_dir + '/' + prefix + '.pkl'
        # if os.path.exists(cache_file):
        #     self.examples = pickle.load(open(cache_file, 'rb'))
        # else:
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                js = js["function"]
                data.append((js, tokenizer, self.args))
        self.examples = []
        for j, i in enumerate(tqdm(data)):
            self.examples.append(convert_examples_to_features(i))
        # self.examples = pool.map(convert_examples_to_features, tqdm(data[:10], total=10))
        # pickle.dump(self.examples, open(cache_file, 'wb'))

        if True:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))
                # logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                # logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool_)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx))


def build_dataset(file_path):
    cpu_cont = 16
    pool = multiprocessing.Pool(cpu_cont)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    return TextDataset(tokenizer, file_path, pool)