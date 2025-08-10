from collections import defaultdict
from search_r1.llm_agent.thinking_prompts import revision_prompts
import pickle
from transformers import AutoTokenizer

import os

class PrefixIndex:
    def __init__(self,tokenizer = None):
        if os.path.exists('/workspace/repository/verl/verl/workers/actor/prefix_index_random_15.pkl'):
            with open('/workspace/repository/verl/verl/workers/actor/prefix_index_random_15.pkl', 'rb') as f:
                data = pickle.load(f)
            self.vocabulary = data['vocabulary']
            self.vocabulary_index = data['vocabulary_index']
            self.index = data['index']
        else:
            self.tokenizer = tokenizer

            ks = []
            for prompt in revision_prompts:
                tokens = self.tokenizer(revision_prompts[prompt], return_tensors='pt')['input_ids'].tolist()[0]
                ks.append(tokens)

            # ks: 二维数组
            vocabulary = set()
            for seq in ks:
                for token in seq:
                    vocabulary.add(token)
            self.vocabulary = list(vocabulary)
            self.vocabulary_index = {token: i for i, token in enumerate(self.vocabulary)}

            self.index = defaultdict(lambda : [0] * len(self.vocabulary))  # key: prefix tuple, value: list of next values
            for seq in ks:
                for i in range(len(seq) - 1):  # 不包括最后一个元素
                    prefix = tuple(seq[:i+1])
                    next_val = seq[i+1]
                    self.index[prefix][self.vocabulary_index[next_val]] += 1

            self.index = dict(self.index)

            for k, v in self.index.items():
                total = sum(v)
                for i, count in enumerate(v):
                    self.index[k][i] = count / total

    def query(self, seq, next_token):
        index = self.vocabulary_index[next_token]
        return self.index[tuple(seq)][index]