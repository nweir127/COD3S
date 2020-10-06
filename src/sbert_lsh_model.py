global Device
from typing import List, Tuple, Callable, Optional
import numpy as np
from src.utils import *
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from nearpy.hashes import RandomBinaryProjections
from scipy.spatial.distance import hamming, cosine
from copy import deepcopy
from sentence_transformers import SentenceTransformer


class LSHModel:
    def __init__(self, OPTS, device):
        self.OPTS: Config = deepcopy(OPTS)
        self.comparator: Callable[[np.ndarray, np.ndarray], float]
        self.hasher = None
        self.do_lsh: bool = False
        self.dimension: int = -1
        self.device = device
        self.batch_size: int = OPTS.batch_size

        self.do_lsh = True

        print("initializing random projection LSH model")
        self.hasher = RandomBinaryProjections('rbp_perm', projection_count=(OPTS.lsh_dim if  hasattr(OPTS, 'lsh_dim') else 1), rand_seed=1234)
        self.comparator = lambda x, y: hamming(*[
            np.fromstring(self.hasher.hash_vector(i)[0], 'u1') - ord('0')
            for i in [x, y]])

        self.comparator = lambda x, y: cosine(x, y)


    def compute_distances(self, refs: List[str], cands: List[str]) -> np.ndarray:
        '''
        :param refs: list of reference sentences
        :param cands: list of candidate sentences to compute similarity distances from references
        :return:
        '''
        assert len(refs) == len(cands)
        results = np.zeros(len(refs))
        i = 0
        for batch in batched(zip(refs, cands), self.batch_size, total=len(refs)):
            (ref_b, cands_b) = list(zip(*batch))
            assert len(ref_b) <= self.batch_size
            [ref_features, cand_features] = [self.get_embeddings(x) for x in [ref_b, cands_b]]


            if i == 0:
                print(f"comparing vectors of dimension {ref_features.shape[-1]}")
            results[i:i + len(ref_b)] = np.fromiter(
                map(lambda args: self.comparator(*args), zip(ref_features, cand_features)), dtype=float)
            i += len(ref_b)

        return results

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        '''
        retrieve np array of sentence embeddings from sentence iterator
        :param sents: set of sentence strings
        :return: extracted embeddings
        '''
        raise NotImplementedError()


class BertLSHModel(LSHModel):
    def __init__(self, OPTS, device):
        super(BertLSHModel, self).__init__(OPTS, device)
        self.dimension = 768

        print('loading model...')
        if OPTS.bert_type == 'pbert':
            bert_classifier_model = BertForSequenceClassification.from_pretrained(OPTS.bert_dir)
            self.bert_model = bert_classifier_model.bert
        elif OPTS.bert_type == 'bert':
            self.bert_model = BertModel.from_pretrained('bert-base-cased')
        self.bert_model.eval()
        self.device.move(self.bert_model)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.hasher.reset(dim=self.dimension)

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        _, inputs = bert_utils.collate_features(sents, self.tokenizer, seq_len=128)
        features = bert_utils.extract_features(inputs, self.bert_model, device=self.device)
        return features


class SBERTLSHModel(LSHModel):
    def __init__(self, OPTS, device):
        super(SBERTLSHModel, self).__init__(OPTS, device)
        self.dimension = 1024 if 'large' in OPTS.sbert_type else 768

        print(f'loading SBERT {OPTS.sbert_type} model...')
        # self.embedder = SentenceTransformer(f"{OPTS.sbert_type}-nli-mean-tokens")
        self.embedder = SentenceTransformer(f"{OPTS.sbert_type}-nli-stsb-mean-tokens")
        self.embedder.eval()
        self.device.move(self.embedder)
        self.embedder.eval()

        self.hasher.reset(dim=self.dimension)

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        all_embeddings = self.embedder.encode(sents, batch_size=self.batch_size)
        return np.stack(all_embeddings)
