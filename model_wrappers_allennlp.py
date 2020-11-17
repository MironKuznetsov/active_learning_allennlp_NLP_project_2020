from allennlp.data import allennlp_collate
from allennlp.predictors import SentenceTaggerPredictor
from libact.base.interfaces import ProbabilisticModel
from libact.base.dataset import Dataset
# from libact.query_strategies import RandomSampling
from allennlp.data.tokenizers import Token


import torch
import numpy as np
from overrides import overrides
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
import gc
import random
import math
import sys

from isanlp.ru.processor_tokenizer_ru import ProcessorTokenizerRu
from torch.utils.data import DataLoader


def find_in_between(offsets, start, end):
    res = []
    for i, offset in enumerate(offsets):
        if start <= offset and offset <= end:
            res.append(i)
    return res


def tokenize_X(X):
    tokenizer = ProcessorTokenizerRu()
    res = []
    for sent in X:
        res.append(tokenizer(sent))

    return [[e.text for e in res_sent] for res_sent in res]


def convert_to_bio_format(X, y):
    tokenizer = ProcessorTokenizerRu()
    final_res = []
    X_tokenized = []
    for i, sent_y in enumerate(y):
        sent = X[i]

        tokens = tokenizer(sent)
        offsets = [token.begin for token in tokens]
        X_tokenized.append([token.text for token in tokens])

        good_ys = ['O'] * len(tokens)

        if not sent_y or sent_y[0] == 'None':
            final_res.append(good_ys)
            continue

        for w_y in sent_y:
            positions = find_in_between(offsets, w_y['start'], w_y['end'])
            good_ys[positions[0]] = 'B-' + w_y['tag']

            for pos in positions[1:]:
                good_ys[pos] = 'I-' + w_y['tag']

        final_res.append(good_ys)

    return X_tokenized, final_res

class CustomSentenceTaggerPredictor(SentenceTaggerPredictor):

    def __init__(self,  model, dataset_reader, batch_size):
        super().__init__(model, dataset_reader)
        self.batch_size = batch_size
    @overrides
    def predict(self, list_of_lists):
        preds = {'logits': [], 'class_probabilities': [], 'words': [], 'tags': []}
        i = 0
        while i < len(list_of_lists):
            preds_0 = self.predict_batch_json(list_of_lists[i:i + self.batch_size])
            for pred in preds_0:

                for key in preds.keys():
                    if key in ['logits', 'class_probabilities']:
                        buffer = np.array(pred[key])
                    else:
                        buffer = pred[key][:len(pred['words'])]
                    preds[key].append(buffer)
                
            i+=self.batch_size
        preds_0 = self.predict_batch_json(list_of_lists[i-self.batch_size:len(list_of_lists)])
        for pred in preds_0:
            for key in preds.keys():
                if key in ['logits', 'class_probabilities']:
                    buffer = np.array(pred[key])
                else:
                    buffer = pred[key][:len(pred['words'])]
                preds[key].append(buffer)
        return preds

    @overrides
    def _json_to_instance(self, sentence):
        tokens = [Token(e) for e in sentence]
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def _batch_json_to_instances(self, list_of_lists):
        """
        Converts a list of JSON objects into a list of `Instance`s.
        By default, this expects that a "batch" consists of a list of JSON blobs which would
        individually be predicted by `predict_json`. In order to use this method for
        batch prediction, `_json_to_instance` should be implemented by the subclass, or
        if the instances have some dependency on each other, this method should be overridden
        directly.
        """
        instances = []
        for sentence in list_of_lists:
            instances.append(self._json_to_instance(sentence))
        return instances


class LibActNN(ProbabilisticModel):
    def __init__(self,
                 model_ctor,
                 trainer_ctor,
                 reader,
                 vocab,
                 batch_size=16,
                 bs_pred=256,
                 retrain_epochs=3,
                 iter_retrain=1,
                 train_from_scratch=True,
                 valid_ratio=0.25,
                 string_input=True,
                 self_training_samples=0,
                 autofill_similar_objects=False,
                 n_upsample_positive=0,
                 additional_X=None,
                 additional_y=None,
                 ):
        self._model_ctor = model_ctor
        self._trainer_ctor = trainer_ctor
        self._model = None
        self._trainer = None
        self._batch_size = batch_size
        self._bs_pred = bs_pred
        self._retrain_epochs = retrain_epochs
        self._batch_size = batch_size
        self._iter_retrain = iter_retrain
        self._train_from_scratch = train_from_scratch
        self._valid_ratio = valid_ratio
        self._string_input = string_input
        self._self_training_samples = self_training_samples
        self._autofill_similar_objects = autofill_similar_objects
        self._n_upsample_positive = n_upsample_positive
        self.reader = reader
        self._additional_X = additional_X if additional_X else []
        self._additional_y = additional_y if additional_y else []
        self.vocab = vocab
        self._iter = 0

    def _predict_core(self, X):
        if self._string_input:
            X = tokenize_X(X)

        gc.collect()
        torch.cuda.empty_cache()
        # print('X:',X)
        predictor = CustomSentenceTaggerPredictor(self._model, self.reader, self._bs_pred)
        preds = predictor.predict(X)
        # probs = preds['class_probabilities']
        # return self._model.predict(X)
        return preds
#TODO переделать эти две функции
    def predict_proba(self, X):
#         print('lol', np.asarray(self._predict_core(X)['class_probabilities']).reshape(-1, 1)[0])
        
#         print('lolkek', np.asarray(self._predict_core(X)['class_probabilities'])[0])
#         print('kek', self._predict_core(X)['class_probabilities'][0])
        
        return np.array(self._predict_core(X)['class_probabilities'])#.reshape(-1, 1)

    def predict(self, X):
        return self._predict_core(X)

    def train(self, libact_dataset, new_indexes=None):
        # print('New indexes', new_indexes)

        if new_indexes is not None and self._autofill_similar_objects:
            n_updated = 0
            for new_ind in new_indexes:
                new_example = libact_dataset.data[new_ind]

                for i in range(len(libact_dataset.data)):
                    if libact_dataset.data[i][1] is not None:
                        continue
                    else:
                        train_object = libact_dataset.data[i][0]
                        if train_object == new_example[0]:
                            libact_dataset.data[i] = (train_object, new_example[1])
                            n_updated += 1

            print('Number of updated examples', n_updated)

        gc.collect()
        torch.cuda.empty_cache()

        collate_fn = lambda inpt: tuple(zip(*inpt))

        if (new_indexes is not None) and (self._iter % self._iter_retrain) != 0:
            libact_dataset = Dataset([libact_dataset.data[i][0] for i in new_indexes],
                                     [libact_dataset.data[i][1] for i in new_indexes])
            n_epochs = 1
        else:
            n_epochs = self._retrain_epochs

        if libact_dataset.get_labeled_entries():
            X, y = libact_dataset.format_sklearn()
            X = X.tolist()
            y = y.tolist()
        else:
            X = []
            y = []

        X += self._additional_X
        y += self._additional_y

        if self._string_input:
            X, y = convert_to_bio_format(X, y)

        if not X:
            return

        if self._valid_ratio > 0.:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self._valid_ratio)
            valid_data = list(zip(X_valid, y_valid))
        else:
            X_train, y_train = X, y
            valid_data = None

        train_data = list(zip(X_train, y_train))

        if self._n_upsample_positive:
            n_upsample = self._n_upsample_positive

            positive_examples = [(x, py) for x, py in zip(X_train, y_train) if not all((tag == 'O' for tag in py))]

            if type(n_upsample) is float:
                n_upsample = int(math.ceil(max(0, n_upsample - (len(positive_examples) / len(X_train))) * len(X_train)))

            if n_upsample > 0:
                upsampled_examples = random.choices(positive_examples, k=n_upsample)
                train_data += upsampled_examples

        if self._self_training_samples and self._model is not None:
            unlabeled = libact_dataset.get_unlabeled_entries()
            unlabeled = random.sample(unlabeled, min(self._self_training_samples, len(unlabeled)))

            X = [e[1] for e in unlabeled]

            if self._string_input:
                X = [sent.split(' ') for sent in X]

            pred_y = self._model.predict(X)[0]
            self_training_examples = [(x, py) for x, py in zip(X, pred_y) if not all((tag == 'O' for tag in py))]

            train_data += self_training_examples
        self.train_data_for_allenlp = self.reader.from_list_to_dataset(train_data)
        self.val_data_for_allenlp = self.reader.from_list_to_dataset(valid_data)
        self.train_data_for_allenlp.index_with(self.vocab)
        self.val_data_for_allenlp.index_with(self.vocab)
        self.train_data_loader = DataLoader(dataset=self.train_data_for_allenlp, batch_size=self._batch_size,
                                            collate_fn=allennlp_collate)
        self.val_data_loader = DataLoader(dataset=self.val_data_for_allenlp, batch_size=self._batch_size,
                                          collate_fn=allennlp_collate)
        print('Number of all valid examples: ', len(valid_data))
        print('Number of all training examples: ', len(train_data))

        if (self._model is None) or self._train_from_scratch:
            self._model = self._model_ctor()
            self._trainer = self._trainer_ctor(self._model, len(X_train),
                                               self.train_data_loader, self.val_data_loader)

            gc.collect()
            torch.cuda.empty_cache()

        self._trainer.train()

        self._iter += 1

    def score(self):
        pass
