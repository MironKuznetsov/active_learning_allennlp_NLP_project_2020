import os
import copy
import logging
import warnings
import itertools
import collections
from pathlib import Path
from IPython.core.display import display, HTML
from typing import Dict, List, Sequence, Iterable

warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from libact.query_strategies import UncertaintySampling
# from libact.base.dataset import Dataset

from allennlp.data.dataset_readers import Conll2003DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.data.vocabulary import Vocabulary

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, AllennlpDataset
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from actleto import ActiveLearner, ActiveLearnerUiWidget, make_libact_strategy_ctor
from actleto.annotator.annotator_widget import AnnotatorWidget
from actleto.annotator.visualizers.seq_annotation import SeqAnnotationVisualizer

from flair.datasets import ColumnCorpus
from bert_sequence_tagger.bert_utils import make_bert_tag_dict_from_flair_corpus

from libact_bert_creator_allennlp import LibActBertCreator
from libact_bert_creator_allennlp import prepare_corpus

from bert_sequence_tagger.bert_utils import prepare_flair_corpus
from bert_sequence_tagger.metrics import f1_entity_level, f1_token_level

from seqeval.metrics import f1_score, precision_score, recall_score

from isanlp.ru.processor_tokenizer_ru import ProcessorTokenizerRu
from isanlp.processor_sentence_splitter import ProcessorSentenceSplitter
from isanlp.annotation_repr import CSentence

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from model_wrappers_allennlp import CustomSentenceTaggerPredictor
from utils_data import create_helper, convert_y_to_dict_format
from strategy_mnlp import StrategyMNLP


logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


class GeniaDatasetReader(Conll2003DatasetReader):
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, ner_tags = fields
                    # TextField requires `Token` objects
                    tokens = [Token(token) for token in tokens_]

                    yield self.text_to_instance(tokens, ner_tags=ner_tags)

    def _from_list_to_instance(self, data) -> Iterable[Instance]:
        list_of_instances = []
        for text_tags in data:
            list_of_instances.append(self.text_to_instance([Token(token) for token in text_tags[0]], ner_tags=text_tags[1]))
        return list_of_instances
    def from_list_to_dataset(self, data):
        return AllennlpDataset(self._from_list_to_instance(data))


def get_seq_tagger(active_learner):
    return active_learner._active_learn_algorithm._libact_query_alg.impl.model._model


def insert_tags(text, tags):
    html = ''
    end = 0
    for tag in tags:
        html += text[end:tag['offset']] + tag['tag']
        end = tag['offset']

    html += text[end:]
    return html


def format_entities(sentence, tags):
    annots = []
    curr_tag = {}

    for i, tag in enumerate(tags):
        real_tag = tag.split('-')

        if real_tag[0] == 'B':
            if curr_tag:
                curr_tag['end'] = sentence[i - 1].end
                annots.append(curr_tag)

            curr_tag['tag'] = real_tag[1]
            curr_tag['begin'] = sentence[i].begin

        elif real_tag[0] == 'O':
            if curr_tag:
                curr_tag['end'] = sentence[i - 1].end
                annots.append(curr_tag)
                curr_tag = {}

    return annots


class ALSystem:
    def __init__(self, config, save_path):
        self.config = config
        self.evaluation_results = []
        self.save_path = save_path
        self._additional_X = []
        self._additional_y = []

    def _get_libact_nn(self):
        return self.active_learner._active_learn_algorithm._libact_query_alg.impl.model

    def load_dataset(self, data_folder):
        corpus = ColumnCorpus(data_folder, {0: 'text', 1: 'ner'},
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')  # We do not need dev set

        indexer = PretrainedTransformerMismatchedIndexer(model_name=self.config.BERT_MODEL_TYPE)
        self.reader = GeniaDatasetReader(token_indexers={'tokens': indexer}, tag_label='ner')
        self.train_dataset_for_voc = self.reader.read(data_folder + '/train.txt')

        # Creating tag dictionaries
        self.idx2tag, self.tag2idx = make_bert_tag_dict_from_flair_corpus(corpus)
        self.tags = list(set((tag.split('-')[1] for tag in self.idx2tag if len(tag.split('-')) > 1)))
        print('Tags:', self.tags)

        # Convert into the format suitable for training
        self.X_train, self.y_train = prepare_corpus(corpus.train)
        
        self.X_test, self.y_test = prepare_corpus(corpus.test)
#         print(self.X_test)
#         print('------------------')
#         print(self.y_test)
#         sent = CSentence(tokens, sentences[i])
#         entities = format_entities(sent, result[i])
#         tokenizer = ProcessorTokenizerRu()
#         splitter = ProcessorSentenceSplitter()
#         for x, y in zip(self.X_test, self.y_test):
#             print(x, y)
#             abc = format_entities(x, y)
#             if abc:
#                 print(abc, '------------')
            
        # Convert into the format suitable for visualization
        # self.y_train_dict = convert_y_to_dict_format(self.X_train, self.y_train)
        self.X_helper = create_helper(self.X_train)
        print("self.X_train[:1]:")
        print(self.X_train[:1])
        print("self.X_helper.head():")
        print(self.X_helper.head())

        self.y_seed_dict = [None for _ in range(len(self.X_helper))]
    
    def load_train_dataset(self, data_folder, volume):
        X = self.X_train[:round(volume*len(self.X_train))]
        y = self.y_train[:round(volume*len(self.X_train))]
        dict_format = convert_y_to_dict_format(X, y)
#             sent = CSentence(x, self.X_train[i])
#             entities = format_entities(sent, self.y_train[i])
#             entities_list.append(entities)
        self.y_seed_dict = dict_format
        
    def load_annotations(self, annotation_path):
        self.y_seed_dict = np.load(os.path.join(annotation_path, 'annotation.npy'), allow_pickle=True).tolist()
#         for i in self.y_seed_dict:
#             if i!=['None'] and i is not None:
#                 print(i)
            
        custom_X_path = Path(annotation_path) / 'custom_X.npy'
        custom_y_path = Path(annotation_path) / 'custom_y.npy'

        if os.path.exists(custom_X_path) and (os.path.exists(custom_y_path)):
            self._additional_X = np.load(custom_X_path, allow_pickle=True).tolist()
            self._additional_y = np.load(custom_y_path, allow_pickle=True).tolist()

    def parse(self, text):
        tagger = get_seq_tagger(self.active_learner)
        tokenizer = ProcessorTokenizerRu()
        splitter = ProcessorSentenceSplitter()

        tokens = tokenizer(text)
        sentences = splitter(tokens)

        sent_toks = [[word.text for word in CSentence(tokens, sent)] for sent in sentences]
        predictor = CustomSentenceTaggerPredictor(tagger, self.reader, self.config.PRED_BATCH_SIZE)

        preds_full = predictor.predict(sent_toks)

        result = preds_full['tags'][:len(sent_toks)]
        print(result)
#         result = tagger.predict(sent_toks)[0]
        # print(result)

        html_tags = []
        for i in range(len(result)):
            sent = CSentence(tokens, sentences[i])
            entities = format_entities(sent, result[i])
            print(entities)
            for ent in entities:
                html_tags.append({'tag': '<span style="background-color: #ff9999">',
                                  'offset': ent['begin']})
                html_tags.append({'tag': '</span>',
                                  'offset': ent['end']})

        html = insert_tags(text, html_tags)

        return HTML(html)

    def create_active_learner(self):
        torch.manual_seed(self.config.RANDOM_STATE)

        vocab = Vocabulary.from_instances(self.train_dataset_for_voc.instances)

        bert_creator = LibActBertCreator(
                                         # idx2tag=self.idx2tag,
                                         # tag2idx=self.tag2idx,
                                         tokenizer_name=self.config.BERT_MODEL_TYPE,
                                         bert_model_type=self.config.BERT_MODEL_TYPE,
                                         cache_dir=self.config.CACHE_DIR,
                                         n_epochs=self.config.N_EPOCHS,
                                         lr=self.config.LEARNING_RATE,
                                         bs=self.config.BATCH_SIZE,
                                         ebs=self.config.PRED_BATCH_SIZE,
                                         patience=self.config.PATIENCE,
                                         additional_X=self._additional_X,
                                         additional_y=self._additional_y,
                                         vocab=vocab,
                                         reader=self.reader,
                                         bs_pred=self.config.PRED_BATCH_SIZE)

        active_learn_alg_ctor = make_libact_strategy_ctor(lambda trn_ds: StrategyMNLP(dataset=trn_ds,
                                                                                             model=bert_creator(
                                                                                                 valid_ratio=self.config.VALIDATION_RATIO,
                                                                                                 retrain_epochs=self.config.N_EPOCHS,
                                                                                                 autofill_similar_objects=True,
                                                                                                 n_upsample_positive=self.config.UPSAMPLE_POSITIVE)
                                                                                             ),
                                                          max_samples_number=self.config.MAX_SAMPLES_NUMBER)

        if all([e is None for e in self.y_seed_dict]):
            rnd_start_steps = 1
        else:
            rnd_start_steps = 0
        # Creating ActiveLearning object that implements AL logic.
        self.active_learner = ActiveLearner(active_learn_alg_ctor=active_learn_alg_ctor,
                                            X_full_dataset=self.X_helper.texts.tolist(),
                                            y_full_dataset=self.y_seed_dict,
                                            rnd_start_steps=rnd_start_steps)

        self.active_learner.start()

    def create_active_learning_widget(self):
        self.stop_ui()

        self.al_widget = ActiveLearnerUiWidget(active_learner=self.active_learner,
                                               X_helper=self.X_helper,
                                               display_feature_table=False,
                                               drop_labels=[],
                                               y_labels=None,
                                               save_path=os.path.join(self.save_path, 'annotation.npy'),
                                               save_time=120,
                                               visualizer=SeqAnnotationVisualizer(tags=self.tags))
        return self.al_widget

    def evaluate(self):
        seq_tagger = get_seq_tagger(self.active_learner)
#         print(seq_tagger)
#         print(type(seq_tagger))
        
        predictor = CustomSentenceTaggerPredictor(seq_tagger, self.reader, self.config.PRED_BATCH_SIZE)
#         print(self.X_test[0])
        
        preds_full = predictor.predict(self.X_test)
        preds = preds_full['tags']
#         print("=========================")
#         print("=========================")
#         print(preds_full['tags'][0], preds_full['words'][0], preds_full['logits'][0], preds_full['class_probabilities'][0])
#         print("=========================")
        
#         print("=========================")
#         preds = seq_tagger.predict(self.X_test)[0]
        #         print(self.X_test)
        #         print("==========")
#         print(preds[0])
#         print(type(preds))
#         print("=========================")
#         print("=========================")
#         print("=========================")
#         print("=========================")
#         print(type(self.y_test))
#         print(self.y_test[0])
#         try:
        f1 = f1_score(self.y_test, preds)
#         except:
#             print(self.y_test)
#             print('--------------------------------------------------------------------------')
#             print(preds)
        prec = precision_score(self.y_test, preds)
        rec = recall_score(self.y_test, preds)

        print(f'F1_score: {f1} , precision: {prec} , recall: {rec}')

        self.evaluation_results.append(f1)

        print('Evaluation results for multiple iterations:')
        print(self.evaluation_results)

        array_of_y_test = []
        array_of_preds = []
        array_of_x = []
        number_of_correct = 0

        for i in range(len(self.y_test)):
            if self.y_test[i] == preds[i]:
                number_of_correct += 1
            else:
                array_of_preds.append(preds[i])
                array_of_y_test.append(self.y_test[i])
                array_of_x.append(self.X_test[i])

        diagnosis = 'vd'
        name_of_file = 'wrong_answers/' + diagnosis + '_epoch_wrong_answers.txt'
        sample = open(name_of_file, 'w')
        print("correct answers:", number_of_correct, file=sample)
        print("wrong answers:", len(array_of_preds), file=sample)

        print(f'F1_score: {f1} , precision: {prec} , recall: {rec}', file=sample)
        print('--------\nlike a classification problem', file=sample)
        #         print(f'F1_score: {f1_s} , precision: {prec_s} , recall: {rec_s}', file = sample)

        print(file=sample)

        for i_of_error in range(len(array_of_y_test)):
            a_set = set(array_of_y_test[i_of_error])
            b_set = set(array_of_preds[i_of_error])
            if ('B-' + diagnosis in a_set or 'I-' + diagnosis in a_set) and (
                    'B-' + diagnosis in b_set or 'I-' + diagnosis in b_set):
                flag_of_class = '+++'
            else:
                flag_of_class = '---'

            print('Text   ||Actual||Prediction', flag_of_class, file=sample)
            for i in range(len(array_of_y_test[i_of_error])):
                if array_of_y_test[i_of_error][i] != array_of_preds[i_of_error][i]:
                    print(array_of_x[i_of_error][i], "  ", array_of_y_test[i_of_error][i], "   ",
                          array_of_preds[i_of_error][i], file=sample)

            print("actual:", array_of_y_test[i_of_error], file=sample)
            print("pred:  ", array_of_preds[i_of_error], file=sample)
            print(array_of_x[i_of_error], file=sample)
            print(file=sample)

    def add_custom_examples(self):
        all_custom_examples = []
        for val, rep in self.custom_examples:
            for _ in range(rep):
                all_custom_examples.append(copy.deepcopy(val))

        print("type of custom_examples:", type(self.custom_examples))
        print("custom_examples:", self.custom_examples)
        all_answers = []

        for answer, rep in zip(self.custom_annotation_widget.get_answers().tolist(),
                               [e[1] for e in self.custom_examples]):
            print("answer:", answer)
            print("rep:", rep)
            for _ in range(rep):
                all_answers.append(copy.deepcopy(answer))

#         print("type answer:", type(answer), "type rep:", type(rep))
#         print("all_custom_examples", all_custom_examples)
#         print("all_answers", all_answers)
        self.active_learner._active_learn_algorithm._libact_query_alg.impl.model._additional_X += all_custom_examples
        self.active_learner._active_learn_algorithm._libact_query_alg.impl.model._additional_y += all_answers

        np.save(Path(self.save_path) / 'custom_X.npy', self._get_libact_nn()._additional_X, allow_pickle=True)
        np.save(Path(self.save_path) / 'custom_y.npy', self._get_libact_nn()._additional_y, allow_pickle=True)

    #         self.active_learner._X_full_dataset += all_custom_examples
    #         self.active_learner._y_full_dataset += all_answers

    #         new_dataset = Dataset(self.active_learner._X_full_dataset, self.active_learner._y_full_dataset)
    #         new_dataset._update_callback = set()

    #         self.active_learner._active_learn_algorithm._train_dataset = new_dataset
    #         self.active_learner._active_learn_algorithm._libact_query_alg.impl._dataset = new_dataset

    def create_custom_annotator_widget(self, custom_examples):
        self.custom_examples = custom_examples
        self.custom_annotation_widget = AnnotatorWidget(pd.DataFrame([e[0] for e in custom_examples],
                                                                     columns=['text']),
                                                        answers=None,
                                                        visualize_columns=[],
                                                        drop_labels=[],
                                                        visualizer=SeqAnnotationVisualizer(tags=self.tags),
                                                        display_feature_table=False,
                                                        y_labels=None)

        return self.custom_annotation_widget

    def stop_ui(self):
        try:
            if active_learn_ui:
                self.active_learn_ui.stop()
        except NameError:
            pass

    def print_test(self):
        for x, y in zip(self.X_test, self.y_test):
            if not all([e == 'O' for e in y]):
                print(list(zip(x, y)))
