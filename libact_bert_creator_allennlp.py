from model_wrappers_allennlp import LibActNN

from bert_sequence_tagger.bert_utils import make_bert_tag_dict_from_flair_corpus, prepare_flair_corpus
from bert_sequence_tagger import BertForTokenClassificationCustom, SequenceTaggerBert, ModelTrainerBert
from bert_sequence_tagger.bert_utils import get_parameters_without_decay, get_model_parameters
from bert_sequence_tagger.metrics import f1_entity_level

from pytorch_transformers import BertTokenizer, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models import SimpleTagger
from allennlp.modules.seq2seq_encoders import PassThroughEncoder

from allennlp.training.learning_rate_schedulers import ReduceOnPlateauLearningRateScheduler

from allennlp.data.dataloader import DataLoader
from allennlp.training import GradientDescentTrainer


def prepare_corpus(corpus):
    X, y = [], []
    for X_i, y_i in prepare_flair_corpus(corpus):
        X.append(X_i)
        y.append(y_i)

    return X, y


# class BertTrainerWrapper:
#     def __init__(self, trainer, n_epochs):
#         self._trainer = trainer
#         self._n_epochs = n_epochs
# 
#     def train(self):
#         return self._trainer.train(self._n_epochs)


class LibActBertCreator:
    def __init__(self, tokenizer_name, bert_model_type, #tag2idx, idx2tag,
                 cache_dir, n_epochs, lr, bs, ebs, patience, additional_X, additional_y, vocab, reader, bs_pred):
        self._bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_name,
                                                             cache_dir=cache_dir,
                                                             do_lower_case=('uncased' in tokenizer_name))
        # self._tag2idx = tag2idx
        # self._idx2tag = idx2tag
        self._cache_dir = cache_dir
        self._lr = lr
        self._n_epochs = n_epochs
        self._bs = bs
        self._ebs = ebs
        self._patience = patience
        self._bert_model_type = bert_model_type
        self._additional_X = additional_X
        self._additional_y = additional_y
        self.vocab = vocab
        self.reader = reader
        self.bs_pred = bs_pred

    def __call__(self, **libact_nn_args):
        def model_ctor():
            # model = BertForTokenClassificationCustom.from_pretrained(self._bert_model_type,
            #                                                          cache_dir=self._cache_dir,
            #                                                          num_labels=len(self._tag2idx)).cuda()
            #
            # seq_tagger = SequenceTaggerBert(model, self._bert_tokenizer, idx2tag=self._idx2tag,
            #                                 tag2idx=self._tag2idx, pred_batch_size=self._ebs)


            embedder = PretrainedTransformerMismatchedEmbedder(model_name=self._bert_model_type)
            text_field_embedder = BasicTextFieldEmbedder({'tokens': embedder})

            seq2seq_encoder = PassThroughEncoder(input_dim=embedder.get_output_dim())

            tagger = SimpleTagger(text_field_embedder=text_field_embedder,
                               vocab=self.vocab,
                               encoder=seq2seq_encoder,
                               calculate_span_f1=True,
                               label_encoding='IOB1').cuda()

            return tagger

        def trainer_ctor(tagger, corpus_len, train_dataloader, val_dataloader):
            optimizer = AdamW(tagger.parameters(),
                              lr=self._lr, betas=(0.9, 0.999),
                              eps=1e-6, weight_decay=0.01, correct_bias=True)

            # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=self._patience)
            #
            # trainer = ModelTrainerBert(model=seq_tagger,
            #                            optimizer=optimizer,
            #                            lr_scheduler=lr_scheduler,
            #                            train_dataset=train_data,
            #                            val_dataset=val_data,
            #                            validation_metrics=[f1_entity_level],
            #                            batch_size=self._bs,
            #                            update_scheduler='ee',
            #                            keep_best_model=True,
            #                            restore_bm_on_lr_change=True,
            #                            max_grad_norm=1.,
            #                            smallest_lr=self._lr / 4)

            lr_scheduler = ReduceOnPlateauLearningRateScheduler(optimizer, mode='max', factor=0.5, patience=self._patience)

            trainer = GradientDescentTrainer(
                model=tagger,
                validation_metric='-loss',
                optimizer=optimizer,
                data_loader=train_dataloader,
                validation_data_loader=val_dataloader,
                num_epochs=self._n_epochs,
                # cuda_device=cuda_device,
                learning_rate_scheduler=lr_scheduler,
                patience=self._patience,
                num_gradient_accumulation_steps=self._bs)

            return trainer

        return LibActNN(model_ctor=model_ctor,
                        trainer_ctor=trainer_ctor,
                        additional_X=self._additional_X,
                        additional_y=self._additional_y,
                        reader=self.reader,
                        vocab=self.vocab,
                        bs_pred=self.bs_pred,
                        **libact_nn_args)
