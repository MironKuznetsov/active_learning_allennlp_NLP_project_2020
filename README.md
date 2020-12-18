# Active learning with deep pre-trained models and Bayesian uncertaintyestimates for biomedical entity recognition
## Statistical NLP course project, Skoltech, Term 2, 2020

### Authors:
```
Miron Kuznetsov, Aleksandr Belov, Dmitry Vypiraylenko
```


### Models
The following models were used for experiments:
- SlavicBERT [1]
- RuBERT [2]
- Sentence RuBERT [3]
- RoBERTa [4]
- DistilBERT [5, 6]


### Results:

| Models        | F1-score     |Recall score  |Precision score|
| ------------- |-------------| -----| ------|
|RuBERT        | 0.45       | 0.41      | 0.51  |
|RoBERTa       | 0.47        | 0.63      | 0.37  |
|Sent RuBERT     | 0.53  |     0.68      | 0.43 |
|SlavicBERT       | 0.69 |         0.78     | 0.62  |
|**DistilBERT**   | **0.87** |**0.95**      | **0.8** |


## References:
1. Mikhail Arkhipov et.al., 2019. Tuning Multilingual Transformers for Language-Specific Named Entity Recognition. Association for Computational Linguistics
2. Yuri Kuratov et. al., 2019. Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language. arXiv:1905.07213
3. Nils Reimers et. al., 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv:1908.10084
4. Yinhan Liu et al., 2019. Roberta: A robustly optimized BERT pretraining approach. arXiv:1907.11692
5. Geoffrey Hinton et. al., 2015. Distilling the Knowledge in a Neural Network. arXiv:1503.02531
6. Victor Sanh et al., 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108
