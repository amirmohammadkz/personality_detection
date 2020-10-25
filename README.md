# BB-SVM model for automatic personality detection of the Essays dataset (Big-Five personality labelled traits)
This repository containts Bagging SVM over BERT model for classifying Essays dataset.

## Installation

See the requirements.txt for the list of dependent packages which can be installed via:

```bash
pip -r requirements.txt
```
Specified versions are used in the paper. Note that the updated versions of the requirement modules may change the results. Some experiments verified that the updated sklearn improves the accuracy. However, please also check the [bert-as-service](https://github.com/hanxiao/bert-as-service) requirements (e.g. 1.10<Tensorflow ver<2 is required). The code can be run using Python ver 3.7 . Users' feedback indicated that it cannot be run on Python ver> 3.8

## Usage
1- Run shrink_data.py to convert documents to subdocuments. By running this step, BERT can process the whole sub-documents.

```bash
python shrink_data.py
```

2- Run BERT server on all layers from cmd/terminal (more information [here](https://github.com/hanxiao/bert-as-service))

```bash
bert-serving-start -model_dir uncased_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=NONE -show_tokens_to_client -pooling_layer -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1
```

3- Run process_data_with_sentence_bert.py to extract BERT word embeddings.

```bash
python process_data_with_sentence_bert.py
```

4- Run svm_result_calculator.py to extract the personality traits. (you can change the svm.py code to use Bagging or not)

```bash
python svm_result_calculator.py
```

## Running Time

On an Intel Core i7-4720 HQ CPU, our fine-tuning model only takes about 7 minutes to train.

## Citation

If you use this code in your work then please cite the paper - [Personality Trait Detection Using Bagged SVM over BERT Word Embedding Ensembles](https://sentic.net/personality-detection-using-bagged-svm-over-bert.pdf) with the following:

```
@inproceedings{kazameinipersonality,
  title={Personality Trait Detection Using Bagged SVM over BERT Word Embedding Ensembles},
  author={Kazameini, Amirmohammad and Fatehi, Samin and Mehta, Yash and Eetemadi, Sauleh and Cambria, Erik},
  booktitle={Proceedings of the The Fourth Widening Natural Language Processing Workshop},
  Organization = {Association for Computational Linguistics},
  year={2020}}
```
