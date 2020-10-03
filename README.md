# BB-SVM model for automatic personality detection of the Essays dataset (Big-Five personality labelled traits)
This repository containts Bagging SVM over BERT model for classifying Essays dataset.

## Installation

See the requirements.txt for the list of dependent packages which can be installed via:

```bash
pip -r requirements.txt
```
Specified versions are used in the paper. Note that the updated versions of the requirement modules may change the results. Some experiments verified that the updated sklearn improves the accuracy.

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
