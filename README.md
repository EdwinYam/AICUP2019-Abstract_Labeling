# <p align=center>AI CUP 2019 Abstract Labeling</p>

We share the [report](https://drive.google.com/drive/folders/1OT7nlSMP1-E_FJkDYe6V4xI0KYhyEJdt) that displays the details of our training process.

### How to run

```
pip install -r requirements.txt
(sudo) python -m spacy download en_core_web_sm
```
Use above command to install required packages

```
python csv2json.py [trainset_path] 'train'
python csv2json.py [public_testset_path] 'test'
python csv2json.py [private_testset_path] 'test'
```
Use the command `python csv2json.py [datapath] [mode]` to preprocess training and testing data

```
bash scripts/train.sh output/all
bash scripts/train_BACKGROUND.sh output/background
bash scripts/train_OBJECTIVES.sh output/objectives
...
```
Update the `scripts/train.sh` script with the appropriate datapaths.

```
mkdir preds
bash scripts/predict_public.sh
bash scripts/predict_private.sh
```
Use above command to predict labels, then use `python submission.py` to get the final prediction file `final_submission.csv`

