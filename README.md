# <p align=center>AI CUP 2019 Abstract Labeling</p>

### How to run

```
pip install -r requirements.txt
```
Use above command to install required packages

Use the command `python csv2json.py [datapath] [mode]` to preprocess training and testing data
```
python csv2json.py [trainset_path] 'train'
python csv2json.py [public_testset_path] 'test'
python csv2json.py [private_testset_path] 'test'
```

```
bash scripts/train.sh output_dir
bash scripts/train_BACKGROUND.sh output_dir
bash scripts/train_OBJECTIVES.sh output_dir
...
```
Update the `scripts/train.sh` script with the appropriate hyperparameters and datapaths.

