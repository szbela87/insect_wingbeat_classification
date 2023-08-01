# Insect Wingbeat Classification by ResNet9

### Data Preparation

You can obtain all the three datasets from [Google Drive](https://drive.google.com/drive/folders/1kt94eoQ4LKunu0DCHxmZfUbXmmrlpdK2?usp=sharing).

```
mkdir data
```
**Please put them in the `./data` directory**

**Also put the `evaluation.py` and `train_mgpu.py` files in the corresponding directory**

### Training Example
In the different directories, we provide the training scripts, i.e. for the *Wingbeats* dataset and the large *ResNet9* model can be found
at `./wingbeats_large`.
To train the model use the command `./train_wingbeats_large.sh` command.
The output files, i.e. the best models are in the `./wingbeats_large/results` directory after the training.
To evaluate the results use the `./evaluation_wingbeats_large.sh` command.

 
