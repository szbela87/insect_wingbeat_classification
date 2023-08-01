# ResNet-9 Models for Insect Wingbeat Sound Classification

### Data Preparation

You can obtain all the three datasets from [Google Drive](https://drive.google.com/drive/folders/1kt94eoQ4LKunu0DCHxmZfUbXmmrlpdK2?usp=sharing).

```
mkdir data
```
**Please put them in the `./data` directory**

**Also put the `evaluation.py` and `train_mgpu.py` files in the corresponding directory**

### Training Example
In the different directories, we provide the training scripts, i.e. for the *Wingbeats* dataset and the large *ResNet9* model can be found
at `./wingbeats_large`. Each training script produces 5 training sessions.

To train the model use the command `./train_wingbeats_large.sh` command. 

The output files, i.e. the best models and logs are in the `./wingbeats_large/results` directory after the training, these are:
* `best_model_{i}.pt` - the model with best validation accuracy from the i-th training,
* `inrun_results_{i}.csv` - the results during the training by the i-th training (just for logging),
* `train_results_{i}.csv` - the training results corresponding to the i-th training process,
* `valid_results_{i}.csv` - the validation results corresponding to the i-th training process,
Here, $i=0...4$

To evaluate these results, and evaluate these `best_model_{i}`, $i=0...4$ models on the test set use the `./evaluation_wingbeats_large.sh` command. It requires the files
`best_model_{i}.pt`, `inrun_results_{i}.csv`, `train_results_{i}.csv`, `valid_results_{i}.csv` for $i=0...4$ from the `results` directory and
creates the `results.dat` file which contains evaluation metrics achieved by the model with the highest validation accuracy. 
It also creates the confusion matrix corresponding the model with the highest validation accuracy,
and makes plots about the accuracies and the losses in `.svg` format among the 5 independent runs. 




 