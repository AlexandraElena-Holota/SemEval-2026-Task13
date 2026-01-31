This folder contains an end-to-end classifiation pipeline made to address SemEval13-subtask B. It uses CodeBert and comprehends a notebook for execution and some helper scripts.

            Contents:

Jupiter notebook:
                 -Task B.ipynb used to launch dataprocessing, training, calibration, ensembles and evaluation in sequence.
Helper scripts:     
                 -Calibrate_probs.py applies temperature scaling and Class bias correction
                 -train_codebert_v2.py contains the class used to train the model "bestStrategy"
                 -train_codebert_v2.py contains the class used to train the model "focalModel"
                 -train_codebert_v3.py contains the class used to train the model "m3" and the discarded "m1" and "m2"
jsons:          
                 - id_to_label.json
                 - label_to_id.json
Exported Artifacts:
                 -logits folder included to evaluate quickly the performance of the ensembles without retraining the model and predict
                 -probs folder included to evaluate the probabilities calculated while predicting and using them in ensembles without retraining and prediction

            Quick Start:

Install dependancies: 
                        pip install torch transformers datasets scikit-learn numpy pandas tqdm
add the datasets needed in this folder (training, evaluation and test parquets)
Run the notebook executing cells from top to bottom, if you don't wish to retrain the model you can just evaluate the results using the logits and probs precalculated.

            Training experiments:

train_codebert_v2.py contains a trainer for "bestStrategy". It implements Weighted Cross Entropy loss and tokenize the datasets with max length 512 and no additional preprocessing. 

train codebert_v3.py contains a trainer for m3 model. It implements weighted Cross Entropy loss and tokenize the dataset with max lenght of 256 and preprocessing removing comments, python docstrings and normalizing whitespaces. 
                model m1 and m2 were not included in ensembles due to poor performance and lack of improvements. m1 was trained with cross entropy loss on preprocessed tokenized dataset with max lenght 512. m2 used focal loss, no grad clipping on a 512 max lenght preprocessed tokenized dataset

train_codebert_v2_focal contains a trainer for "focalModel". It implements focal loss and gradient clipping. it operates on the same tokenized dataset used by best strategy. 

All the trainings lasted between 1.5 and 2 epochs. 
The tokenized datasets are made persistent in order to avoid to retokenize the datasets each time.
Logits of predictions on val and test datasets are stored to reproduce the results.

            Calibration:
For each model used, temperature and class bias are calculated and applied on the logits. 
The probabilities for validation and test predictions are then calculated and saved for later use.

            Ensembles:
The idea is to enhance the macro f1 score's performance by calculating a weighted average of the probabilities of different models, each providing a different perspective due to the different strategies on which they were trained.
The most benefits come from m3+bestStrategy with 0.5 and 0.5 weights and m3+bestStrategy+focalModel with weights calculated with grid search in order to maximaze macro f1 in evaluation.
BestStrategy gives most of the performances, m3 focus solely on code without considering comments, focal model focus more on difficult cases providing different kind of insights.

