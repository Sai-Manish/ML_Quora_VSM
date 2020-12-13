# AI511 ML Project Working repo
---------------------------------

### Team member
- Sai Manish Sasanapuri (IMT2018520)
- Shubhayu Das (IMT2018523)
- Veerendra S Devaraddi(IMT2018529)

### Folder structure
--------------------
```$ tree
.
├── code
│   ├── BernoulliNB.py
│   ├── confusion_matrices
│   │   ├── bernoulli_NB__CM.png
│   │   ├── ensemble_model_CM.png
│   │   ├── linear_SVC_CM.png
│   │   └── sgd_model_CM.png
│   ├── CustomVectorizer.py
│   ├── Ensemble.py
│   ├── LSVC.py
│   ├── models
│   │   ├── bernoulli_NB.sav
│   │   ├── Ensemble.sav
│   │   ├── linear_SVC.sav
│   │   └── SGDModel.sav
│   ├── Preprocesser.py
│   ├── run_all.sh
│   ├── run_model.sh
│   ├── SGD.py
│   ├── submission
│   │   ├── EnsembleSubmissions.csv
│   │   ├── FastSubmissions.csv
│   │   ├── LSVC_submissions.csv
│   │   └── SGD_Submissions.csv
│   ├── TFIDF_words.txt
│   └── Tree.py
├── dataset
│   ├── ideal_40.csv
│   ├── README.md
│   ├── test.csv
│   └── train.csv
├── processed
│   ├── preprocessed_test.csv
│   ├── preprocessed_train.csv
│   └── README.md
├── Quora Insincere Question Report 1.pdf
├── README.md
└── requirements.txt

6 directories, 32 files

```

### Changelog
-------------

17:00 22/10/2020 - by Shubhayu Das
1. Folder structure was changed to keep the processed datasets in a separate folder.
2. ```.gitignore``` was added to prevent pushing large files to Github.
3. Model.py was modified to improve performance and reduce run time.
4. Preprocesser.py was modified to remove some contractions.
5. ```requirements.txt``` was added, for convenience in installing libraries.
6. ```run.sh``` was updated to correct for mistake in folder path testing.

11:45 13/12/2020 - by Shubhayu Das
1. Folder structures changed inside code folder
2. All programs are replaced with our best models.
3. ```run.sh``` was replaced with ```run_all.sh``` which will run preprocessing, along with all our models.
   ```run_model.sh``` to only run training on all the models(except LGBM).

### Libraries needed
--------------------
pandas
numpy
scipy
nltk
matplotlib
pickle
sklearn
imblearn
contractions
lightgbm

See ```requirements.txt``` for greater details. Or simply run ```pip install -r requirements.txt```.

### General instructions
---------------------------
1. Commit anytime you get a model which gives a higher score on Kaggle.
2. **Do not push any file larger than 10mb to Github**.

### Instructions for use
------------------------

1. Download ```train.csv``` and ```test.csv``` from the [contest page](https://www.kaggle.com/c/quora/data) and place in
   the ```dataset``` folder. Further download our secret test dataset from [here](https://www.kaggle.com/shubhayudas6/test-40)
   and rename it to ```ideal_40.csv``` and place in the dataset folder.
2. Delete ```preprocessed_train.csv``` and ```preprocessed_test.csv``` from the ```code``` folder.
3. Run ```Preprocesser.py``` to run all the preprocessing steps on the datasets. This code is parallelized and runs of
   all cores of the processor. Nominal RAM usage is nearly 3GB. Generates ```preprocessed_train.csv``` and ```preprocessed_test.csv```.
4. Run ```Model.py``` to generate the TFIDF matrix and then train the model. It also runs the predictions on the
   submissions dataset. Generating the TFIDF matrix is extremely RAM intensive and can easily take 4-6GB RAM. So be
   careful. Generates ```<model>_CM.png``` and ```<model_name>.sav``` along with the submission csv file in the
   respective folders.

   Alternatively, you can run all of steps 2-4 by running ```sh ./run.sh```.

### Performance of model on ideal_40 test set, with the current Preprocessor.py:
1. BernoulliNB - 0.47007
2. SGD - 0.63403
3. LinearSVC - 0.63445
4. Ensemble of above three - 0.635874
5. LightGBM classifier - ~0.62
 
### TODO:
------------------------

1. Figure out better preprocessing methods.
