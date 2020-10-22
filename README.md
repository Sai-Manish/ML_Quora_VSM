# AI511 ML Project Working repo
---------------------------------

### Team member
- Sai Manish Sasanapuri (IMT2018520)
- Shubhayu Das (IMT2018523)
- Veerendra S Devaraddi(IMT2018529)

### Folder structure
--------------------
```$ tree -L 2
.
├── code
│   ├── current
│   ├── CustomVectorizer.py
│   ├── Fastmodel_CM.png
│   ├── FastModel.py
│   ├── Logistic.py
│   ├── model_CM.png
│   ├── Model.py
│   ├── Preprocesser.py
│   └── run.sh
├── dataset
│   ├── ideal_40.csv
│   ├── README.md
│   ├── test.csv
│   └── train.csv
├── processed
│   ├── preprocessed_test.csv
│   ├── preprocessed_train.csv
│   └── README.md
└── README.md

5 directories, 15 files
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

### Libraries needed
--------------------
pandas
numpy
scipy
matplotlib
pickle
sklearn
imblearn
contractions

See ```requirements.txt``` for greater details. Or simply run ```pip install -r requirements.txt```.

### General instructions
---------------------------
1. Commit anytime you get a model which gives a higher score of Kaggle. Also copy ```Model.py, Preprocesser.py``` and
   ```Model.sav``` to the ```current``` directory, after renaming the model file.
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
   careful. Generates ```model_CM.png```, ```Vectorizer.sav``` and ```Model.sav```.

   Alternatively, you can run all of steps 2-4 by running ```sh ./run.sh```.

   The FastModel.py is outdated and doesn't yield good performance anymore. The ```Logistic.py``` takes over 40 minutes
   to converge and doesn't give better results.

### TODO:
------------------------

1. Figure out better preprocessing methods.
2. Check for overfitting of the current best model.
