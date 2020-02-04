# fasttext-tensorflow

We implemented facebook's fasttext supervised model architecture in tensorflow. There were several motivations for us:
+ Customization of model architecture
+ Experiments with loss functions and learning rate schedules
+ GPU utilization, which is helpful for training with large number of labels

The implementation was done with python3.6 and tensorflow1.12 but is compatible till 1.15. 
We had no intention to fully replicate their implementation. We needed some customization for our internal use and thought to make it open-source to make it also available for the community. Everyone is welcome to make improvement suggestions if they are found.  

We will also rewrite it using libraries of the newest versions in the nearest fututre. The implementation doesn't include character level representations or any subword information. Also, the implementation doesn't make use of the hashing trick which is used in the original one, for storing subwords and word n-grams. We are keeping all the word 
ngrams which appear in the data more than equal than the `min_word_count` parameter. 

There are 2 ways to use the code. 
From terminal:
``` bash
python3 main.py --train_path train.txt --test_path test.txt \
          --num_epochs 10 --learning_rate 0.3 --min_word_count 5 --word_ngrams 2 \
          --batch_size 4000 --use_gpu 1 --gpu_fraction 0.7
```

or from jupyter notebook: 
``` python
from fasttext_model import train_supervised
train_supervised("train.txt", "test.txt", use_gpu=True, gpu_fraction=0.7,
                hyperparams={"num_epochs": 10, "learning_rate": 0.3, "word_ngrams": 2
                             "min_word_count": 5, "batch_size": 4000})
```

The full list of hyperparameters with their explanations can be found `main.py`

We experimented with some customizations on model (Dropout, Weight Reularization, Batch Normalization) and some of them worked quite well for our internal data. Also we have a parameter which controls the learning rate changes, while in fasttext it forcibly goes to 0. 
In our text pre-processing experiments we tried some kind of regularization for ngrams, which was to sort the words of ngram alphabetically (ex. if you are using bigrams and "dell notebook" and "notebook dell" both appear in the corpus they will be stored under "dell notebook"). This might be helpful depending on the task and the word corpus.   

