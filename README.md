# 2019-08-22-1143 MLOps - DevOps for Machine Learning with Maquette, DVC and MLFlow

§ml §datascience §ai $devops §data §dataplatform

---

# Links

* https://mlops.org/


# Motivation

* 60% if AI projects are never implemented (https://gritdaily.com/why-60-percent-of-machine-learning-projects-are-never-implemented/)

# Components required for MLOps

* **Data Catalog.** A data catalog helps data scientists to discover the right data for their purposes. To allow In an enterprise environment the catalog should also...

    * Search & Find Data
    * Reuse existing/ prepared data
    * Access Control
    * Simple Data Collection (for Providers and Data Scientists)

    Possible tools:

    * ckan
    * dkan
    * Apache Atlas
    * own implementation

* **Data Repository.** Data and transformed data which is used in an ML project needs to be versioned (and cached) with the code. As well as the way which describes the way/ transofrmation from a data soruce (data repository) to the transofrmed data or model which is stored in the project.

    Possible tools:

    * git-lfs (Git Large File System)
    * DVC

* **Code Repository.** A code repository keeps track of the code and is the glue between data repository and experiment tracking. Source of truth. 

    * Git

* **Experiments.** When creating a model a Data Scientist usually tries different approaches to create the model. These experiments need to be tracked including code, runtime and resultung metrics of the experiment.

    * DVC
    * MLFlow

# Example

Based on Kaggle [Titanic Competion](https://www.kaggle.com/c/titanic/data).

## Requirements

* Kaggle CLI/ API to download dataset

## Steps

```bash
$ mkdir mlops-titanic
$ cd mlopstitanic

$ echo "# MLOps example with the Titanic dataset" > README.md
$ git init
$ git add
$ git commit -am "Initial commit."

$ dvc init
$ dvc remote add -d fs /Users/michael/Downloads/dvc-remote
$ mkdir data
$ mkdir dvc
$ cd data

$ dvc run -f ../dvc/01-fetch-data.dvc -o gender_submission.csv -o test.csv -o train.csv kaggle competitions download -c titanic
$ git add ../dvc/01-fetch-data.dvc .gitignore
$ cd ..
$ git commit -am "Fetched data."

$ mkdir code
$ cd code
$ # Fetch the prepare.py code from anywhere ....
$ dvc run -d ../data/train.csv -d ../data/test.csv -o ../data/train_prepared.pkl -o ../data/test_prepared.pkl -f ../dvc/02-prepare-data.dvc python prepare.py
```

For the first try we do quite simple training `./code/train.py`:

```python
import pandas as pd
import numpy as np
import json

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# Result Dictionary
results = {}

# Split into x and y
train = pd.read_pickle('../data/train_prepared.pkl')
x_test = pd.read_pickle('../data/test_prepared.pkl')
x_train = train.drop(['Survived'], axis = 1)
y_train = train['Survived']

# Logistic Regression
logreg = LogisticRegression(solver='lbfgs', max_iter = 1000)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
results['Logistic Regression (ACC)'] = acc_log

# Support Vector Machines
svc = SVC(gamma='auto')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
results['Support Vector Machines (ACC)'] = acc_svc

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
results['Decission Tree (ACC)'] = acc_decision_tree

with open('../data/metrics.json', 'w') as fp:
    json.dump(results, fp, indent=2, sort_keys=True)
```

No we can run the training program and execute the training.

```bash
$ dvc run -d train.py -d ../data/train_prepared.pkl -d ../data/test_prepared.pkl -m ../data/metrics.json -f ../Dvcfile python train.py
$ cd ..
$ git add .
$ git commit -am "Added initial trainings script."
```

We also tag the first results of our model to know our baseline:

```bash
$ git tag baseline
```

This result can now be easily reproduced, even if we delete the data directory:

```bash
$ dvc repro
```

We can also share our results with others

```bash
$ git remote add origin git@github.com:cokeSchlumpf/mlops-titanic.git
$ git push -u origin master
```

In another directory you could reproduce the result with ```dvc repro```.

Now we can also try to another approach in a new branch: 

```bash
$ git checkout -b tensorflow
```

... Replace `train.py` with a new training program which uses tensorflow.

```python
import pandas as pd
import numpy as np
import json
import tensorflow as tf

from collections import namedtuple
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y

def build_neural_network(train_x, hidden_units=10):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training=tf.Variable(True,dtype=tf.bool)
    
    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
    fc=tf.layers.batch_normalization(fc, training=is_training)
    fc=tf.nn.relu(fc)
    
    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

def get_batch(data_x,data_y,batch_size=32):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        
        yield batch_x,batch_y

# Result Dictionary
results = {
    'NN (ACC)': 0
}

# Split into x and y
train = pd.read_pickle('../data/train_prepared.pkl')
x_test = pd.read_pickle('../data/test_prepared.pkl')

x_train, y_train, x_valid, y_valid = split_valid_test_data(train)
model = build_neural_network(x_train)

epochs = 200
train_collect = 50
train_print=train_collect*2

learning_rate_value = 0.001
batch_size=16

x_collect = []
train_loss_collect = []
train_acc_collect = []
valid_loss_collect = []
valid_acc_collect = []

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration=0
    for e in range(epochs):
        for batch_x,batch_y in get_batch(x_train,y_train,batch_size):
            iteration+=1
            feed = {model.inputs: x_train,
                    model.labels: y_train,
                    model.learning_rate: learning_rate_value,
                    model.is_training:True
                   }

            train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
            
            if iteration % train_collect == 0:
                x_collect.append(e)
                train_loss_collect.append(train_loss)
                train_acc_collect.append(train_acc)

                if iteration % train_print==0:
                     print("Epoch: {}/{}".format(e + 1, epochs),
                      "Train Loss: {:.4f}".format(train_loss),
                      "Train Acc: {:.4f}".format(train_acc))
                        
                feed = {model.inputs: x_valid,
                        model.labels: y_valid,
                        model.is_training:False
                       }
                val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                valid_loss_collect.append(val_loss)
                valid_acc_collect.append(val_acc)
                
                if iteration % train_print==0:
                    print("Epoch: {}/{}".format(e + 1, epochs),
                      "Validation Loss: {:.4f}".format(val_loss),
                      "Validation Acc: {:.4f}".format(val_acc))

                    if results['NN (ACC)'] < val_acc:
                        results['NN (ACC)'] = val_acc
                

# saver.save(sess, "../data/titanic.ckpt")

results = {
    'NN (ACC)': str(results['NN (ACC)'])
}

with open('../data/metrics.json', 'w') as fp:
    json.dump(results, fp, indent=2, sort_keys=True)
```

Now we can use DVC to train the model with Tensorflow and we can compare metrics with other tries:

```
$ dvc repro
$ dvc metrics show 
$ dvc metrics show -a -T
```



## Notes

* In the example, the starting point is a curl command. This command is always executed as DVC doesn't know that the result will be the same!
* Stages in Pipeline can only be linked via files, not linked via dvc-files...

## Open Questions

* Does DVC support directories?