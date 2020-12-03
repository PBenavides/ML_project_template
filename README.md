# Machine Learning Project Template

Here is the way how I arrange my ML projects. This will be in constant development since the requirements and good practices that i'll see. As an example I'm using Titanic dataset, but this would change throught time. 


## Execute
You can execute the ML-Project with src/main.py specifying the model in src/model_dispatcher.py
```
#In model_dispatcher.py:

models = {"decision_tree_gini": DecisionTreeClassifier(
    criterion="gini"
    ),
    "decision_tree_entropy" : DecisionTreeClassifier(
        criterion="entropy"
    )
        }
```

Then you can execute from main.py

```
python project-template/src/main.py --fold 0 -- model decision_tree_gini
```

Or also, you can append all the criterias on run.sh

## Usage
This is a Template. All the Preprocessing was made as practice as possible.

---