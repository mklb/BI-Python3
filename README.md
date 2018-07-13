# Titanic ID3 tree survival prediction in Python
This repository is made for the course: Business Intelligence 2018 @ Freie Universität Berlin

## Install packages (via your terminal)
**Note:** We use Python 3

```
pip3 install pandas
pip3 install missingno
pip3 install numpy
pip3 install decision-tree-id3
pip3 install IPython 
```

## Depentencies

A local graphviz version:

- Mac: `brew install graphviz`.
- Windows: https://graphviz.gitlab.io/_pages/Download/Download_windows.html

## Run the scripts
**Note:** All output generated is saved in the `./output` folder

```
# run the main script which loads & cleans the train data and predicts the survival for the test data (tree depth 8)
python3 run.py

# run the tree generator which generates trees of different depths and scores them (tree depth 1-11)
# /output folder names: "tree-generator-X". X indicates the tree max depth and id
python3 treegenerator.py

# run the nested holdout testing script (3 trees)
# /output folder names: "overfitting-X". X indicates the id of the tree
python3 treefitter.py
```

run.py samlpe output prediction for the test.csv:
```
[0 0 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 0 1 0 1 1 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 1 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 1 1 0 1 0 1 0 0 1 0 1 1 1 0 0 0 0 1 0 1 1 1 0 1 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 1 0 1 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 1 0 0 1 0 0 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 1 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 1 1 1 0 1 0 1 0 1 1 0 1 0 0 0 1 0 0 0 0 1 0 1 1 1 1 0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 1 1 1 0 0 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 1 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 0 1 0 0 0 0 0 1 1 1 1 1 0 1 0 0 1]
```

treegenerator.py sample output:
```
All scores:

{'tree-generator-1': 0.7867564534231201, 'tree-generator-2': 0.7867564534231201, 'tree-generator-3': 0.813692480359147, 'tree-generator-4': 0.8507295173961841, 'tree-generator-5': 0.8597081930415263, 'tree-generator-6': 0.8664421997755332, 'tree-generator-7': 0.8686868686868687, 'tree-generator-8': 0.8698092031425365, 'tree-generator-9': 0.8709315375982043, 'tree-generator-10': 0.8754208754208754, 'tree-generator-11': 0.8754208754208754}

Highest score: ('tree-generator-10', 0.8754208754208754)
```

treefitter.py sample output:
```
Model scores:
Model 1: 0.8417508417508418
Model 2: 0.8249158249158249
Model 3: 0.7744107744107744
```

## Notes

Skylearn shrows an error like `sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.` This is a dependency issue and can be safely ignored..[see here](https://stackoverflow.com/questions/48687375/deprecation-error-in-sklearn-about-empty-array-without-any-empty-array-in-my-cod)