# Install packages (via your terminal)
**Note:** We use Python 3

```
pip3 install pandas
pip3 install missingno
pip3 install numpy
pip3 install decision-tree-id3
pip3 install IPython 
```

# Depentencies

- a local graphviz version. Mac: `brew install graphviz`. Windows: https://graphviz.gitlab.io/_pages/Download/Download_windows.html

# Run the scripts
**Note:** All output generated is saved in the `./output` folder

```
# run the main script which loads & cleans the train data and predicts the survival for the test data
python3 run.py

# run the tree generator which generates trees of different depths and scores them
python3 treegenerator.py
```

treegenerator.py sample output. "tree-generator-X" indicating the tree max depth of X:
```
All scores:

{'tree-generator-1': 0.7867564534231201, 'tree-generator-2': 0.7934904601571269, 'tree-generator-3': 0.835016835016835, 'tree-generator-4': 0.8406285072951739, 'tree-generator-5': 0.8451178451178452, 'tree-generator-6': 0.8484848484848485, 'tree-generator-7': 0.8496071829405163, 'tree-generator-8': 0.8540965207631874, 'tree-generator-9': 0.8552188552188552, 'tree-generator-10': 0.8552188552188552, 'tree-generator-11': 0.8552188552188552}

Highest score: ('tree-generator-9', 0.8552188552188552)
```

# TODOS

- [ ] Image generation for description https://community.modeanalytics.com/python/tutorial/python-histograms-boxplots-and-distributions/
- [ ] Master -> Child unter 12 years .. not a slaves
- [ ] Intense commenting
- [ ] Missing values 
- [x] Max tree depth
- [X] Decision tree calculation (ID3)
- [x] Model calculation testing (score)

## Notes

Skylearn shrows an error like `sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.` This is a dependency issue and can be safely ignored..[see here](https://stackoverflow.com/questions/48687375/deprecation-error-in-sklearn-about-empty-array-without-any-empty-array-in-my-cod)