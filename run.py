import pandas as pd
from datapredictionmachine import DataPredictionMachine

# load the train data, preview, transform , clean and generate a tree
train_data_frame = pd.read_csv('train.csv')
trainer = DataPredictionMachine(train_data_frame, "output", True)
trainer.describe()
trainer.prepare()
trainer.handle_missing_values()
trainer.clean()
trainer.preview()
trainer.generate_tree()

# load the test data, transform & clean
test_data_frame = pd.read_csv('test.csv')
tester = DataPredictionMachine(test_data_frame, None, False)
tester.prepare()
tester.handle_missing_values()
tester.clean()

# calc score with train set
score = trainer.calc_score(trainer.get_dataframe())
print("Model score (train data):", score)
# calc score with test set
predict_survival = trainer.predict(tester.get_dataframe())
print("Predicted survival for test data:\n", predict_survival)
