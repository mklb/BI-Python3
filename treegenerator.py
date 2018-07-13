import pandas as pd
from datapredictionmachine import DataPredictionMachine

min_depth = 1
# without restriction the generated depth is 11
max_depth = 11
# holds all scores from all generated trees
scores = {}

# -----------------------------------------------------
# LOAD TRAIN DATA, CLEAN, GENERATE TREE OF DEPTH i
# -----------------------------------------------------
for i in range(min_depth,max_depth+1):
  test_id = "tree-generator-" + str(i)
  train_data_frame = pd.read_csv('./data/train.csv')
  trainer = DataPredictionMachine(test_id, train_data_frame, False)
  trainer.describe()
  trainer.prepare()
  # trainer.create_describing_images()
  trainer.handle_missing_values()
  trainer.create_dummy_vars()
  trainer.clean()
  # trainer.preview()
  trainer.generate_tree(i)

  scores[test_id] = trainer.calc_score(trainer.get_dataframe())

# print all scores and the highest score
print("\nAll scores:\n")
print(scores)

highscore = max(scores.items(), key = lambda x: x[1])
print("\nHighest score:", highscore)
