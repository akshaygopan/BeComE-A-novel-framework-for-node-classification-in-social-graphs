from pipeline import Pipeline
import pandas as pd


results = pd.DataFrame()
data_list = ['cora', 'citeseer']
for data in data_list:
  results = Pipeline.pipeline(data, results)