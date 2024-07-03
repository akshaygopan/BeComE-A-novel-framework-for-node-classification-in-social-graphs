# BeComE
In this study, we explore the important role of graph embedding methods in extracting valuable insights from graph structures, specifically focusing on node classification tasks. It is important to know the structural and semantic features and connections within nodes in a graph to learn more detailed hidden patterns. Hence, we propose a hybrid architecture, BeComE (Bert-ComplEx Embedding Model), a novel framework that employs both semantic and structural features from social network structures extracted through label - aware embedding models  to aid in node classification in social graphs. A Support Vector Machine(SVM) classifier receives these vector embeddings as input features for classification tasks on social graphs and networks. The evaluation shows that BeComE gives state-of-the-art results on the 'Cora' and 'CiteSeer' data sets.

# Installation
To  clone this repository:
```
git clone https://github.com/akshaygopan/BeComE-A-novel-framework-for-node-classification-in-social-graphs.git
```

# To setup the environment:
To  install the required libraries, run this:
```
pip install -r requirements.txt
```
# Quickstart
The following script can also found in usage.ipynb. A sample run on google colab can be found in colab_run.ipynb.
```
from pipeline import Pipeline
import pandas as pd

results = pd.DataFrame()
data_list = ['cora', 'citeseer']
for data in data_list:
  results = Pipeline.pipeline(data, results)

```
Arguments to the Pipeline.pipeline function:
1. data: Available datasets are ['cora', 'citeseer']. The input datatype is a list.

Architecture:
![Model Architecture](https://github.com/akshaygopan/BeComE-A-novel-framework-for-node-classification-in-social-graphs/blob/main/architecture.png?raw=true)
   
