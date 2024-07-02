from sentence_transformers import SentenceTransformer
from datasets import load_dataset

class Semantic:

    def introduceLabelAwareness(self, text, label):
        return  text + '<SEP>The label is ' + str(label)
    
    def getTextData(self, data, label_col):

        datasets = {'cora': 'Akshayxx/CoraDatasetV6_Two', 'citeseer' :'Akshayxx/citeseerV1'}

        dataset = load_dataset(datasets[data])

        text_df_test = dataset['validation'].to_pandas()
        text_df_train = dataset['train'].to_pandas()
        text_df_train['text'] = text_df_train.apply(lambda x: self.introduceLabelAwareness(x['text'], x[label_col]), axis = 1)

        return text_df_train, text_df_test

    def getTextModel(self, model_name = 'bert-base-nli-mean-tokens'): #Akshayxx/bert-base-cased-finetuned-cora

        model = SentenceTransformer(model_name)
        return model
    
    def getTextEmbeddings(self, text_df_train, text_df_test, text_col = 'text', model_name = 'bert-base-nli-mean-tokens'):

        model = self.getTextModel(model_name)
        text_vecs_train = model.encode(text_df_train[text_col])
        text_vecs_test = model.encode(text_df_test[text_col])

        return text_vecs_train, text_vecs_test
