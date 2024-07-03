from utils.evaluate import Evaluate
from utils.semantic import Semantic
from utils.structural import Structural
from utils.concat import Concat

class Pipeline:

    def pipeline(data, results):

        label_col = {'cora' : 'label', 'citeseer': 'encoded_labels'}
        sm = Semantic()
        text_df_train, text_df_test = sm.getTextData(data, label_col = label_col[data])
        text_vecs_train, text_vecs_test = sm.getTextEmbeddings(text_df_train, text_df_test, text_col = 'text', model_name = 'bert-base-nli-mean-tokens')
        
        st = Structural()
        
        if(data =='cora'):
            edges = st.getCoraEdges()
        else:
            edges = st.getCiteSeerEdges()

        
        node_embeddings_train, node_embeddings_test = st.getStructuralEmbeddings(text_df_train, text_df_test, edges, label_col = label_col[data])
        labels_train, labels_test = Concat.getLabels(text_df_train, text_df_test, col = label_col[data])
        X_train, y_train, X_test, y_test = Concat.getCombinedEmbeddings(text_vecs_train, node_embeddings_train, labels_train, text_vecs_test, node_embeddings_test, labels_test)
        preds, labels = Evaluate.trainAndPredict(X_train, X_test, y_train, y_test)
        name = data + 'BERT + ComplEx + SVM'
        results = Evaluate.getResults(preds, labels, len(y_train.unique()), results, name)

        return results
    
    def pipeline_only_structural(data, results):
 
        label_col = {'cora' : 'label', 'citeseer': 'encoded_labels'}
        sm = Semantic()
        text_df_train, text_df_test = sm.getTextData(data, label_col = label_col[data])
        #text_vecs_train, text_vecs_test = sm.getTextEmbeddings(text_df_train, text_df_test, text_col = 'text', model_name = 'bert-base-nli-mean-tokens')
       
        st = Structural()
       
        if(data =='cora'):
            edges = st.getCoraEdges()
        else:
            edges = st.getCiteSeerEdges()
 
       
        node_embeddings_train, node_embeddings_test = st.getStructuralEmbeddings_noLabels(text_df_train, text_df_test, edges, label_col = label_col[data])
        labels_train, labels_test = Concat.getLabels(text_df_train, text_df_test, col = label_col[data])
        X_train, y_train, X_test, y_test = Concat.getCombinedEmbeddings(node_embeddings_train, labels_train, node_embeddings_test, labels_test)
        preds, labels = Evaluate.trainAndPredict(X_train, X_test, y_train, y_test)
        name = data + 'ComplEx + SVM'
        results = Evaluate.getResults(preds, labels, len(y_train.unique()), results, name)
 
        return results
 
    def pipeline_only_semantic(data, results):
 
        label_col = {'cora' : 'label', 'citeseer': 'encoded_labels'}
        sm = Semantic()
        text_df_train, text_df_test = sm.getTextData(data, label_col = label_col[data])
        text_vecs_train, text_vecs_test = sm.getTextEmbeddings(text_df_train, text_df_test, text_col = 'text', model_name = 'bert-base-nli-mean-tokens')
       
        #node_embeddings_train, node_embeddings_test = st.getStructuralEmbeddings(text_df_train, text_df_test, edges, label_col = label_col[data])
        labels_train, labels_test = Concat.getLabels(text_df_train, text_df_test, col = label_col[data])
        X_train, y_train, X_test, y_test = Concat.concatSemanticEmbeddings(text_vecs_train, labels_train, text_vecs_test, labels_test)
        preds, labels = Evaluate.trainAndPredict(X_train, X_test, y_train, y_test)
        name = data + 'BERT + SVM'
        results = Evaluate.getResults(preds, labels, len(y_train.unique()), results, name)
 
        return results
    
    def pipeline_without_label_awareness(data, results):
 
        label_col = {'cora' : 'label', 'citeseer': 'encoded_labels'}
        sm = Semantic()
        text_df_train, text_df_test = sm.getTextData_noLabels(data, label_col = label_col[data])
        text_vecs_train, text_vecs_test = sm.getTextEmbeddings(text_df_train, text_df_test, text_col = 'text', model_name = 'bert-base-nli-mean-tokens')
       
        st = Structural()
       
        if(data =='cora'):
            edges = st.getCoraEdges()
        else:
            edges = st.getCiteSeerEdges()
 
       
        node_embeddings_train, node_embeddings_test = st.getStructuralEmbeddings_noLabels(text_df_train, text_df_test, edges, label_col = label_col[data])
        labels_train, labels_test = Concat.getLabels(text_df_train, text_df_test, col = label_col[data])
        X_train, y_train, X_test, y_test = Concat.getCombinedEmbeddings(text_vecs_train, node_embeddings_train, labels_train, text_vecs_test, node_embeddings_test, labels_test)
        preds, labels = Evaluate.trainAndPredict(X_train, X_test, y_train, y_test)
        name = data + 'BERT + ComplEx + SVM'
        results = Evaluate.getResults(preds, labels, len(y_train.unique()), results, name)
 
        return results
    
    def pipeline_varying_models(data, results, ):

        label_col = {'cora' : 'label', 'citeseer': 'encoded_labels'}
        sm = Semantic()
        text_df_train, text_df_test = sm.getTextData(data, label_col = label_col[data])
        text_vecs_train, text_vecs_test = sm.getTextEmbeddings(text_df_train, text_df_test, text_col = 'text', model_name = 'bert-base-nli-mean-tokens')
        
        st = Structural()
        
        if(data =='cora'):
            edges = st.getCoraEdges()
        else:
            edges = st.getCiteSeerEdges()

        
        node_embeddings_train, node_embeddings_test = st.getStructuralEmbeddings(text_df_train, text_df_test, edges, label_col = label_col[data])
        labels_train, labels_test = Concat.getLabels(text_df_train, text_df_test, col = label_col[data])
        X_train, y_train, X_test, y_test = Concat.getCombinedEmbeddings(text_vecs_train, node_embeddings_train, labels_train, text_vecs_test, node_embeddings_test, labels_test)
        
        models = {'SVM': Evaluate.trainAndPredict, 
        'LogisticRegression': Evaluate.trainAndPredict_LogisticRegression,
        'RandomForest': Evaluate.trainAndPredict_RandomForest,
        'KNeighbors': Evaluate.trainAndPredict_KNeighbors,
        'DecisionTree': Evaluate.trainAndPredict_DecisionTree,
        'GradientBoosting': Evaluate.trainAndPredict_GradientBoosting
        }

        for i in models.keys():
            preds, labels = models[i](X_train, X_test, y_train, y_test)
            name = data + 'BERT + ComplEx + ' + i
            results = Evaluate.getResults(preds, labels, len(y_train.unique()), results, name)

        return results
    
    
