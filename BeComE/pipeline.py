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
