import numpy as np
import pandas as pd

class Concat:
    
    def getLabels(text_df_train, text_df_test, col):

        labels_train = text_df_train[col].to_numpy()
        labels_test = text_df_test[col].to_numpy()
        labels_train = labels_train.reshape(-1, 1)
        labels_test = labels_test.reshape(-1, 1)

        return labels_train, labels_test
    
    def getCombinedEmbeddings(text_vecs_train, node_embeddings_train, labels_train, text_vecs_test, node_embeddings_test, labels_test):

        data_np_train = np.hstack((text_vecs_train, node_embeddings_train, labels_train))
        data_np_test = np.hstack((text_vecs_test, node_embeddings_test, labels_test))

        data_actual_train = pd.DataFrame(data_np_train)
        data_actual_test = pd.DataFrame(data_np_test)

        data_actual_train[data_actual_train.shape[1]-1] = data_actual_train[data_actual_train.shape[1]-1].astype(int)
        data_actual_test[data_actual_test.shape[1]-1] = data_actual_test[data_actual_test.shape[1]-1].astype(int)

        X_train = data_actual_train.drop(columns=[data_actual_train.shape[1]-1])
        y_train = data_actual_train[data_actual_train.shape[1]-1]
        X_test = data_actual_test.drop(columns=[data_actual_test.shape[1]-1])
        y_test = data_actual_test[data_actual_test.shape[1]-1]

        return X_train, y_train, X_test, y_test
    
    def concatStructualEmbeddings(node_embeddings_train, labels_train, node_embeddings_test, labels_test):
 
        data_np_train = np.hstack((node_embeddings_train, labels_train))
        data_np_test = np.hstack((node_embeddings_test, labels_test))
 
        data_actual_train = pd.DataFrame(data_np_train)
        data_actual_test = pd.DataFrame(data_np_test)
 
        data_actual_train[data_actual_train.shape[1]-1] = data_actual_train[data_actual_train.shape[1]-1].astype(int)
        data_actual_test[data_actual_test.shape[1]-1] = data_actual_test[data_actual_test.shape[1]-1].astype(int)
 
        X_train = data_actual_train.drop(columns=[data_actual_train.shape[1]-1])
        y_train = data_actual_train[data_actual_train.shape[1]-1]
        X_test = data_actual_test.drop(columns=[data_actual_test.shape[1]-1])
        y_test = data_actual_test[data_actual_test.shape[1]-1]
 
        return X_train, y_train, X_test, y_test
 
    def concatSemanticEmbeddings(text_vecs_train, labels_train, text_vecs_test, labels_test):
 
        data_np_train = np.hstack((text_vecs_train, labels_train))
        data_np_test = np.hstack((text_vecs_test, labels_test))
 
        data_actual_train = pd.DataFrame(data_np_train)
        data_actual_test = pd.DataFrame(data_np_test)
 
        data_actual_train[data_actual_train.shape[1]-1] = data_actual_train[data_actual_train.shape[1]-1].astype(int)
        data_actual_test[data_actual_test.shape[1]-1] = data_actual_test[data_actual_test.shape[1]-1].astype(int)
 
        X_train = data_actual_train.drop(columns=[data_actual_train.shape[1]-1])
        y_train = data_actual_train[data_actual_train.shape[1]-1]
        X_test = data_actual_test.drop(columns=[data_actual_test.shape[1]-1])
        y_test = data_actual_test[data_actual_test.shape[1]-1]
 
        return X_train, y_train, X_test, y_test