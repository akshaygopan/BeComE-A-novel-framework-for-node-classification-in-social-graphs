import numpy as np
import pandas as pd
from ampligraph.compat import ComplEx

class Structural:

    def getCiteSeerEdges(self):

        file_path= 'data\CiteSeer_original.cites'

        with open(file_path, 'r') as file:
                lines = file.readlines()

        starts = []
        dests = []

        for i in lines:

                cities = i.split()
                starts.append(cities[0])
                dests.append(cities[1])

        edges = pd.DataFrame({
                'source_id': starts,
                'dest_id': dests
            })

        return edges
    
    def getCoraEdges(self):

        edges = pd.read_csv('data\df_cln2.csv')
        return edges
    
    def getStructuralEmbeddings(self, text_df_train, text_df_test, edges, label_col):

        df = pd.concat([text_df_train, text_df_test])
        present = df.node_id.unique()
        edges = edges[(edges['dest_id'].isin(present)) | (edges['source_id'].isin(present))]
        edges_list = []

        for row in range(len(edges)):

            relation = -1
            source = edges.iloc[row]['source_id']
            dest = edges.iloc[row]['dest_id']

            try:

                #source_label = df[df['node_id'] == source].iloc[0][label_col]
                dest_label = df[df['node_id'] == dest].iloc[0][label_col]

            except:

                relation = -1

            #if(source_label == dest_label):
            relation = dest_label

            #if(source in text_df_train.node_id.unique()):

            #    relation = dest_label

            #if(source in text_df_test.node_id.unique()):

            #   relation = dest_label

            #if(dest in text_df_test.node_id.unique()):
            #    relation = -1


            #if(source in text_df_test.node_id.unique()):
            #    relation = -1

            edge = [source, relation, dest]
            edges_list.append(edge)

        embeds_model = self.getStructualEmbeddingModel(edges_list)
        node_embeddings_train = embeds_model.get_embeddings(text_df_train['node_id'].values, embedding_type='entity')
        node_embeddings_test = embeds_model.get_embeddings(text_df_test['node_id'].values, embedding_type='entity')

        return node_embeddings_train, node_embeddings_test
    

    def getStructualEmbeddingModel(self, edges_list):

        edges_list = np.array(edges_list)
        embed_dim = 200
        epochs =  200
        batches_count = 250
        verbose = True
        optimizer = "adam"
        optimizer_params = {'lr' :  0.001}
        embeds_model = ComplEx(k = embed_dim, epochs =  epochs, batches_count =  batches_count, verbose=verbose, optimizer = optimizer, optimizer_params = optimizer_params)
        embeds_model.fit(edges_list)

        return embeds_model
            

    

    
