from scipy.sparse import csr_matrix, vstack
from joblib import Parallel, delayed

import networkx as nx
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import pickle
import scipy
from scipy import sparse
from scipy.sparse import issparse
import os        


def save_list_as_pickle(L, given_path, file_name):
    import pickle
    print(f'saving to {given_path}/{file_name}.pkl')
    with open(f'{given_path}/{file_name}.pkl', 'wb') as file:
        pickle.dump(L, file)

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict  
        
        

class Reduction:
    ''' The goal of this class is to:
        1. read a list of similarity matrices.
        2. select the highest-weighted edges
        3. convert them to edge_list
        4. save the final edge_weight per matrix.'''
        
    def __init__(self, base_path = '', gpu = False, PSGs = False):
        saving_path = f'{base_path}/HGNN_data/'                        
        Ws1, meta_path_list = self.read_Ws(saving_path, 'As')
        if PSGs:
            Ws2, PSGs_List = self.read_PSGs(base_path)
            
            Ws1 = Ws1 + Ws2
            meta_path_list = meta_path_list + PSGs_List

        Ws, selected_meta_paths = self.selecting_high_edges(Ws1, meta_path_list)
        self.final_Ws_with_unique_list_of_edges(Ws, selected_meta_paths, saving_path)    
            
        
    def read_Ws(self, saving_path, folder_name):
        '''reading the similarity matrices and corresponding names'''
        selected_i = load_dict_from_pickle(f"{saving_path}/{folder_name}/selected_i.pkl")
        print(selected_i[-1])
        metapath_list = load_dict_from_pickle(f'{saving_path}/{folder_name}/metapath_list.pkl')
        return [self.get_edges_dict(f'{saving_path}/{folder_name}/sparse_matrix_{i}.npz') for i in selected_i], metapath_list

    def read_PSGs(self, base_path):
        '''reading the PSG matrices and corresponding names'''
        PSGs_List = load_dict_from_pickle(f'{base_path}/PSGs/PSGs_List.pkl')
        return [self.get_edges_dict(f'{base_path}/PSGs/{i}.npz') for i in ['B', 'D', 'L', 'M', 'P']], PSGs_List

    def get_edges_dict(self, the_path):
        A = sparse.load_npz(the_path)
        if not isinstance(A, scipy.sparse.coo_matrix):
            A = A.tocoo()
        filtered_entries = (A.col > A.row) & (A.data > 0)
        upper_triangle_positive = {(row, col): data for row, col, data in zip(A.row[filtered_entries], A.col[filtered_entries], A.data[filtered_entries])}
        return upper_triangle_positive

    def selecting_high_edges(self, Ws, meta_path_list):
        '''Selecting top weighted edges'''
        D, D_names = [], []
        for i, A in enumerate(Ws):
            # A is a dictionary, no need to check for sparsity
            num_non_zeros = len(A)
            print(f"Matrix {i}: {num_non_zeros} non-zero elements")
            
            if num_non_zeros > 0:
                if num_non_zeros > 1000000:
                    B = keep_top_million(A, top_n=1000000)  # Define this function to keep top N edges in the dictionary
                    print(f"\tSaving one million non-zero values... (after reduction: {len(B)} non-zero elements)")
                else:
                    B = A
                    print(f"\tSaving all non-zero values... ({num_non_zeros} non-zero elements)")
            
                D.append(B)  # Store the dictionary directly
                D_names.append(meta_path_list[i])
            else:
                print(f'Matrix {i} has zero values. Not saving...')
        
        return D, D_names


    def final_Ws_with_unique_list_of_edges(self, D, selected_meta_paths, saving_path):        
        '''creating and saving the last representation of the edges_list and edges_weight'''
        sorted_list_of_dicts = sorted(D, key=lambda x: len(x), reverse=True)        
        unique_edges = set()        
        for e in sorted_list_of_dicts:
            unique_edges.update(e.keys())
        
        create_folder(f'{saving_path}/edges')
        with open(f'{saving_path}/edges/edge_list.pkl', 'wb') as file:
            pickle.dump(unique_edges, file)
        
        print('done saving [unique edges]: ', len(unique_edges))        

        # ======================================== Reflect the (edge_list) into all A's =======================================
        for i, d in enumerate(D):
            print(f'Working on {i}th file...')
            results = []
            for e in unique_edges:
                if e in d:
                    results.append(d[e])
                else:
                    results.append(0)
                    
            with open(f'{saving_path}/edges/edge_weight{i}.pkl', 'wb') as file:
                pickle.dump(results, file)
        
        # saving the selected_meta_paths
        
        save_list_as_pickle(selected_meta_paths, f'{saving_path}/edges', 'selected_meta_paths')
    
    
    def get_SNF(self):
        SNF_inst = SNF_class(self.HG, self.Nodes)
        return SNF_inst.A


def create_folder(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path)


def keep_top_million(edge_dict, top_n=1000000):
    if len(edge_dict) <= top_n:
        return edge_dict
    
    # Step 1: Sort the dictionary by values (edge weights) and keep only the top N
    sorted_edges = sorted(edge_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Step 2: Keep the top N edges
    top_edges = dict(sorted_edges[:top_n])
    
    return top_edges