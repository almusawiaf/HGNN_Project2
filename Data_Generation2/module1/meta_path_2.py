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
        
        

class Meta_path:
    ''' The goal of this class is to read a heterogeneous graph HG, 
        type of similarity metrics 1. PC: Path-Count, 2. SPS: Symmetric PathSim,
        and create the list of meta-path-based similarities. 
        Then, we saved them to the given saving_path.'''
    
    def __init__(self, HG, similarity_type = 'PC', saving_path = ''):
        self.HG = HG
        self.saving_path = saving_path
        Nodes = list(self.HG.nodes())       
        
        self.metapath_List = []

        self.Patients =    [v for v in Nodes if v[0]=='C']
        self.Visits =      [v for v in Nodes if v[0]=='V']
        self.Medications = [v for v in Nodes if v[0]=='M']
        self.Diagnosis  =  [v for v in Nodes if v[0]=='D']
        self.Procedures =  [v for v in Nodes if v[0]=='P']
        self.Labs       =  [v for v in Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in Nodes if v[0]=='B']
        self.Nodes = Nodes
        
        print('extracting As from HG\n')
        W_cv = self.subset_adjacency_matrix(self.Patients + self.Visits)
        W_vm = self.subset_adjacency_matrix(self.Visits + self.Medications)
        W_vd = self.subset_adjacency_matrix(self.Visits + self.Diagnosis)
        W_vp = self.subset_adjacency_matrix(self.Visits + self.Procedures)  
        
        W_vl = self.subset_adjacency_matrix(self.Visits + self.Labs)  
        W_vb = self.subset_adjacency_matrix(self.Visits + self.MicroBio)   
        
        
        self.save_sparse(W_cv, 'cv')
        self.save_sparse(W_vm, 'vm')
        self.save_sparse(W_vd, 'vd')
        self.save_sparse(W_vp, 'vp')
        self.save_sparse(W_vl, 'vl')
        self.save_sparse(W_vb, 'vb')           
        print('=============================================================')
        # PathCount 
        # Heterogeneous similarity
        
        ## patient-visit-item
        print('Patients:')
        self.M(W_cv, W_vm, 'Patient-Medication', 'cvm')
        self.M(W_cv, W_vd, 'Patient-Diagnosis', 'cvd')
        self.M(W_cv, W_vp, 'Patient-Procedure', 'cvp')
        self.M(W_cv, W_vl, 'Patient-Lab', 'cvl')
        self.M(W_cv, W_vb, 'Patient-MicroBiology', 'cvb')
        
        print('Diagnoses:')
        self.M(W_vd.T, W_vm, 'Diagnosis-Medication', 'dvm')
        self.M(W_vd.T, W_vp, 'Diagnosis-Procedure', 'dvp')
        self.M(W_vd.T, W_vl, 'Diagnosis-Lab', 'dvl')
        self.M(W_vd.T, W_vb, 'Diagnosis-MicroBiology', 'dvb')
        
        print('Procedures:')
        self.M(W_vp.T, W_vm, 'Procedure-Medication', 'pvm')
        self.M(W_vp.T, W_vl, 'Procedure-Lab', 'pvl')
        self.M(W_vp.T, W_vb, 'Procedure-MicroBiology', 'pvb')
        
        self.M(W_vm.T, W_vl, 'Medication-Lab', 'mvl')
        self.M(W_vm.T, W_vb, 'Medication-MicroBiology', 'mvb')
        
        self.M(W_vl.T, W_vb, 'Lab-MicroBiology', 'lvb')
        
        print('Homogeneous similarity')        
        print('1. Patient-Patient')
        
        W_CVM = self.load_sparse('cvm')
        self.M(W_CVM, W_CVM.T, 'Patient-Visit-Medication-Visit-Patient', 'cvmvc')
        del W_CVM
        
        W_CVD = self.load_sparse('cvd')
        self.M(W_CVD, W_CVD.T, 'Patient-Visit-Diagnosis-Visit-Patient', 'cvdvc')
        del W_CVD
        
        W_CVP = self.load_sparse('cvp')
        self.M(W_CVP, W_CVP.T, 'Patient-Visit-Procedure-Visit-Patient', 'cvpvc')
        del W_CVP
        
        W_CVL = self.load_sparse('cvl')
        self.M(W_CVL, W_CVL.T, 'Patient-Visit-Lab-Visit-Patient', 'cvlvc')
        del W_CVL
        
        W_CVB = self.load_sparse('cvb')
        self.M(W_CVB, W_CVB.T, 'Patient-Visit-MicroBiology-Visit-Patient', 'cvbvc')
        del W_CVB
        
        print('2. visit-visit')
        self.M(W_vm, W_vm.T, 'Visit-Medication-Visit', 'vmv')
        self.M(W_vd, W_vd.T, 'Visit-Diagnosis-Visit', 'vdv')
        self.M(W_vp, W_vp.T, 'Visit-Procedure-Visit', 'vpv')
        self.M(W_vl, W_vl.T, 'Visit-Lab-Visit', 'vlv')
        self.M(W_vb, W_vb.T, 'Visit-MicroBiology-Visit', 'vbv')

        # ======================================================================================       
        save_list_as_pickle(self.metapath_List, f"{saving_path}/As", 'metapath_list')
        
        
    def save_sparse(self, A, f):
        self.metapath_List.append(f)
        if not issparse(A):
            A = sparse.csr_matrix(A)
        sparse.save_npz(f"{self.saving_path}/As/{f}.npz", A)
    
    def load_sparse(self, f):
        file_path = f"{self.saving_path}/As/{f}.npz"
        return sparse.load_npz(file_path)
    
    def M(self, W1, W2, msg, t, n_jobs=-1):
        if msg!='':
            print(f'\tWorking on: {msg}')
            
        if sparse.issparse(W1):
            W1 = W1.toarray()  # Convert sparse to dense NumPy array
        if sparse.issparse(W2):
            W2 = W2.toarray()  # Convert sparse to dense NumPy array

        # Convert to CSR format if not already
        W1_csr = csr_matrix(W1) if not isinstance(W1, csr_matrix) else W1
        W2_csr = csr_matrix(W2) if not isinstance(W2, csr_matrix) else W2

        # Get the number of rows in W1
        n_rows = W1_csr.shape[0]

        # Determine the chunk size per job
        chunk_size = n_rows // n_jobs if n_jobs > 1 else n_rows

        # Create row indices to split the workload
        row_chunks = [range(i, min(i + chunk_size, n_rows)) for i in range(0, n_rows, chunk_size)]

        # Use Parallel to distribute the computation
        # print(f'multiplying {W1.shape} * {W2.shape} in parallel...')
        results = Parallel(n_jobs=n_jobs)(
            delayed(parallel_multiply_chunk)(W1_csr, W2_csr, row_indices) for row_indices in row_chunks
        )

        # Stack the results vertically as a sparse matrix
        result = vstack(results)
        
        self.save_sparse(result, t)

        return result
        
    def subset_adjacency_matrix(self, subset_nodes):

        adj_matrix = nx.to_numpy_array(self.HG)
        mask = np.isin(self.Nodes, subset_nodes)
        for i in range(len(self.Nodes)):
            if not mask[i]:
                adj_matrix[i, :] = 0  # Zero out the row
                adj_matrix[:, i] = 0  # Zero out the column
        
        return adj_matrix
        
def parallel_multiply_chunk(W1_csr, W2_csr, row_indices):
    # Multiply a chunk of rows from W1_csr with W2_csr
    result_chunk = W1_csr[row_indices].dot(W2_csr)
    return result_chunk



def symmetricPathSim_3(PC, Nodes, selected_nodes):
    '''SPS
        G: heterogeneous graph,
       p: meta-path, 
       |p| = 3,
       return A(N by N).'''
       
    global PC_shared
    PC_shared = PC  # Use shared memory

    selected_indeces = [Nodes.index(n) for n in selected_nodes]
    n = len(Nodes)
    SPS = np.zeros((n, n))

    # Prepare the pairs of indices for parallel processing
    index_pairs = [(i, j) for i in range(len(selected_indeces) - 1)
                          for j in range(i + 1, len(selected_indeces))]

    # Use a backend that supports shared memory
    with parallel_backend('loky', n_jobs=40):
        results = Parallel()(delayed(calculate_sps)(selected_indeces[i], selected_indeces[j])
                             for i, j in index_pairs)

    # Populate the SPS matrix with the computed results
    for idx, (i, j) in enumerate(index_pairs):
        ni, nj = selected_indeces[i], selected_indeces[j]
        SPS[ni, nj] = results[idx]
        SPS[nj, ni] = results[idx]  # Ensure symmetry

    return SPS


def create_folder(the_path):
    if not os.path.exists(the_path):
        os.makedirs(the_path)