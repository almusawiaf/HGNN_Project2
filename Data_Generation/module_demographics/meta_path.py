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

class Meta_path:
    
    def __init__(self, HG, similarity_type = 'PC', saving_path = ''):
        self.HG = HG

        Nodes = list(self.HG.nodes())

        self.Patients =    [v for v in Nodes if v[0]=='C']
        self.Gender =      [v for v in Nodes if v[0]=='G']
        self.Expire_Flag = [v for v in Nodes if v[0]=='E']
        self.Visits =      [v for v in Nodes if v[0]=='V']
        self.Medications = [v for v in Nodes if v[0]=='M']
        self.Diagnosis  =  [v for v in Nodes if v[0]=='D']
        self.Procedures =  [v for v in Nodes if v[0]=='P']
        self.Labs       =  [v for v in Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in Nodes if v[0]=='B']
        self.Nodes = Nodes
        
        print('extracting As from HG\n')
        W_cg = self.subset_adjacency_matrix(self.Patients + self.Gender)
        W_ce = self.subset_adjacency_matrix(self.Patients + self.Expire_Flag)
        W_cv = self.subset_adjacency_matrix(self.Patients + self.Visits)
        W_vm = self.subset_adjacency_matrix(self.Visits + self.Medications)
        W_vd = self.subset_adjacency_matrix(self.Visits + self.Diagnosis)
        W_vp = self.subset_adjacency_matrix(self.Visits + self.Procedures)  
        
        W_vl = self.subset_adjacency_matrix(self.Visits + self.Labs)  
        W_vb = self.subset_adjacency_matrix(self.Visits + self.MicroBio)          
        print([a.shape for a in [W_cv, W_vm, W_vd, W_vp, W_vl, W_vb]])

        print('=============================================================')
        print('Multiplication phase...\n')
        # PathCount 
        # Heterogeneous similarity
        
        ## patient-visit-item
        W_CVM = M(W_cv, W_vm)
        W_CVD = M(W_cv, W_vd)
        W_CVP = M(W_cv, W_vp)
        W_CVL = M(W_cv, W_vl)
        W_CVB = M(W_cv, W_vb)
        
        W_DVM = M(W_vd.T, W_vm)
        W_DVP = M(W_vd.T, W_vp)
        W_DVL = M(W_vd.T, W_vl)
        W_DVB = M(W_vd.T, W_vb)
        
        W_PVM = M(W_vp.T, W_vm)
        W_PVL = M(W_vp.T, W_vl)
        W_PVB = M(W_vp.T, W_vb)
        
        W_MVL = M(W_vm.T, W_vl)
        W_MVB = M(W_vm.T, W_vb)
        
        W_LVB = M(W_vl.T, W_vb)
        
        # Homogeneous similarity        
        W_CGC   = M(W_cg, W_cg.T)
        W_CEC   = M(W_ce, W_ce.T)
        
        W_CVMVC = M(W_CVM, W_CVM.T)
        W_CVDVC = M(W_CVD, W_CVD.T)
        W_CVPVC = M(W_CVP, W_CVP.T)
        W_CVLVC = M(W_CVL, W_CVL.T)
        W_CVBVC = M(W_CVB, W_CVB.T)
        print('Patient-Patient completed!\n')
        
        W_VMV = M(W_vm, W_vm.T)
        W_VDV = M(W_vd, W_vd.T)
        W_VPV = M(W_vp, W_vp.T)
        W_VLV = M(W_vl, W_vl.T)
        W_VBV = M(W_vb, W_vb.T)
        print('visit-visit completed!\n')
        
        W_DVMVD = M(W_DVM, W_DVM.T)
        W_DVPVD = M(W_DVP, W_DVP.T)
        W_DVLVD = M(W_DVL, W_DVL.T)
        W_DVBVD = M(W_DVB, W_DVB.T)
        print('Diagnoses-Diagnoses completed!\n')
        
        W_MVDVM = M(W_DVM.T, W_DVM)
        W_MVPVM = M(W_PVM.T, W_PVM)
        W_MVLVM = M(W_MVL, W_MVL.T)
        W_MVBVM = M(W_MVB, W_MVB.T)
        print('Med-Med completed!\n')
        
        W_PVDVP = M(W_DVP.T, W_DVP)
        W_PVMVP = M(W_PVM, W_PVM.T)
        W_PVLVP = M(W_PVL, W_PVL.T)
        W_PVBVP = M(W_PVB, W_PVB.T)
        print('Proced-Proced completed!\n')
        
        W_LVDVL = M(W_DVL.T, W_DVL)
        W_LVPVL = M(W_PVL.T, W_PVL)
        W_LVMVL = M(W_MVL.T, W_MVL)
        W_LVBVL = M(W_LVB, W_LVB.T)
        print('Labs-Labs completed!\n')
        
        W_BVDVB = M(W_DVB.T, W_DVB)
        W_BVMVB = M(W_MVB.T, W_MVB)
        W_BVPVB = M(W_PVB.T, W_PVB)
        W_BVLVB = M(W_LVB, W_LVB.T)
        print('Micro-Bio - Micro-Bio completed!\n')

        print('=============================================================')
        print('Multiplication phase...\n')

        if similarity_type=='SPS':
            SPS_CVMVC = symmetricPathSim_3(W_CVMVC, self.Nodes, self.Patients)
            SPS_CVDVC = symmetricPathSim_3(W_CVDVC, self.Nodes, self.Patients)
            SPS_CVPVC = symmetricPathSim_3(W_CVPVC, self.Nodes, self.Patients)
            SPS_CVLVC = symmetricPathSim_3(W_CVLVC, self.Nodes, self.Patients)
            SPS_CVBVC = symmetricPathSim_3(W_CVBVC, self.Nodes, self.Patients)
            print('Patients-Patients completed!\n')
            
            SPS_VMV = symmetricPathSim_3(W_VMV, self.Nodes, Visits)
            SPS_VDV = symmetricPathSim_3(W_VDV, self.Nodes, Visits)
            SPS_VPV = symmetricPathSim_3(W_VPV, self.Nodes, Visits)
            SPS_VLV = symmetricPathSim_3(W_VLV, self.Nodes, Visits)
            SPS_VBV = symmetricPathSim_3(W_VBV, self.Nodes, Visits)
            print('visit-visit completed!\n')
            
            SPS_DVMVD = symmetricPathSim_3(W_DVMVD, self.Nodes, Diagnosis)
            SPS_DVPVD = symmetricPathSim_3(W_DVPVD, self.Nodes, Diagnosis)
            SPS_DVLVD = symmetricPathSim_3(W_DVLVD, self.Nodes, Diagnosis)
            SPS_DVBVD = symmetricPathSim_3(W_DVBVD, self.Nodes, Diagnosis)
            print('Diagnoses-Diagnoses completed!\n')
            
            SPS_MVDVM = symmetricPathSim_3(W_MVDVM, self.Nodes, Medications)
            SPS_MVPVM = symmetricPathSim_3(W_MVPVM, self.Nodes, Medications)
            SPS_MVLVM = symmetricPathSim_3(W_MVLVM, self.Nodes, Medications)
            SPS_MVBVM = symmetricPathSim_3(W_MVBVM, self.Nodes, Medications)
            print('Med-Med completed!\n')
            
            SPS_PVDVP = symmetricPathSim_3(W_PVDVP, self.Nodes, Procedures)
            SPS_PVMVP = symmetricPathSim_3(W_PVMVP, self.Nodes, Procedures)
            SPS_PVLVP = symmetricPathSim_3(W_PVLVP, self.Nodes, Procedures)
            SPS_PVBVP = symmetricPathSim_3(W_PVBVP, self.Nodes, Procedures)
            print('Proced-Proced completed!\n')
        
            SPS_LVDVL = symmetricPathSim_3(W_LVDVL, self.Nodes, Labs)
            SPS_LVMVL = symmetricPathSim_3(W_LVMVL, self.Nodes, Labs)
            SPS_LVPVL = symmetricPathSim_3(W_LVPVL, self.Nodes, Labs)
            SPS_LVBVL = symmetricPathSim_3(W_LVBVL, self.Nodes, Labs)
            print('Labs-Labs completed!\n')
        
            SPS_BVDVB = symmetricPathSim_3(W_BVDVB, self.Nodes, MicroBio)
            SPS_BVMVB = symmetricPathSim_3(W_BVMVB, self.Nodes, MicroBio)
            SPS_BVPVB = symmetricPathSim_3(W_BVPVB, self.Nodes, MicroBio)
            SPS_BVLVB = symmetricPathSim_3(W_BVLVB, self.Nodes, MicroBio)
            print('Proced-Proced completed!\n')

            Ws = [SPS_CVMVC, SPS_CVDVC, SPS_CVPVC, SPS_CVLVC, SPS_CVBVC,          
                  SPS_VMV,   SPS_VDV,   SPS_VPV,   SPS_VLV,   SPS_VBV,          
                  SPS_DVMVD, SPS_DVPVD, SPS_DVLVD, SPS_DVBVD,         
                  SPS_MVPVM, SPS_MVDVM, SPS_MVLVM, SPS_MVBVM,        
                  SPS_PVDVP, SPS_PVMVP, SPS_PVLVP, SPS_PVBVP, 
                  W_cv, W_vd, W_vm, W_vp, W_vl, W_vb,       
                  W_CVM, W_CVD, W_CVP, W_CVL, W_CVB,      
                  W_DVM, W_DVP, W_DVL, W_DVB,
                  W_PVM, W_PVL, W_PVB,
                  W_MVL, W_MVB,
                  W_LVB,
                 ]
        else:
            Ws = [W_CGC,   W_CEC,
                  W_CVMVC, W_CVDVC, W_CVPVC, W_CVLVC, W_CVBVC,
                  W_VMV,   W_VDV,   W_VPV,   W_VLV,   W_VBV,          
                  W_DVMVD, W_DVPVD, W_DVLVD, W_DVBVD, 
                  W_MVPVM, W_MVDVM, W_MVLVM, W_MVBVM,        
                  W_PVDVP, W_PVMVP, W_PVLVP, W_PVBVP,        
                  W_cv, W_vd, W_vm, W_vp, W_vl, W_vb,       
                  W_CVM, W_CVD, W_CVP, W_CVL, W_CVB,      
                  W_DVM, W_DVP, W_DVL, W_DVB,
                  W_PVM, W_PVL, W_PVB,
                  W_MVL, W_MVB,
                  W_LVB,
                 ]
            

        print(f'Number of meta-paths = {len(Ws)}')
        print('==============================================================')
        for i, A in enumerate(Ws):
            # Check if A is sparse or dense
            if issparse(A):
                non_zero_count = A.count_nonzero()
            else:
                non_zero_count = np.count_nonzero(A)
            
            print(f"Matrix {i}: {non_zero_count} non-zero elements")
            
            if non_zero_count > 1000000:
                B = keep_top_million(A, top_n=1000000)
                print(f"\tSaving one million non-zero values... (after reduction: {B.count_nonzero()} non-zero elements)")
            else:
                B = A
                print(f"\tSaving all non-zero values... ({non_zero_count} non-zero elements)")
        
            # Convert to CSR format if not already sparse
            if not issparse(B):
                B = sparse.csr_matrix(B)
            
            # Save the sparse matrix
            sparse.save_npz(f"{saving_path}/sparse_matrix_{i}.npz", B)
        
        
        print('==============================================================')
        num_As = len(Ws)
        # Check if the directory exists, and create it if it doesn't
        Nodes_path = f'{saving_path}/edges'
        if not os.path.exists(Nodes_path):
            os.makedirs(Nodes_path)
        
        D = [get_edges_dict(f'{saving_path}/sparse_matrix_{i}.npz') for i in range(num_As)]
        sorted_list_of_dicts = sorted(D, key=lambda x: len(x), reverse=True)
        
        unique_edges = set()
        
        for e in sorted_list_of_dicts:
            unique_edges.update(e.keys())
        
            
        with open(f'{saving_path}/edges/edge_list.pkl', 'wb') as file:
            pickle.dump(unique_edges, file)
        
        print('done saving [unique edges]: ', len(unique_edges))
        
        for i, d in enumerate(D):
            print(f'Working on {i}th file...')
            results = []
            for e in unique_edges:
                if e in d:
                    results.append(d[e])
                else:
                    results.append(0)
            print(f'\tdone...')
            print('\tSaving...')
        
            with open(f'{saving_path}/edges/edge_weight{i}.pkl', 'wb') as file:
                pickle.dump(results, file)
        
            print(f'\tDone saving edge_weight{i}...')
    
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

def M(W1, W2, n_jobs=-1):
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
    print(f'multiplying {W1.shape} * {W2.shape} in parallel...')
    results = Parallel(n_jobs=n_jobs)(
        delayed(parallel_multiply_chunk)(W1_csr, W2_csr, row_indices) for row_indices in row_chunks
    )

    # Stack the results vertically as a sparse matrix
    result = vstack(results)
    print("Done multiplication...")

    return result

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

def keep_top_million(sparse_matrix, top_n=1000000):
    # Convert to CSR if the matrix is dense
    if isinstance(sparse_matrix, np.ndarray):
        sparse_matrix = sparse.csr_matrix(sparse_matrix)

    # Step 1: Find the threshold for the top million values
    if sparse_matrix.nnz <= top_n:
        # If the matrix has fewer non-zeros than top_n, return it as is
        return sparse_matrix
    
    # Extract the non-zero data from the sparse matrix
    sorted_data = np.sort(sparse_matrix.data)[-top_n]
    
    # Step 2: Filter the sparse matrix based on this threshold
    mask = sparse_matrix.data >= sorted_data
    
    # Apply the mask to keep only values in the top million
    filtered_data = sparse_matrix.data[mask]
    filtered_indices = sparse_matrix.indices[mask]
    filtered_indptr = np.zeros(sparse_matrix.shape[0] + 1, dtype=int)
    
    # Recalculate indptr array based on the filtered data
    current_index = 0
    for i in range(sparse_matrix.shape[0]):
        row_start = sparse_matrix.indptr[i]
        row_end = sparse_matrix.indptr[i+1]
        
        # Count non-zero elements in the current row that are in the top million
        filtered_indptr[i+1] = filtered_indptr[i] + np.sum(mask[row_start:row_end])
    
    # Create a new sparse matrix with the filtered data
    filtered_matrix = sparse.csr_matrix((filtered_data, filtered_indices, filtered_indptr), shape=sparse_matrix.shape)
    
    return filtered_matrix


def get_edges_dict(the_path):
    A = sparse.load_npz(the_path)
    if not isinstance(A, scipy.sparse.coo_matrix):
        A = A.tocoo()
    filtered_entries = (A.col > A.row) & (A.data > 0)
    upper_triangle_positive = {(row, col): data for row, col, data in zip(A.row[filtered_entries], A.col[filtered_entries], A.data[filtered_entries])}
    return upper_triangle_positive