from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy import sparse
import numpy as np


class Patients_Similarity:
    
    def __init__(self, HG, Nodes, PSGs_path):
        '''reading a HG and Nodes
        1. create OHV per node type
        2. measure the similarity
        3. measure SNF and hold it as A.
        '''
        self.HG = HG
        self.Nodes = Nodes
        self.PSGs_path = PSGs_path
        # ======================================================
        self.Patients =    [v for v in self.Nodes if v[0]=='C']
        self.Visits =      [v for v in self.Nodes if v[0]=='V']
        self.Medications = [v for v in self.Nodes if v[0]=='M']
        self.Diagnosis  =  [v for v in self.Nodes if v[0]=='D']
        self.Procedures =  [v for v in self.Nodes if v[0]=='P']
        self.Labs       =  [v for v in self.Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in self.Nodes if v[0]=='B']
        # ======================================================
        for code in ['M', 'D', 'P', 'L', 'B']:
            self.process(code)
        # ======================================================

        
    def process(self, Code):
        print(f'Measure the similarity, expand it and save to PSGs/{Code}.npz')
        X = self.get_X(Code)
        X = cosine_similarity(X)
        X = self.expand_A(X)
        self.save_npz(X, Code)
        
            
    def save_npz(self, A, file_name):
        sparse_A = sparse.csr_matrix(A)  # convert the dense matrix A to a sparse format
        sparse.save_npz(f'{self.PSGs_path}/PSGs/{file_name}.npz', sparse_A)

    def expand_A(self, A):
        n = len(self.Patients)
        m = len(self.Nodes)
        expanded_matrix = np.zeros((m, m))
        expanded_matrix[:n, :n] = A
        
        return expanded_matrix


    def get_X(self, clinical_type):
        '''Extract the clinical_type based features for patients only...'''
        
        print(f'Getting the OHV for {clinical_type}')
        if clinical_type=='M':
            F = self.Medications
        elif clinical_type=='P':
            F = self.Procedures
        elif clinical_type=='L':
            F = self.Labs
        elif clinical_type=='D':
            F = self.Diagnosis
        elif clinical_type=='B':
            F = self.MicroBio
            
        # get the indices of the selected clinical type.
        F_indeces = {p:k for k,p in enumerate(F)}

        # extracting features for patients only.
        # we need to append zero rows and cols to it later 
        # after calculating the similarity.

        X = []
        for v in self.Patients:
            f = [0] * len(F)
            for u_visit in self.HG.neighbors(v):
                for u in self.HG.neighbors(u_visit):
                    if u[0] in [clinical_type]:
                        f[F_indeces[u]] = 1
            X.append(f)
        
        return np.array(X)

    def get_X_sub_case(self, clinical_type):
        '''Gender and Expire flag'''
        print(f'Getting the OHV for {clinical_type}')
        if clinical_type=='G':
            F = self.Gender
        elif clinical_type=='E':
            F = self.Expire_Flag
            
        F_indeces = {p:k for k,p in enumerate(F)}

        X = []
        for v in self.Patients:
            f = [0] * len(F)
            for u in self.HG.neighbors(v):
                if u[0] in [clinical_type]:
                    f[F_indeces[u]] = 1
            X.append(f)
        
        return np.array(X)