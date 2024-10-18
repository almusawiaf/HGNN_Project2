'''
generate_HG is class used to generate the 203 heterogeneous graph only.
We only included patients with diagnoses.
'''
import numpy as np
import networkx as nx


def remove_patients_and_linked_visits(nodes, HG):
    """
    Remove patients and their linked visits from the graph HG.
    
    Parameters:
    nodes (list): List of patient nodes to be removed.
    HG (networkx.Graph): The heterogeneous graph.
    
    Returns:
    networkx.Graph: The modified graph with patients and their visits removed.
    """
    print(f'Number of PATIENTS to remove: {len(nodes)}')
    
    # Using a set to store nodes to avoid duplicates
    nodes_to_remove = set(nodes)
    
    # Find all visit nodes connected to the patient nodes
    for patient in nodes:
        visit_neighbors = {v for v in HG.neighbors(patient) if v[0] == 'V'}
        nodes_to_remove.update(visit_neighbors)
    
    print(f'Number of nodes to remove: {len(nodes_to_remove)}')

    # Removing nodes from the graph in place to avoid deepcopy
    HG.remove_nodes_from(nodes_to_remove)
    
    return HG


'''
generate_HG is class used to generate the 203 heterogeneous graph only.
We only included patients with diagnoses.
'''


class XY_preparation:
    icd_code_ranges = [
            ('001', '139'),  # infectious and parasitic diseases
            ('140', '239'),  # neoplasms
            ('240', '279'),  # endocrine, nutritional and metabolic diseases, and immunity disorders
            ('280', '289'),  # diseases of the blood and blood-forming organs
            ('290', '319'),  # mental disorders
            ('320', '389'),  # diseases of the nervous system and sense organs
            ('390', '459'),  # diseases of the circulatory system
            ('460', '519'),  # diseases of the respiratory system
            ('520', '579'),  # diseases of the digestive system
            ('580', '629'),  # diseases of the genitourinary system
            ('630', '679'),  # complications of pregnancy, childbirth, and the puerperium
            ('680', '709'),  # diseases of the skin and subcutaneous tissue
            ('710', '739'),  # diseases of the musculoskeletal system and connective tissue
            ('740', '759'),  # congenital anomalies
            ('760', '779'),  # certain conditions originating in the perinatal period
            ('780', '799'),  # symptoms, signs, and ill-defined conditions
            ('800', '999'),  # injury and poisoning
            ('E', 'V'),  # E and V codes are not included since they don't fit the numerical pattern
        ]

    
    def __init__(self, HG):

        self.set_HG_components(HG)
        self.Y = self.get_Y()
        patients_to_remove = self.identify_patients_with_no_labels()
        new_G = remove_patients_and_linked_visits(patients_to_remove, self.HG)
        
        self.set_HG_components(new_G)
        self.Y = self.get_Y()                
        self.Y = self.remove_one_class_columns(self.Y)                
        self.X = self.get_X()
        
        
    
    def set_HG_components(self, HG):
        self.HG = HG        
        HG_Nodes = list(self.HG.nodes())
        self.Patients =    [v for v in HG_Nodes if v[0]=='C']
        self.Visits =      [v for v in HG_Nodes if v[0]=='V']
        self.Medications = [v for v in HG_Nodes if v[0]=='M']
        self.Diagnosis  =  [v for v in HG_Nodes if v[0]=='D']
        self.Procedures =  [v for v in HG_Nodes if v[0]=='P']
        self.Labs       =  [v for v in HG_Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in HG_Nodes if v[0]=='B']
        self.Nodes = self.Patients + self.Visits + self.Medications + self.Diagnosis + self.Procedures + self.Labs + self.MicroBio
        
        
    def get_X(self):
        print('getting the feature set for all nodes')
        F = self.Medications  + self.Procedures + self.Labs + self.MicroBio 
        F_indeces = {p:k for k,p in enumerate(F)}

        X = []
        for v in self.Nodes:
            f = [0] * len(F)
            if v[0]=='C':
                for u_visit in self.HG.neighbors(v):
                    for u in self.HG.neighbors(u_visit):
                        if u[0] in ['P', 'M', 'L', 'B']:
                            f[F_indeces[u]] = 1
            X.append(f)
        
        return np.array(X)
    
    
    

    def get_Y(self):
        '''return a binary matrix of diagnoses, if exites across all visits.'''

        def and_over_rows(binary_lists):
            """Compute the AND function across all rows of a list of lists of binary numbers."""
            binary_array = np.array(binary_lists)
            return np.all(binary_array == 1, axis=0).astype(int).tolist()          
            
        F = self.Diagnosis
        F_indeces = {p:k for k,p in enumerate(F)}

        Y = []
        for v in self.Nodes:
            if v[0]=='C':
                temp_Y = []
                for u_visit in self.HG.neighbors(v):
                    f = [0] * len(F)
                    for u in self.HG.neighbors(u_visit):
                        if u[0] in ['D']:
                            f[F_indeces[u]] = 1
                    temp_Y.append(f)
                final_Y = and_over_rows(temp_Y)
                Y.append(final_Y)
            else:
                Y.append([0] * len(F))
        
        return np.array(Y)
        
                
    def remove_one_class_columns(self, Y):
        def column_contains_one_class(column):
            unique_values = np.unique(column)  # Get unique values in the column
            return len(unique_values) == 1  # Column contains only one class if unique values are 1
    
        columns_to_keep = []
        num_Patients = len(self.Patients)
    
        # Iterate over each column in Y
        for column_index in range(Y.shape[1]):
            column = Y[:num_Patients, column_index]  # Extract the specific column
            if not column_contains_one_class(column):
                columns_to_keep.append(column_index)
    
        # Create a new array Y_new with only the columns that are not one-class
        Y_new = Y[:, columns_to_keep]
    
        return Y_new

    
    def identify_patients_with_no_labels(self):
        # Step 1: Find the rows where the first element of v is 'C'
        N = [i for i, v in enumerate(self.Nodes) if v[0] == 'C']

        # Step 2: Calculate the row sums for the selected rows
        row_sums = np.sum(Y[N], axis=1)

        # Step 3: Identify which rows have a zero sum
        zero_sum_rows = row_sums == 0

        # Step 4: Get the indices of the rows with zero sum
        zero_sum_indices = np.where(zero_sum_rows)[0]

        # Step 5: Print the number of rows and the indices
        count_zero_sum_rows = len(zero_sum_indices)
        print("Number of rows with zero sum:", count_zero_sum_rows)
        # print("Indices of rows with zero sum:", zero_sum_indices)
        
        return np.array(self.Nodes)[zero_sum_indices]

    def get_Y_superclasses(self):
        print('Create a new One Hot vector per node')

        def get_D(OHV):
            return set([diagnosis for diagnosis, value in zip(self.Diagnosis, OHV) if value == 1])

        def get_the_index_of_range(i):
            new_ICD_range = []
            for (s,e) in icd_code_ranges:
                if (s,e) != icd_code_ranges[-1]:
                    new_ICD_range.append([int(s), int(e)])
            # ++++++++++++++++++++++++++++++++++++++++++++++
            if i[2] in ['V', 'E']:
                return len(new_ICD_range)
            else:
                for item in new_ICD_range:
                    if int(i[2:]) >= item[0] and int(i[2:]) <= item[1]:
                        return new_ICD_range.index(item)

        # =============================================================
        superclass_Y = []
        for y in self.Y:
            temp_Y = [0] * len(icd_code_ranges)

            # 1. get the set of diagnoses per node...
            D_y = get_D(y)
            
            # 2. get the index per diagnoses:
            D_indeces = set([get_the_index_of_range(i) for i in D_y])
            
            # 3. Set the final ONV 
            for d in D_indeces:
                temp_Y[d] = 1
                
            superclass_Y.append(temp_Y)

        return np.array(superclass_Y)

    def get_Y_superclasses2(self):
        print('Create a new One Hot vector per node')

        def get_D(OHV):
            return set([diagnosis for diagnosis, value in zip(self.Diagnosis, OHV) if value == 1])

        def get_the_index_of_range(i):
            new_ICD_range = []
            for (s,e) in icd_code_ranges:
                if (s,e) != icd_code_ranges[-1]:
                    new_ICD_range.append([int(s), int(e)])
            # ++++++++++++++++++++++++++++++++++++++++++++++
            if i[2] in ['V', 'E']:
                return len(new_ICD_range)
            else:
                for item in new_ICD_range:
                    if int(i[2:]) >= item[0] and int(i[2:]) <= item[1]:
                        return new_ICD_range.index(item)

        # =============================================================
        superclass_Y = []
        Y = []
        for v in self.Nodes:
            if v[0]=='C':
                temp_Y = []
                for u_visit in self.HG.neighbors(v):
                    f = [0] * len(F)
                    for u in self.HG.neighbors(u_visit):
                        if u[0] in ['D']:
                            f[F_indeces[u]] = 1
                    temp_Y.append(f)
                
                Y.append(and_over_rows(temp_Y))
            else:
                Y.append([0] * len(icd_code_ranges))


