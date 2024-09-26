'''
generate_HG is class used to generate the 203 heterogeneous graph only.
We only included patients with diagnoses.
'''
import pandas as pd
import networkx as nx
from copy import deepcopy

'''NO DEMOGRAPHICS INFORMATION ADDED'''

class Generate_HG:
    
    def __init__(self, sampling = False, num_Patients = 500):

        self.sampling = sampling
        self.num_Patients = num_Patients
        self.folder_path = '/lustre/home/almusawiaf/PhD_Projects/MIMIC_resources'
        # self.folder_path = '/home/almusawiaf/MyDocuments/PhD_Projects/Data/MIMIC_resources'
        
        
        print('Loading the dataframes...')
        
        df_Patients, new_Diagnosis, new_Prescriptions, new_Procedures, new_LabTest, new_MicroBio = self.load_patients_data()

        new_LabTest2 = self.split_lab_test(new_LabTest)
        
        print('Extracting bipartite networks...')

        self.HG = nx.Graph()

        self.get_Bipartite(new_Diagnosis,    'SUBJECT_ID', 'HADM_ID', 'C', 'V', 'Visits')
        self.get_Bipartite(new_Diagnosis,    'HADM_ID', 'ICD9_CODE',  'V', 'D', 'Diagnosis')
        self.get_Bipartite(new_Procedures,   'HADM_ID', 'ICD9_CODE', 'V', 'P', 'Procedures')
        self.get_Bipartite(new_Prescriptions,'hadm_id', 'drug', 'V', 'M', 'Medications')
        self.get_Bipartite(new_LabTest2,     'HADM_ID', 'ITEMID_FLAG', 'V', 'L', 'Lab tests')
        self.get_Bipartite(new_MicroBio,     'HADM_ID', 'SPEC_ITEMID', 'V', 'B', 'MicroBiology tests')

        G_statistics(self.HG)
        self.HG = self.selecting_top_labs()
      
        G_statistics(self.HG)
        
        self.HG = self.remove_isolated_nodes()        

        Nodes = list(self.HG.nodes())

        self.Patients =    [v for v in Nodes if v[0]=='C']
        self.Visits =      [v for v in Nodes if v[0]=='V']
        self.Medications = [v for v in Nodes if v[0]=='M']
        self.Diagnosis  =  [v for v in Nodes if v[0]=='D']
        self.Procedures =  [v for v in Nodes if v[0]=='P']
        self.Labs       =  [v for v in Nodes if v[0]=='L']
        self.MicroBio   =  [v for v in Nodes if v[0]=='B']
        self.Nodes = Nodes
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    def get_Bipartite(self, DF, id1, id2, c1, c2, msg):
        '''DF: dataframe, id1: row1, id2: row2, c1, c2: node code'''  
        print(f'\nExtracting and adding data of {msg}')
        print(id1, id2, c1, c2)
        print(DF.head(5))

        DF2    = self.getDict2(DF,  id1, id2, c1, c2)
        edges  = self.getEdges(DF2, id1, id2)
        self.HG.add_edges_from(edges)
    
    def split_lab_test(self, lab_df):
        print('Splitting lab tests')
        # Step 1: Fill NaN values in the 'FLAG' column with 'normal'
        lab_df['FLAG'] = lab_df['FLAG'].fillna('normal')
        
        # Step 2: Remove rows where 'FLAG' equals 'delta'
        lab_df = lab_df[lab_df['FLAG'] != 'delta']
        
        # Step 3: Create a new DataFrame with HADM_ID and a concatenated column 'itemid_flag'
        # Concatenate 'ITEMID' and 'FLAG' as strings
        lab_df.loc[:, 'ITEMID_FLAG'] = lab_df['ITEMID'].astype(str) + '_' + lab_df['FLAG'].astype(str)

        
        # Create the new DataFrame with 'HADM_ID' and the concatenated 'itemid_flag' column
        new_df = lab_df[['HADM_ID', 'ITEMID_FLAG']].copy()
        
        # Display the resulting DataFrame
        print(new_df.head(10))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        return new_df

    def remove_isolated_nodes(self):
        print('Removing isolated nodes')
        isolated_nodes = [v for v in self.HG.nodes() if self.HG.degree(v)==0]
        G_statistics(self.HG)    
        new_HG = remove_patients_and_linked_visits(isolated_nodes, self.HG)
        G_statistics(new_HG)    
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        return deepcopy(new_HG)


    def extract3(self, code):
        return str(code)[:3]
    
    def extract2(self, code):
        return str(code)[:2]
    
    def load_patients_data(self):
        import random
        # 1. read the prescreption file...
        
        # Loading the data
        df_Medications   = pd.read_csv(f'{self.folder_path}/PRESCRIPTIONS.csv')
        df_Patients      = pd.read_csv(f'{self.folder_path}/PATIENTS.csv')
        df_DiagnosisICD  = pd.read_csv(f'{self.folder_path}/DIAGNOSES_ICD.csv')    # Diagnosis!
        df_ProceduresICD = pd.read_csv(f'{self.folder_path}/PROCEDURES_ICD.csv')    # Procedures!
        df_labs          = pd.read_csv(f'{self.folder_path}/LABEVENTS.csv')    # Lab test!
        df_microbio      = pd.read_csv(f'{self.folder_path}/MICROBIOLOGYEVENTS.csv')    # Microbiology!
        
        # Handling missing values upfront (dropping rows with missing important columns)
        df_DiagnosisICD.dropna(subset=['HADM_ID', 'ICD9_CODE'], inplace=True)
        df_ProceduresICD.dropna(subset=['ICD9_CODE'], inplace=True)
        df_Medications.dropna(subset=['drug'], inplace=True)
        df_labs.dropna(subset=['ITEMID'], inplace=True)
        df_microbio.dropna(subset=['SPEC_ITEMID'], inplace=True)
        
        # Extract unique visits and patients from the diagnosis DataFrame
        visits = df_DiagnosisICD['HADM_ID'].unique()
        patients = df_DiagnosisICD['SUBJECT_ID'].unique()

        if self.sampling:
            print('\nWe are SAMPLING\n')
            patients = random.sample(list(patients), self.num_Patients)

        
        # Filtering the data for selected patients and visits
        print('Use the patients inside the new DataFrame....')
        new_Patients = df_Patients[df_Patients['SUBJECT_ID'].isin(patients)].copy()
        new_Diagnosis = df_DiagnosisICD[df_DiagnosisICD['SUBJECT_ID'].isin(patients)].copy()
        new_Procedures = df_ProceduresICD[df_ProceduresICD['SUBJECT_ID'].isin(patients)].copy()
        new_Medication = df_Medications[df_Medications['subject_id'].isin(patients)].copy()
        new_LabTest = df_labs[df_labs['SUBJECT_ID'].isin(patients)].copy()
        new_MicroBio = df_microbio[df_microbio['SUBJECT_ID'].isin(patients)].copy()
        
        print('Dropping NaN visits')
        new_Diagnosis.dropna(subset=['HADM_ID'], inplace=True)
        new_Procedures.dropna(subset=['HADM_ID'], inplace=True)
        new_Medication.dropna(subset=['hadm_id'], inplace=True)
        new_LabTest.dropna(subset=['HADM_ID'], inplace=True)
        new_MicroBio.dropna(subset=['HADM_ID'], inplace=True)
    
        new_Diagnosis['ICD9_CODE']  = new_Diagnosis['ICD9_CODE'].apply(self.extract3)
        new_Procedures['ICD9_CODE'] = new_Procedures['ICD9_CODE'].apply(self.extract2)
        # ----------------------------------------------------------------------------
        
        diag_frequency = new_Diagnosis['ICD9_CODE'].value_counts().head(203).index.tolist()        
        new_Diagnosis  = new_Diagnosis[new_Diagnosis['ICD9_CODE'].isin(diag_frequency)]        
        
        # ----------------------------------------------------------------------------
        # extracting the unique sets of nodes of diff category.
        Procedures = sorted(new_Procedures['ICD9_CODE'].unique())
        Medication = sorted(new_Medication['drug'].unique())
        Diagnosis  = new_Diagnosis['ICD9_CODE'].unique()
        LabTests   = new_LabTest['ITEMID'].unique()
        MicroBio   = new_MicroBio['SPEC_ITEMID'].unique()
    
        print('General Information:\n---------------------------')
        print(f'Number of Patients = {len(patients)}')
        print(f'Number of Diagnosis = {len(Diagnosis)}')
        print(f'Number of procedures = {len(Procedures)}')
        print(f'Number of Medication = {len(Medication)}')
        print(f'Number of Lab tests  = {len(LabTests)}')
        print(f'Number of MicroBio   = {len(MicroBio)}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        return new_Patients, new_Diagnosis, new_Medication, new_Procedures, new_LabTest, new_MicroBio



    def selecting_top_labs(self):
        new_HG = deepcopy(self.HG)
        Nodes = new_HG.nodes()
        node_degrees = {n: new_HG.degree(n) for n in Nodes if n[0] == 'L'}
        top_nodes = dict(sorted(node_degrees.items(), key=lambda item: item[1], reverse=True)[:480])
        labs_to_delete = [n for n in node_degrees if n not in top_nodes]
        new_HG.remove_nodes_from(labs_to_delete)
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        return new_HG
            

    def getDict2(self, df, id1, id2, c1, c2):
        # Create a copy of the relevant columns
        new_df = df[[id1, id2]].copy()
        
        # Drop rows with NaN values in either id1 or id2
        new_df = new_df.dropna(subset=[id1, id2])
        
        # Explicitly cast columns to string to avoid dtype compatibility issues
        new_df[id1] = new_df[id1].astype(str)
        new_df[id2] = new_df[id2].astype(str)
        
        # Add the prefixes to each column after ensuring there are no NaNs
        new_df.loc[:, id1] = c1 + '_' + new_df[id1]
        new_df.loc[:, id2] = c2 + '_' + new_df[id2]
        
        # Remove duplicate rows
        new_df = new_df.drop_duplicates()
        
        return new_df

    def getEdges(self, data, id1, id2):
        # Check if data is a DataFrame and extract edges accordingly
        if isinstance(data, pd.DataFrame):
            # Extract edges from the DataFrame
            EdgesList = list(data[[id1, id2]].itertuples(index=False, name=None))
        else:
            # Assuming data is a list of dictionaries
            EdgesList = [(d[id1], d[id2]) for d in data]
        
        return EdgesList


def G_statistics(G):
    Nodes = list(G.nodes())

    Patients =    [v for v in Nodes if v[0]=='C']
    Visits =      [v for v in Nodes if v[0]=='V']
    Medications = [v for v in Nodes if v[0]=='M']
    Diagnosis  =   [v for v in Nodes if v[0]=='D']
    Procedures =  [v for v in Nodes if v[0]=='P']
    Labs       =  [v for v in Nodes if v[0]=='L']
    MicroBio   =  [v for v in Nodes if v[0]=='B']
    

    print(f'number of patients = {len(Patients)}')
    print(f'number of visits = {len(Visits)}')
    print(f'number of Medication = {len(Medications)}')
    print(f'number of Diagnoses = {len(Diagnosis)}')
    print(f'number of Procedures = {len(Procedures)}')
    print(f'number of Labs = {len(Labs)}')
    print(f'number of MicoBio = {len(MicroBio)}')
    
    print(f'number of Edges = {G.number_of_edges()}')
    
    print('------------------------------------------\n')

def remove_patients_and_linked_visits(nodes, HG):
    '''remove patients and their visits from HG'''
    print('Number of PATIENTS to remove: ', len(nodes))
    
    new_HG = deepcopy(HG)
    nodes_to_remove = nodes
    for node in nodes:
        for v in HG.neighbors(node):
            if v[0]=='V':
                nodes_to_remove.append(v)
        nodes_to_remove.append(node)
    print('Number of nodes to remove: ', len(nodes_to_remove))
    new_HG.remove_nodes_from(nodes_to_remove)
    return new_HG     


