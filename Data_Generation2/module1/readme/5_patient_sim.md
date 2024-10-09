# Patient Similarity Based on Clinical Data

This project contains the `Patients_Similarity` class, which processes a heterogeneous graph (HG) to measure patient similarities based on different clinical features (e.g., medications, diagnoses, procedures). The class computes cosine similarity between patients, expands the similarity matrix, and saves it as a sparse matrix.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
  - [Main Functions](#main-functions)
- [Similarity Calculation Process](#similarity-calculation-process)

## Introduction

The `Patients_Similarity` class reads a heterogeneous graph (HG) and its associated nodes, creates one-hot vectors (OHVs) for different node types (e.g., medications, diagnoses, procedures), and calculates patient similarity using cosine similarity. The computed similarity matrices are saved as sparse matrices for further analysis.

## Usage

To compute patient similarity, initialize the `Patients_Similarity` class by providing the heterogeneous graph (HG), the nodes, and the path to save the resulting patient similarity graphs (PSGs). The class processes clinical data and saves similarity matrices in the specified path.

### Inputs

- **HG (Heterogeneous Graph)**: A NetworkX graph object that contains nodes representing:
  - Patients (`C`)
  - Visits (`V`)
  - Medications (`M`)
  - Diagnoses (`D`)
  - Procedures (`P`)
  - Lab Tests (`L`)
  - Microbiology Tests (`B`)

- **Nodes**: The list of nodes in the heterogeneous graph.
- **PSGs_path**: The directory path where the patient similarity graphs (PSGs) will be saved.

### Outputs

The class generates and saves patient similarity matrices for each clinical type (e.g., medications, diagnoses, procedures) as sparse `.npz` files. These matrices can be found in the `PSGs` folder under the specified path.

### Main Functions

Here are the main functions used in the code:

- **`__init__(HG, Nodes, PSGs_path)`**: Initializes the class, reads the heterogeneous graph, and processes clinical types to compute patient similarity.
- **`process(Code)`**: Computes cosine similarity for the specified clinical type, expands the similarity matrix, and saves it as a sparse matrix.
- **`save_npz(A, file_name)`**: Saves the computed similarity matrix as a sparse `.npz` file.
- **`expand_A(A)`**: Expands the similarity matrix to include all nodes, filling zeros for non-patient nodes.
- **`get_X(clinical_type)`**: Extracts the one-hot vector (OHV) representation for the specified clinical type for patient nodes.
- **`get_X_sub_case(clinical_type)`**: Retrieves OHV features for specific sub-cases (e.g., gender, expiration flag).

## Similarity Calculation Process

The process for calculating patient similarity follows these steps:

1. **Create One-Hot Vectors (OHVs)**: The clinical data for each patient (e.g., medications, diagnoses, procedures) is transformed into OHV format.
2. **Cosine Similarity**: Cosine similarity is computed between the OHVs of patients for each clinical type.
3. **Matrix Expansion**: The similarity matrix is expanded to match the full set of nodes in the graph, filling zeros for non-patient nodes.
4. **Saving Results**: The similarity matrices are saved as sparse matrices (`.npz` format) in the `PSGs` folder for later use.

This approach allows the efficient computation and storage of patient similarities based on various clinical features.
