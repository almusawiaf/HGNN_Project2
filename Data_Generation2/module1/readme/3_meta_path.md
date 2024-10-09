# Meta-Path Similarity Generator for Heterogeneous Graphs

This project provides a class `Meta_path` to calculate meta-path-based similarities for a heterogeneous graph (HG). The similarities are based on various meta-paths and similarity metrics, including Path-Count (PC) and Symmetric PathSim (SPS). The results are saved in sparse matrix format to the specified saving path.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
  - [Main Functions](#main-functions)
- [Meta-Path-Based Similarities](#meta-path-based-similarities)

## Introduction

The `Meta_path` class calculates different meta-path-based similarity metrics for nodes in a heterogeneous graph. It supports two similarity metrics:
1. **Path-Count (PC)**: Basic path count between two nodes over the meta-path.
2. **Symmetric PathSim (SPS)**: Symmetric similarity measure that balances the contribution of meta-paths.

The class extracts adjacency matrices from the graph and computes various similarities between nodes based on patients, visits, medications, diagnoses, procedures, lab tests, and microbiology tests.

## Usage

To generate the meta-path-based similarities, initialize the `Meta_path` class by providing the heterogeneous graph (HG) and the desired similarity type (`PC` or `SPS`). The results will be saved in the specified path as sparse matrices.

### Inputs

- **HG (Heterogeneous Graph)**: A NetworkX graph object that contains nodes representing:
  - Patients (`C`)
  - Visits (`V`)
  - Medications (`M`)
  - Diagnoses (`D`)
  - Procedures (`P`)
  - Lab Tests (`L`)
  - Microbiology Tests (`B`)

### Outputs

The class generates and saves several sparse matrices representing various meta-path-based similarities between nodes, such as:
- **Patient-Medication**
- **Patient-Diagnosis**
- **Patient-Procedure**
- **Patient-Lab**
- **Patient-MicroBiology**

The results are saved in `.npz` format under the specified `saving_path`.

### Main Functions

Here are the main functions used in the code:

- **`__init__(HG, similarity_type, saving_path)`**: Initializes the class with a heterogeneous graph (HG), the type of similarity to calculate (`PC` or `SPS`), and the path where results will be saved.
- **`save_sparse(A, i)`**: Saves a sparse matrix `A` to disk as a `.npz` file.
- **`load_sparse(i)`**: Loads a sparse matrix from disk using its index `i`.
- **`M(W1, W2, msg, t)`**: Multiplies two matrices `W1` and `W2` in parallel and saves the result as a sparse matrix.
- **`subset_adjacency_matrix(subset_nodes)`**: Extracts a sub-adjacency matrix for a subset of nodes from the graph.
- **`symmetricPathSim_3(PC, Nodes, selected_nodes)`**: Calculates Symmetric PathSim (SPS) for nodes based on a given path count (PC) matrix and selected nodes.

## Meta-Path-Based Similarities

The class computes similarities for various combinations of nodes in the graph. Some key meta-path-based similarities include:

- **Patient-Medication Similarity**
- **Patient-Diagnosis Similarity**
- **Patient-Procedure Similarity**
- **Diagnosis-Medication Similarity**
- **Procedure-Lab Similarity**
- **Visit-MicroBiology Similarity**
  
In addition, homogeneous similarities between patients and visits are calculated using these meta-paths.

