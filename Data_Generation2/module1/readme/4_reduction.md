# Edge Reduction for Similarity Matrices

This project provides a class `Reduction` to process a list of similarity matrices. The goal is to select the highest weighted edges from these matrices, convert them into an edge list, and save the final edge weights per matrix.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
  - [Main Functions](#main-functions)
- [Edge Selection Process](#edge-selection-process)

## Introduction

The `Reduction` class reads a list of similarity matrices, selects the highest weighted edges, and stores the results as edge lists with corresponding weights. It can also handle patient similarity graphs (PSGs) and integrate them with the base similarity matrices for further processing.

## Usage

To perform edge reduction, initialize the `Reduction` class by providing the base path and, optionally, whether to include patient similarity graphs (PSGs). The results are saved in a specified folder.

### Inputs

- **Base Path**: The root directory where similarity matrices are stored.
- **Similarity Matrices**: The matrices are read from the specified folder (`/HGNN_data/As`).
- **Patient Similarity Graphs (PSGs)**: Optionally, additional similarity matrices from the PSGs can be included.

### Outputs

- **Edge List**: A list of unique edges selected from the highest weighted edges across all matrices.
- **Edge Weights**: The weights corresponding to the selected edges are saved for each matrix.

### Main Functions

Here are the main functions used in the code:

- **`__init__(base_path, gpu, PSGs)`**: Initializes the class, reads similarity matrices, and selects the highest weighted edges.
- **`read_Ws(saving_path, folder_name)`**: Reads the similarity matrices from the given path and folder.
- **`read_PSGs(base_path)`**: Reads the patient similarity graphs (PSGs) from the specified base path.
- **`get_edges_dict(the_path)`**: Converts the sparse matrix from the specified path into a dictionary of edges and weights.
- **`selecting_high_edges(Ws)`**: Selects the highest weighted edges from the similarity matrices.
- **`final_Ws_with_unique_list_of_edges(D, saving_path)`**: Creates and saves the final edge list and edge weights for each matrix.
- **`keep_top_million(edge_dict, top_n)`**: Keeps only the top N edges with the highest weights from the edge dictionary.

## Edge Selection Process

The edge reduction process follows these steps:

1. **Reading Matrices**: The similarity matrices are read from the specified folder.
2. **Selecting High Weighted Edges**: The highest weighted edges from each matrix are selected. If a matrix contains more than 1,000,000 edges, only the top 1,000,000 edges are kept.
3. **Generating Unique Edge List**: A unique list of edges is generated from all matrices.
4. **Saving Edge Weights**: For each matrix, the weights of the selected edges are saved to a separate file.

This approach ensures that only the most significant edges (based on their weights) are retained for further processing.
