# Heterogeneous Graph Generator for Patient Data

This project generates a heterogeneous graph based on patient data extracted from multiple CSV files. The generated graph represents various relationships between patients, their visits, diagnoses, medications, procedures, lab tests, and microbiology results.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
  - [Main Functions](#main-functions)
- [Graph Components](#graph-components)

## Introduction

This repository contains a Python class `Generate_HG` that constructs a heterogeneous graph (HG) using patient data from CSV files, including diagnosis, prescriptions, procedures, lab tests, and microbiology tests. The graph is built using the NetworkX library and contains bipartite networks between visits and various medical entities. Only patients with available diagnoses are included.

## Usage

To generate the heterogeneous graph, you need to initialize the `Generate_HG` class by providing the folder path where your data files are stored. The class will read the CSV files and process them to build the graph.

### Inputs

The CSV files must be placed in the specified folder path and should include the following files:

- `PRESCRIPTIONS.csv`: Contains medication prescriptions.
- `DIAGNOSES_ICD.csv`: Contains diagnosis information with ICD9 codes.
- `PROCEDURES_ICD.csv`: Contains medical procedure information with ICD9 codes.
- `LABEVENTS.csv`: Contains lab test results.
- `MICROBIOLOGYEVENTS.csv`: Contains microbiology test results.

Ensure that the files contain the appropriate columns mentioned in the code, such as `HADM_ID`, `SUBJECT_ID`, `ICD9_CODE`, and other relevant columns for proper graph generation.

### Outputs

The class generates a heterogeneous graph with the following node types:

- **Patients (C)**: Nodes representing patients.
- **Visits (V)**: Nodes representing hospital admissions.
- **Diagnoses (D)**: Nodes representing diagnoses with ICD9 codes.
- **Medications (M)**: Nodes representing prescribed medications.
- **Procedures (P)**: Nodes representing medical procedures.
- **Lab Tests (L)**: Nodes representing lab test results.
- **Microbiology Tests (B)**: Nodes representing microbiology tests.

Edges represent the relationships between these nodes, such as a visit and its corresponding diagnosis, procedure, or medication.

### Main Functions

Here are the main functions used in the code:

- **`__init__(folder_path)`**: Initializes the class and loads patient data from the specified folder.
- **`get_Bipartite(DF, id1, id2, c1, c2, msg)`**: Extracts bipartite networks from the provided patient data (e.g., between visits and diagnoses).
- **`load_patients_data()`**: Loads patient data from CSV files, handles missing values, processes lab tests, and creates filtered datasets for analysis.
- **`remove_isolated_nodes()`**: Removes isolated nodes (e.g., patients and their associated visits) from the graph that do not contribute to the network.
- **`selecting_top_labs()`**: Filters the graph to retain only the top 480 lab tests by degree (most frequently linked tests).
- **`update_statistics()`**: Updates and categorizes nodes in the graph (e.g., separating patients, visits, diagnoses, medications, etc.).

## Graph Components

The heterogeneous graph includes the following components:

- **Patients (C)**: Nodes representing patients.
- **Visits (V)**: Nodes representing visits (hospital admissions).
- **Diagnoses (D)**: Nodes representing diagnoses based on ICD9 codes.
- **Medications (M)**: Nodes representing prescribed medications.
- **Procedures (P)**: Nodes representing medical procedures.
- **Lab Tests (L)**: Nodes representing lab test results.
- **Microbiology Tests (B)**: Nodes representing microbiology test results.

Each node is represented by a letter code, and edges represent the connections between different entities such as a visit and a diagnosis, or a visit and a procedure.
