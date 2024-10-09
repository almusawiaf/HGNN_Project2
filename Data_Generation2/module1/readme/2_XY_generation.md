# XY Preparation for Heterogeneous Graphs

This project contains the `XY_preparation` class, which processes a heterogeneous graph (HG) to generate feature matrices `X` and target matrices `Y`. The graph consists of various medical entities, and this class extracts and prepares the data in a format suitable for machine learning models.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
  - [Main Functions](#main-functions)
- [Graph Components](#graph-components)
- [License](#license)

## Introduction

The `XY_preparation` class is designed to work with a pre-generated heterogeneous graph (HG) that includes patients, visits, diagnoses, medications, procedures, lab tests, and microbiology results. It prepares two feature matrices `X` (features) and `Y` (diagnosis labels) for each patient and visit. The matrices are processed to remove columns with only one class, ensuring meaningful feature sets for analysis.


## Usage

To generate the feature and label matrices, initialize the `XY_preparation` class by providing the heterogeneous graph (HG) as input. The class processes the graph and returns the feature matrix `X` and target matrix `Y`.

### Inputs

- **HG (Heterogeneous Graph)**: A NetworkX graph object that contains the following types of nodes:
  - Patients (`C`)
  - Visits (`V`)
  - Diagnoses (`D`)
  - Medications (`M`)
  - Procedures (`P`)
  - Lab Tests (`L`)
  - Microbiology Tests (`B`)

### Outputs

The class generates two matrices:
1. **X**: The feature matrix, representing relationships between nodes (e.g., procedures, medications) for each patient or visit.
2. **Y**: The binary label matrix, representing diagnoses across all visits or patient-level data.

Additionally, the class filters out columns that only contain one class (i.e., columns that are either all zeros or all ones), ensuring that only meaningful features are kept.

### Main Functions

Here are the main functions used in the code:

- **`__init__(HG)`**: Initializes the class with a heterogeneous graph (HG) and generates the feature and label matrices for both patient and visit levels.
- **`get_X()`**: Generates the feature matrix `X` for all nodes in the graph (patient level).
- **`get_Y()`**: Generates the binary diagnosis matrix `Y` for all nodes in the graph (patient level).
- **`get_X_visit_level()`**: Generates the feature matrix `X` for visits only (visit level).
- **`get_Y_visit_level()`**: Generates the binary diagnosis matrix `Y` for visits only (visit level).
- **`remove_one_class_columns(Y)`**: Removes columns in the matrix `Y` that contain only one class (either all 0s or all 1s) to reduce noise and improve learning.

## Graph Components

The heterogeneous graph (HG) consists of the following components:

- **Patients (C)**: Nodes representing patients.
- **Visits (V)**: Nodes representing hospital admissions.
- **Diagnoses (D)**: Nodes representing diagnoses based on ICD9 codes.
- **Medications (M)**: Nodes representing prescribed medications.
- **Procedures (P)**: Nodes representing medical procedures.
- **Lab Tests (L)**: Nodes representing lab test results.
- **Microbiology Tests (B)**: Nodes representing microbiology test results.

Each node is identified by a specific letter, and edges represent relationships between these nodes, such as a patient and their corresponding visit, or a visit and a diagnosis.