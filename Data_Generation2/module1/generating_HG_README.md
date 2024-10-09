# Heterogeneous Graph Generator for Patient Data

This project generates a heterogeneous graph based on patient data extracted from multiple CSV files. The generated graph represents various relationships between patients, their visits, diagnoses, medications, procedures, lab tests, and microbiology results.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
  - [Main Functions](#main-functions)
- [Graph Components](#graph-components)
- [License](#license)

## Introduction

This repository contains a Python class `Generate_HG` that constructs a heterogeneous graph (HG) using patient data from diagnosis, prescriptions, procedures, lab tests, and microbiology test CSV files. The graph is created using the NetworkX library and contains bipartite networks between visits and other medical entities. Only patients with diagnoses are included.

## Requirements

To run the code, the following packages must be installed:

- Python 3.x
- pandas
- networkx

You can install the required packages using pip:

```bash
pip install pandas networkx

## Installation

Clone the repository or download the code files.
Ensure that the input CSV files are available in a specified folder.
Install the required Python libraries as described in the Requirements section.

## Usage
To generate the heterogeneous graph, you need to initialize the Generate_HG class by providing the folder path where your data files are stored. The class will read the CSV files and process them to build the graph.

### Inputs
CSV files must be placed in the specified folder path and should include the following files:
PRESCRIPTIONS.csv
DIAGNOSES_ICD.csv
PROCEDURES_ICD.csv
LABEVENTS.csv
MICROBIOLOGYEVENTS.csv
### Outputs
A heterogeneous graph is generated with the following node types:
Patients (C)
Visits (V)
Diagnoses (D)
Medications (M)
Procedures (P)
Lab tests (L)
Microbiology tests (B)
The graph edges represent the relationships between these nodes, e.g., a visit and its corresponding diagnosis, procedure, etc.
### Main Functions
__init__(folder_path): Initializes the class and loads patient data from the given folder.
get_Bipartite(DF, id1, id2, c1, c2, msg): Extracts bipartite networks from patient data.
load_patients_data(): Reads patient data from CSV files, handles missing values, and processes lab tests.
remove_isolated_nodes(): Removes isolated nodes (patients and their associated visits).
selecting_top_labs(): Filters the graph to include only the top 480 lab tests by degree.
update_statistics(): Updates and categorizes nodes in the graph.

## Graph Components
The heterogeneous graph includes the following components:

Patients (C): Nodes representing patients.
Visits (V): Nodes representing visits (hospital admissions).
Diagnoses (D): Nodes representing diagnoses.
Medications (M): Nodes representing prescribed medications.
Procedures (P): Nodes representing medical procedures.
Lab Tests (L): Nodes representing lab test results.
Microbiology Tests (B): Nodes representing microbiology tests.
