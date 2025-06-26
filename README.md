# Document Clustering and Analysis Tool
This project performs text clustering, term extraction, and semantic grouping of documents using TF-IDF vectorization and hierarchical clustering. It is designed for analyzing a collection of text documents (e.g., conference paper submissions), identifying top terms, and grouping similar documents based on content.

## Project structure </br>
* **stop_words.py** </br>
Contains the list of stop words </br>
* **cluster_report.txt** </br>
Auto-generated: Top terms for each document </br>
* **group_explanations.txt** </br>
Auto-generated: Summary of document clusters </br>
* **Clustering.py** </br>
Main script (provided code) </br>
* **README.md** </br>
Project documentation (this file) </br>

## What does it do?
* TF-IDF vectorization with custom stop word preprocessing
* Top term extraction per document
* Hierarchical clustering with dendrogram visualization
* Document grouping with explainable term summaries
* Generates reports for both individual documents and grouped clusters

