# Document Clustering and Analysis Tool
This project performs text clustering, term extraction, and semantic grouping of documents using TF-IDF vectorization and hierarchical clustering. It is designed for analyzing a collection of text documents (e.g., conference paper submissions), identifying top terms, and grouping similar documents based on content.

Project structure
├── stop_words.py              # Contains the list of stop words </br>
├── cluster_report.txt         # Auto-generated: Top terms for each document
├── group_explanations.txt     # Auto-generated: Summary of document clusters
├── Clustering.py              # Main script (provided code)
└── README.md                  # Project documentation (this file)

## What does it do?
* TF-IDF vectorization with custom stop word preprocessing
* Top term extraction per document
* Hierarchical clustering with dendrogram visualization
* Document grouping with explainable term summaries
* Generates reports for both individual documents and grouped clusters

