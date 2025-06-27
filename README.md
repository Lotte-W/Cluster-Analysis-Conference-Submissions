# Document Clustering and Analysis Tool
This project performs text clustering, term extraction, and semantic grouping of documents using [TF-IDF vectorization](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and hierarchical clustering. It is designed for analyzing a collection of text documents (e.g., conference paper submissions), identifying top terms, and grouping similar documents based on content.

## Project structure </br>
* **stop_words.py** </br>
Contains the list of stop words </br>
* **Clustering.py** </br>
Main script (provided code) </br>
* **README.md** </br>
Project documentation (this file) </br>

## Output code </br>
* **cluster_report.txt** </br>
Auto-generated: Top terms for each document </br>
* **group_explanations.txt** </br>
Auto-generated: Summary of document clusters </br>

## What does it do?
* TF-IDF vectorization with custom stop word preprocessing
* Top term extraction per document
* Hierarchical clustering with dendrogram visualization
* Document grouping with explainable term summaries
* Generates reports for both individual documents and grouped clusters

## Customisation
| Parameter  | Location | Purpose |
| ------------- | ------------- |  ------------- |
| max_features=100  | TfidVectorizer  | Limit top N features from corpus |
| top_n=10  | get_top_terms() | Number of key terms to extract per document |
| num_groups=8 | group_documents() | Number of output groups. How many sessions do you want to have? |
| group_size=6 | group_documents() | Number of documents per group. How many submissions do you want to have for each session? |

## Example
### Conference submission abstracts
Folder filled with .txt files, each .txt file was the abstract of a submission for a conference. 50 in total. The dendogram showed that submissions 281 and 296 were close together and would go well together in a session.

**Output cluster_report.txt:** </br>
Document: 281 </br>
Top terms: one, national, 2023, services, data, model, service, platform, carbon, footprint </br>
</br>

Document: 296 </br>
Top terms: preserved, data, computer, long, term, activities, cloud, storage, footprint, carbon </br>
</br>

**output group_explanations.txt** </br>
Group 8: </br>
Documents: 262, 290, 317, 281, 296, 187, 213, 225 </br>
Common terms: org, https, data, cloud, records, first, related, years, one, metadata </br>


