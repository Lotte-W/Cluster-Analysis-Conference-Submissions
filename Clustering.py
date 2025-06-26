import re
import nltk
import os
import numpy as np

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt

# Load stop words from stop_words.py
stop_words_path = r"stop_words.py"
stop_words = []
with open(stop_words_path, 'r') as f:
    exec(f.read(), globals())

# Function to tokenize and process stop words similarly to TfidfVectorizer
def preprocess_stop_words(stop_words, vectorizer):
    processed_stop_words = set()
    for word in stop_words:
        tokenized_word = vectorizer.build_tokenizer()(word)
        processed_stop_words.update(tokenized_word)
    return list(processed_stop_words)

# Initialize vectorizer to preprocess stop words
temp_vectorizer = TfidfVectorizer()
processed_stop_words = preprocess_stop_words(stop_words, temp_vectorizer)

# Load texts from the corpus folder
ignoreFiles = set([".DS_Store", "LICENSE", "README.md"])

submissionTexts = []
submissionTitles = []
for root, dirs, files in os.walk(r"[PATH HERE]"): #[PATH HERE]: fill in the path to the .txt files you want clustered
    for filename in files:
        if filename not in ignoreFiles:
            with open(os.path.join(root, filename)) as rf:
                print(filename)
                submissionTexts.append(rf.read().lower())
                submissionTitles.append(filename[:-4].lower())

# Get the frequencies of the 100 most common ngrams in the corpus. You can adjust the number to your own liking.
vectorizer = TfidfVectorizer(max_features=100, stop_words=processed_stop_words)
tfidf_matrix = vectorizer.fit_transform(submissionTexts)

# Measure distances
similarity = euclidean_distances(tfidf_matrix)

# Check if the matrix is symmetric and non-negative hollow
if np.allclose(similarity, similarity.T) and np.all(np.diag(similarity) == 0):
    # Convert to condensed distance matrix
    condensed_similarity = squareform(similarity, checks=False)
else:
    condensed_similarity = similarity

# Hierarchical clustering using Ward's method
linkages = linkage(condensed_similarity, 'ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkages, labels=submissionTitles, orientation="right", leaf_font_size=8, leaf_rotation=45)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tight_layout()
plt.show()

# Function to get the top n terms, ensuring terms are present in the document
def get_top_terms(tfidf_matrix, feature_names, submissionTexts, top_n=10):
    top_terms = []
    for doc_index in range(tfidf_matrix.shape[0]):
        row_data = tfidf_matrix[doc_index].toarray().flatten()
        top_indices = row_data.argsort()[::-1]  # Sorted indices in descending order of tf-idf scores
        present_terms = [feature_names[i] for i in top_indices if feature_names[i] in submissionTexts[doc_index]]
        
        # Ensure we have exactly top_n terms, all present in the document
        top_terms.append(present_terms[:top_n])
    return top_terms

# Extract feature names
feature_names = vectorizer.get_feature_names_out()

# Get top terms for each document, ensuring terms are present in the document
top_terms_per_doc = get_top_terms(tfidf_matrix, feature_names, submissionTexts, top_n=10)

# Generate cluster report
cluster_report = {}
for i, title in enumerate(submissionTitles):
    cluster_report[title] = top_terms_per_doc[i]

# Print report
for title, terms in cluster_report.items():
    print(f"Document: {title}")
    print(f"Top terms: {', '.join(terms)}")
    print("\n")

# Save report to a file
with open('cluster_report.txt', 'w') as report_file:
    for title, terms in cluster_report.items():
        report_file.write(f"Document: {title}\n")
        report_file.write(f"Top terms: {', '.join(terms)}\n")
        report_file.write("\n")

# Group documents into clusters based on hierarchical clustering results
def group_documents(linkages, titles, num_groups=8, group_size=6):
    clusters = fcluster(linkages, t=num_groups, criterion='maxclust')
    grouped_docs = {i: [] for i in range(1, num_groups + 1)}
    
    for doc_index, cluster_id in enumerate(clusters):
        grouped_docs[cluster_id].append(titles[doc_index])

    # Merge smaller clusters if needed to form exactly `num_groups` groups of `group_size`
    merged_groups = []
    
    current_group = []
    for cluster_id in grouped_docs:
        current_group.extend(grouped_docs[cluster_id])
        while len(current_group) >= group_size:
            merged_groups.append(current_group[:group_size])
            current_group = current_group[group_size:]

    if len(merged_groups) < num_groups:
        # Append the remaining documents to the last group if there are less than `num_groups` groups
        merged_groups[-1].extend(current_group)
    else:
        merged_groups.append(current_group)

    # If there are more than `num_groups` groups, merge the last few groups
    while len(merged_groups) > num_groups:
        merged_groups[-2].extend(merged_groups[-1])
        merged_groups.pop(-1)

    return merged_groups

# Function to explain the selection of documents in each group
def explain_groups(grouped_docs, cluster_report):
    explanations = {}
    for group_id, docs in enumerate(grouped_docs, 1):
        terms_counter = {}
        for doc in docs:
            for term in cluster_report[doc]:
                if term not in terms_counter:
                    terms_counter[term] = 0
                terms_counter[term] += 1
        sorted_terms = sorted(terms_counter.items(), key=lambda item: item[1], reverse=True)
        explanations[group_id] = [term for term, count in sorted_terms[:10]]
    return explanations

# Group documents into clusters of 6
grouped_docs = group_documents(linkages, submissionTitles, num_groups=8, group_size=6)

# Explain the selection of documents in each group
group_explanations = explain_groups(grouped_docs, cluster_report)

# Print group explanations
for group_id, docs in enumerate(grouped_docs, 1):
    print(f"Group {group_id}:")
    print(f"Documents: {', '.join(docs)}")
    print(f"Common terms: {', '.join(group_explanations[group_id])}")
    print("\n")

# Save group explanations to a file
with open('group_explanations.txt', 'w') as file:
    for group_id, docs in enumerate(grouped_docs, 1):
        file.write(f"Group {group_id}:\n")
        file.write(f"Documents: {', '.join(docs)}\n")
        file.write(f"Common terms: {', '.join(group_explanations[group_id])}\n")
        file.write("\n")
