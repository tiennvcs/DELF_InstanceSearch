# instance_search
# instance_search

## Dataset

- Oxford 5k.
- Paris 6k.
- Google Landmark (optional).

## Pipeline

- <b>Feature extraction</b>
    + Input: the file contain list of image paths.
    + Output: the folder contain list of feature files with <b><i>.delf</i></b> extension.

- <b>Query processing</b>
    + Input: selected image.
    + Output: The <i>file query_extracted_feature.delf</i> representation for image query.

- <b>Similarity evaluation</b>
    + Input: Calculate the similarity of query feature and feature database.
    + Output: List of <b>k</b> elements rank by similar measure.

- <b>Reranking by RANSAC</b>
    + Input: the list of top <i>k</i> features from <b>similar evaluation</b> step and the feature image.
    + Output: top k of ranking images.

## Demo on web
