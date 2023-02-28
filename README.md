# Ironhack_final_project_automatic-media-screening-for-aml-failures

Within transaction monitoring and CDD it is a frequent task to conduct an OSINT search on legal entities with regards to possible ML/TF related failures. This is a repetitive and highly manual process.
The script in this repository conducts this search for the investigator based on Natural Language Processing (NLP).
The Support Vector Classifier (SVC) model was trained on a dataset of 302 news articles about the four companies that are involved in some of the biggest AML failures in history (Wachovia, HSBC, BCCI, and Benex). Within this dataset, the articles are manually labelled as 'relevant' or 'not relevant' for a negative media screening. 

The .py-script makes a call to SerpAPI, an API used for retrieving Google-Search results. The search queue is pre-defined and simply requires the investigator to enter the name of the legal entity. After, the search is conducted, the news title and google-snippet is classified as 'relevant' or 'not relevant' using the previously trained SVC model. Then, a .csv file is stored locally containing only those results classified as relevant due to their title, the snippet, or both. The file yields the title, the snippet, web links to the article, and the regarding classifications. After, the script asks the investigator whether they wish to automatically open the web links leading to the relevant articles. 
In case no relevant results are returned, the script will inform the investigator and ask them to manually conduct the screening to double check.

The repository contains:
- a script folder including the script and .pkl files for applying the model
- a model folder including the tarining data and the .ipynb file in which the training took place
- a requirements.txt file
