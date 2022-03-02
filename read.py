from numpy import row_stack
import pandas as pd
import yake
kw_extr = yake.KeywordExtractor()

df1 = pd.read_excel("M://Gianluca/Desktop/msg.xlsx")
for index, row in df1.iterrows():
    content = row['Commits']
    max_ngram_size = 2
    deduplication_threshold = 0.9
    numOfKeywords = 1
    custom_kw_extractor = yake.KeywordExtractor( n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(content)
    for kw in keywords:
        print(kw[0])
