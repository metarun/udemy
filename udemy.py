# 1. Load all the data - COURSES
# 2. Clean the titles and store in new colum [ clean_title ]
# 3. Create vectors for the new clean_title
# 4. for the vectors matrix, calculate cos_simliarity 
# 5. Find a title for which we want to find nearest courses 
# 6. Find the index of the course from COURSES data - TITLE_INDEX
# 7. Find the cos_sim record for TITLE_INDEX and save in - TITLE_COS_SIM as array
# 8. Comvert  TITLE_COS_SIM to a dict TITLE_COS_SIM_DICT
# 9. Sort the Dict based on Value
# 10. Find top 5 values and the key[index]
# 11. Take the index and find title for the index in COURSES

#Import libraries 
import streamlit as st
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

courses = pd.read_csv('/udemy_course_data.csv')
# base_course_name = 'The Complete Chart Pattern Trading Course: A Proven Approach'

courses['clean_title']  = courses['course_title'].apply(nfx.remove_stopwords)
courses['clean_title']  = courses['course_title'].apply(nfx.remove_special_characters)

vectorizer = CountVectorizer()
clean_title_vector = vectorizer.fit_transform(courses['clean_title'])
clean_title_cos_sim = cosine_similarity(clean_title_vector)

def recom(base_course_name):
    name = []
    index_id = (courses[courses.course_title == base_course_name].course_id).index[0]
    title_cos_sim = clean_title_cos_sim[index_id]
    TITLE_COS_SIM_DICT = dict(enumerate(title_cos_sim))
    # use the sorted function with the items method to sort the dictionary by value in reverse order
    TITLE_COS_SIM_DICT_SORTED = dict(sorted(TITLE_COS_SIM_DICT.items(), key=lambda x: x[1], reverse=True))
    no_of_top_records = 6
    # get the first top records
    TOP_SIMILAR_TITLES = (list(TITLE_COS_SIM_DICT_SORTED.items())[:no_of_top_records])

    for key, value in TOP_SIMILAR_TITLES:
        if key != index_id:
           name.append(courses.loc[key].course_title)
        
            # print(f'recommended courses are {name}')
    return(name)


random_indices = random.sample(range(0, len(courses)), 5)
random_rows = courses.iloc[random_indices].course_title.values

st.title('Udemy similar course recommendor')
import streamlit as st

courses['course_title']
base_course_name = st.text_input("Enter a course from above for which you want to find similer course")

st.text('Simlier course are ')
x  = recom(base_course_name)
x


