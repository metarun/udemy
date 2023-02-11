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

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import streamlit as st

# Load all the data - COURSES
courses = pd.read_csv('udemy_course_data.csv')
base_course_name = 'The Complete Chart Pattern Trading Course: A Proven Approach'

# Clean the titles and store in new colum [ clean_title ]
courses['clean_title'] = courses['course_title'].apply(nfx.remove_stopwords)
courses['clean_title'] = courses['course_title'].apply(nfx.remove_special_characters)

# Let Countv not to tokenize below words since they no real meaning and even sometime creates problem in outcome

stop_words = ["Complete", "complete","Course","course", "First", "first"]


# Create vectors for the new clean_title
vectorizer = CountVectorizer(stop_words=stop_words)
clean_title_vector = vectorizer.fit_transform(courses['clean_title'])

# For the vectors matrix, calculate cos_simliarity 
clean_title_cos_sim = cosine_similarity(clean_title_vector)

def recom(base_course_name):
    # Find a title for which we want to find nearest courses 
    name = []
    if base_course_name not in courses['course_title'].values:
        return "Sorry, the course you entered does not exist in the data."

    # Find the index of the course from COURSES data - TITLE_INDEX
    index_id = (courses[courses.course_title == base_course_name].course_id).index[0]

    # Find the cos_sim record for TITLE_INDEX and save in - TITLE_COS_SIM as array
    title_cos_sim = clean_title_cos_sim[index_id]
    
    # Convert  TITLE_COS_SIM to a dict TITLE_COS_SIM_DICT
    title_cos_sim_dict = dict(enumerate(title_cos_sim))
    
    # Sort the Dict based on Value
    title_cos_sim_dict_sorted = dict(sorted(title_cos_sim_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Find top 5 values and the key[index]
    no_of_top_records = 6
    top_similar_titles = (list(title_cos_sim_dict_sorted.items())[:no_of_top_records])
    
    for key, value in top_similar_titles:
        if key != index_id:
            name.append(courses.loc[key].course_title)
        
    return name


st.title('Udemy similar course recommendor')
courses['course_title']

base_course_name = st.text_input("Enter a course from above for which you want to find similar courses")

if base_course_name:
    similar_courses = recom(base_course_name)
    st.write("Similar courses are:")
    for course in similar_courses:
        st.write("-", course)

else:
    st.empty()
