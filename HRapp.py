from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pdfplumber
import re
import spacy
import random
import pickle
import base64
import json



# Load the trained model
nlp = spacy.load("model")


models_paths = {
    'JD1': 'xgb_model_JD1.pkl',
    'JD2': 'xgb_model_JD2.pkl',
    'JD3': 'xgb_model_JD3.pkl',
    'JD4': 'xgb_model_JD4.pkl',
    'JD5': 'xgb_model_JD5.pkl',
}


# Đọc JD từ tập tin JSON
def load_jd_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Lưu JD vào tập tin JSON
def save_jd_to_file(jd_db, filename):
    with open(filename, 'w') as file:
        json.dump(jd_db, file, indent=4)

# Khởi tạo JD DB từ tập tin
jd_db = load_jd_from_file('jd_data.json')

#Các hàm tiền xử lý pdf file
# đọc và trích data từ pdf
def pdf_to_text(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'  # Thêm xuống dòng giữa các trang
    return text

def pdf_to_images(pdf_file):
    images = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Convert each page to an image
            pil_image = page.to_image()
            # Convert PIL image to numpy array
            np_image = pil_image._repr_png_()
            # Append the image to the list
            images.append(np_image)
    return images

# tạo profile ứng cử viên

def extract_info(text):
    doc = nlp(text)
    extracted_info = {
        'name': None,
        'mail': None,
        'degree': None,
        'designation': [],
        'skills': [],
        'years': [],
        'experience': None,
        'languages': [],
        'full_info': text.strip()
    }
    for ent in doc.ents:
        if ent.label_ == 'Name':
            extracted_info['name'] = ent.text
        elif ent.label_ == 'Email Address':
            extracted_info['mail'] = ent.text
        elif ent.label_ == 'Degree':
            extracted_info['degree'] = ent.text
        elif ent.label_ == 'Designation':
            extracted_info['designation'].append(ent.text)
        elif ent.label_ == 'Skills':
            extracted_info['skills'].append(ent.text)
        elif ent.label_ == 'Years of Experience':
            extracted_info['years'].append(ent.text)
        elif ent.label_ == 'experience':
            extracted_info['experience'].append(ent.text)

    # Convert lists to string if necessary
    extracted_info['designation'] = ', '.join(extracted_info['designation'])
    extracted_info['skills'] = ', '.join(extracted_info['skills'])
    extracted_info['years'] = ', '.join(extracted_info['years'])

    # Kiểm tra các trường thông tin không được trích xuất và trích xuất thủ công nếu cần
    if extracted_info['degree'] is None or extracted_info['degree'] == '':
        extracted_info['degree'] = extract_degree(text)
    if extracted_info['skills'] is None or extracted_info['skills'] == '':
        extracted_info['skills'] = extract_skills(text)
    if extracted_info['years'] is None or extracted_info['years'] == '':
        extracted_info['years'] = extract_years_of_experience(text)
    if extracted_info['experience'] is None or extracted_info['experience'] == '':
        extracted_info['experience'] = extract_experience(text)

    # Extract languages
    extracted_info['languages'] = find_languages(text)

    return extracted_info

def extract_degree(text):
    # Search for a line mentioning 'University' or 'College' after 'EDUCATION'
    degree_match = re.search(r'EDUCATION[^\n]*\n([^\n]*(University|College))[^\n]*', text)
    # Search for any four-digit number that could represent a year, likely after mentioning 'University' or 'College'
    year_match = re.search(r'(\d{4})', text)
    current_year = datetime.now().year

    university_name = None
    graduation_year = None

    if degree_match:
        university_name = degree_match.group(1)  # Capture university or college name

    if year_match:
        # Check if the year is a valid graduation year (i.e., before the current year)
        if int(year_match.group(1)) < current_year:
            graduation_year = year_match.group(1)

    # If there's no explicit degree, use the university name; include graduation year if found and valid
    if university_name:
        return f"{university_name}{', ' + graduation_year if graduation_year else ''}"
    return None

def extract_skills(text):
    # Flexible extraction of skills between 'SKILLS' section and another section or end of the segment.
    skills_match = re.search(r'SKILLS[^\n]*\n+([^EDUCATION|EXPERIENCE|CERTIFICATION]+)', text, re.DOTALL)
    if skills_match:
        # Filtering and cleaning up the skills text
        skills_text = re.sub(r'\s+', ' ', skills_match.group(1))  # Replacing large whitespaces with single space
        return skills_text.strip()
    return None

def extract_years_of_experience(text):
    # Generic extraction of years of experience using common phrasing.
    years_of_experience_match = re.search(r'(\d+\+? years? of experience)', text, re.IGNORECASE)
    if years_of_experience_match:
        return years_of_experience_match.group(1)
    return None

def extract_experience(text):
    # Modify to extract everything after "WORK EXPERIENCE" up to another major heading or end of the section
    # We use non-greedy match (.*?) to stop at the first instance of another section or the end of text
    # re.DOTALL allows dot (.) to match newlines as well
    pattern = r"WORK EXPERIENCE\s*(.*?)(?=\n(?:EDUCATION|SKILLS|ADDITIONAL INFORMATION|$))"
    experience_match = re.search(pattern, text, re.DOTALL)
    if experience_match:
        # Clean up the captured text: remove excessive whitespace, split into lines, rejoin with single newline
        experience_text = experience_match.group(1)
        cleaned_lines = [line.strip() for line in experience_text.split('\n') if line.strip()]  # Remove empty lines and strip
        return '\n'.join(cleaned_lines)
    return None


def find_languages(text):
    languages = ['.NET','Ada','Assembly',"Golang",'C#','C+','CSS','Clojure','Dart','Delphi','Erlang','Forth','Fortran','Go','Groovy','HTML','Haskell','Java','JavaScript','Julia','Kotlin','Lua','MATLAB','Objective-C','PHP','Pascal','Perl','PowerShell',"scratch",'Prolog','Python','R','Racket','Ruby','Rust','SQL','Scala','Scheme','Smalltalk','Swift','SwiftUI','TypeScript','Visual Basic','Visual Basic.NET']
    list_languages = set()
    for language in languages:
        if re.search(r'\b{}\b'.format(re.escape(language)), text, re.IGNORECASE):
            list_languages.add(language)
    return list(list_languages)

def extract_years(years_str):
    # Tìm tất cả các chữ số trong chuỗi và ghép lại thành số nguyên
    digits = re.findall(r'\d+', years_str)
    return int(digits[0]) if digits else 0  # Trả về 0 nếu không tìm thấy số

def preprocess(df, job_description):
    """
    Tiền xử lý dữ liệu ứng viên và tính toán độ tương đồng với mô tả công việc.

    Args:
        data (DataFrame): Dữ liệu ứng viên.
        job_description (dict): Mô tả công việc bao gồm Title, Skills và Requirements.

    Returns:
        DataFrame: Dữ liệu ứng viên đã được tiền xử lý.
    """

    # Áp dụng hàm trên để lượng hóa cột 'years'
    df['years'] = df['years'].astype(str).apply(extract_years)

    # Lượng hóa cột 'degree' bằng cách kiểm tra nếu giá trị rỗng thì gán 0, ngược lại gán 1
    df['has_degree'] = df['degree'].apply(lambda x: 0 if pd.isnull(x) or x == '' else 1)

    # Add a column indicating whether 'experience' is non-empty
    df['has_experience'] = df['experience'].apply(lambda x: 1 if x else 0)

    # Tạo vectorizer TF-IDF
    vectorizer = TfidfVectorizer()

    # Làm sạch dữ liệu bằng cách thay thế NaN bằng chuỗi trống trực tiếp trên DataFrame
    df.fillna('', inplace=True)

    # Hợp nhất các cột dữ liệu lại với nhau
    candidates_title = df['designation'].astype(str) + " " + df['degree'].astype(str)
    candidates_skills = df['skills'].astype(str) + " " + df['languages'].astype(str)
    candidates_requirements = df['experience'].astype(str) + " " + df['languages'].astype(str)

    # Tạo lại các văn bản để so sánh
    documents_title = candidates_title.tolist() + [job_description['Title']]
    documents_skills = candidates_skills.tolist() + [job_description['Skills']]
    documents_requirements = candidates_requirements.tolist() + [job_description['Requirements']]

    # Vector hóa và tính toán độ tương đồng lại, đảm bảo không có lỗi
    tfidf_title = vectorizer.fit_transform(documents_title)
    cosine_sim_title = cosine_similarity(tfidf_title[:-1], tfidf_title[-1:])

    tfidf_skills = vectorizer.fit_transform(documents_skills)
    cosine_sim_skills = cosine_similarity(tfidf_skills[:-1], tfidf_skills[-1:])

    tfidf_requirements = vectorizer.fit_transform(documents_requirements)
    cosine_sim_requirements = cosine_similarity(tfidf_requirements[:-1], tfidf_requirements[-1:])

    if isinstance(job_description, dict):
        job_description_text = ' '.join([job_description['Title'], job_description['Skills'], job_description['Requirements']])
    else:
        job_description_text = job_description

    # Creating full_info similarity scores
    documents_full_info = df['full_info'].tolist() + [job_description_text]
    tfidf_full_info = TfidfVectorizer().fit_transform(documents_full_info)
    cosine_sim_full_info = cosine_similarity(tfidf_full_info[:-1], tfidf_full_info[-1:]).flatten()

    # Update your DataFrame with new similarity scores
    df['Overall_Similarity'] = cosine_sim_full_info

    # Combine with your existing similarity scores
    similarity_scores_updated = {
        'Title_Similarity': cosine_sim_title.flatten(),
        'Skills_Similarity': cosine_sim_skills.flatten(),
        'Requirements_Similarity': cosine_sim_requirements.flatten(),
        'Overall_Similarity': cosine_sim_full_info  # Add this line
    }
    similarity_df_updated = pd.DataFrame(similarity_scores_updated)
    df_final = df[['name', 'mail', 'years', 'has_degree', 'has_experience']].copy().join(similarity_df_updated)

    return df_final


# Streamlit app
def main():
    # Set page title and icon
    st.set_page_config(page_title="Resume Analyzer", page_icon=":clipboard:")

    st.title("HR Application for Candidate Screening")
    st.markdown("Analyze resumes and help HR department in screening candidates. Finding potential candidates for specific job description customized by HR team.")

    # Mục cho Job Descriptions
    st.sidebar.header("Job Descriptions")
    selected_jd = st.sidebar.selectbox("Select a JD", list(jd_db.keys()))
    action = st.sidebar.radio("Action", ['View', 'Add', 'Edit', 'Delete'])

    #Load model
    if selected_jd in jd_db:
        bst = None  # Khởi tạo bst là None
        # Tải mô hình tương ứng với JD được chọn
        model_path = models_paths.get(selected_jd)
        if model_path:
            with open(model_path, 'rb') as f:
                bst = pickle.load(f)

    # 'View' action
    if action == 'View':
      jd_info = jd_db.get(selected_jd, {})
      if jd_info:
          st.subheader(f"Job Description: {selected_jd}")
          st.write(f"**Title:** {jd_info.get('Title', 'N/A')}")
          st.write(f"**Skills Required:** {jd_info.get('Skills', 'N/A')}")
          st.write(f"**Requirements:** {jd_info.get('Requirements', 'N/A')}")
          st.write(f"**Responsibilities:** {jd_info.get('Responsibilities', 'N/A')}")

      else:
          st.error("Job description not found.")

    # Handling 'Add' action
    elif action == 'Add':
        new_jd_key = st.text_input("Enter a unique key for the new JD")
        new_title = st.text_input("Enter the title for the new JD")
        new_skills = st.text_input("Enter the skills required for the new JD")
        new_requirements = st.text_input("Enter the requirements for the new JD")
        new_content = st.text_area("Enter additional content for the new JD")
        if st.button('Add New JD'):
            if new_jd_key in jd_db:
                st.error("This key already exists. Please use a different key.")
            elif new_jd_key:
                jd_db[new_jd_key] = {
                    'Title': new_title,
                    'Skills': new_skills,
                    'Requirements': new_requirements,
                    'Responsibilities': new_content
                }
                save_jd_to_file(jd_db, 'jd_data.json')
                st.success(f"JD {new_jd_key} added successfully!")
            else:
                st.error("Please enter a unique key for the new JD.")

    # Handling 'Edit' action
    elif action == 'Edit':
        if selected_jd:  # Check if a JD is selected
            edited_title = st.text_input("Edit the title", jd_db[selected_jd]['Title'])
            edited_skills = st.text_input("Edit the skills", jd_db[selected_jd]['Skills'])
            edited_requirements = st.text_input("Edit the requirements", jd_db[selected_jd]['Requirements'])
            edited_content = st.text_area("Edit additional content", jd_db[selected_jd].get('Responsibilities', ''))
            if st.button('Update JD'):
                jd_db[selected_jd] = {
                    'Title': edited_title,
                    'Skills': edited_skills,
                    'Requirements': edited_requirements,
                    'Responsibilities': edited_content
                }
                save_jd_to_file(jd_db, 'jd_data.json')
                st.success(f"JD {selected_jd} updated successfully!")
        else:
            st.error("Please select a JD to edit.")

    # Handling 'Delete' action
    elif action == 'Delete':
        if selected_jd:  # Check if a JD is selected
            if st.button('Delete JD'):
                del jd_db[selected_jd]
                save_jd_to_file(jd_db, 'jd_data.json')
                st.success(f"JD {selected_jd} deleted successfully!")
        else:
            st.error("Please select a JD to delete.")

    # Upload ứng viên
    st.header("Upload Candidate Profiles")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
      for uploaded_file_idx, uploaded_file in enumerate(uploaded_files):
          st.write(f"## {uploaded_file.name}")
          # Convert PDF to list of images
          images = pdf_to_images(uploaded_file)

          # Display each page as an image with a single "Next" button
          page_number = st.empty()  # Placeholder to display current page number
          current_page = 0
          total_pages = len(images)
          while current_page < total_pages:
              st.image(images[current_page], caption=f"Page {current_page+1}", use_column_width=True)
              # Display current page number
              page_number.write(f"Page {current_page+1} of {total_pages}")
              # Display "Next" button with a unique key combining file index and page number
              next_key = f"next_{uploaded_file_idx}_{current_page}"
              if st.button("Next", key=next_key):
                  current_page += 1
                  st.empty()  # Clear previous image and page number
              else:
                  break  # Exit loop if "Next" button is not clicked
          else:
              st.write("End of document")

          # Process content of each PDF file
          text = pdf_to_text(uploaded_file)
          # st.write(f"Content of {uploaded_file.name}:")
          # st.text(text[:500])  # Display a part of the content to check

          candidate_data = extract_info(text)
          candidate_data = pd.DataFrame([candidate_data])  # Convert from dict to DataFrame

          # Continue only if bst has been initialized
          if bst:
              jd_info = jd_db[selected_jd]
              df_preprocessed = preprocess(candidate_data, jd_info)
              df_preprocessed = df_preprocessed.drop(columns=['name', 'mail'], errors='ignore')

              # Convert the new data into DMatrix
              # new_data_dmatrix = xgb.DMatrix(df_preprocessed)

              # Use the trained model to predict
              new_preds = bst.predict(df_preprocessed)

              # Apply np.argmax based on the shape of new_preds
              if len(new_preds.shape) > 1 and new_preds.shape[1] > 1:
                  candidate_data['Predicted'] = np.argmax(new_preds, axis=1)
              else:
                  candidate_data['Predicted'] = new_preds

              # Display results
              with st.container():
                  st.write(f"Results for the profile of {uploaded_file.name}:")

                  if 'Predicted' in candidate_data.columns:
                      candidate_data['Predicted'] = candidate_data['Predicted'].map({0: 'Not suitable', 1: 'Potential candidate', 2: 'Suitable candidate'})

                      suitable_candidates = candidate_data[candidate_data['Predicted'] != 'Not suitable']
                      if not suitable_candidates.empty:
                          st.write("Suitable candidates:")
                          st.dataframe(suitable_candidates[['name', 'mail', 'Predicted']])
                      else:
                          st.write("No suitable candidates found based on the criteria.")
                  else:
                      st.error("Prediction results unavailable. Please check if the model has generated predictions.")
          else:
              st.error("Model could not be loaded.")
if __name__ == "__main__":
    main()
