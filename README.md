# Admission Prediction Project

## Project Overview

This project aims to build a machine learning model to predict the **Chance of Admit** for students based on various factors such as GRE score, TOEFL score, university rating, and more. The dataset contains relevant academic and research performance metrics for 500 applicants, which will be used to train and evaluate the model.

## Dataset Information

- **File Name**: `Jamboree_Admission.csv`
- **Total Records**: 500
- **Total Features**: 9

The features in the dataset are as follows:

| Column              | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| **Serial No.**       | Unique serial number for each applicant                           |
| **GRE Score**        | GRE (Graduate Record Examination) score (out of 340)              |
| **TOEFL Score**      | TOEFL (Test of English as a Foreign Language) score (out of 120)  |
| **University Rating**| Rating of the university (1 to 5)                                 |
| **SOP**              | Statement of Purpose rating (1 to 5)                              |
| **LOR**              | Letter of Recommendation strength (1 to 5)                        |
| **CGPA**             | Cumulative GPA (on a scale of 10)                                 |
| **Research**         | Research experience (0 = No, 1 = Yes)                             |
| **Chance of Admit**  | Probability of admission (0 to 1)                                 |

## Objective

The main goal of this project is to predict the **Chance of Admit** based on the provided features.

## Deployment

- The project includes a Flask web application to provide an intuitive interface for users to input applicant data and get predictions for the **Chance of Admit**.
- Additionally, the project has been deployed using **Streamlit**, enabling an interactive and user-friendly platform to explore the model's capabilities.

## Conclusion

This project demonstrates the application of machine learning techniques to predict university admissions based on a set of academic and research factors. The deployment through Flask and Streamlit ensures ease of use and accessibility for end users.
