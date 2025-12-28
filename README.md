# Autojudge-project
AutoJudge — Predicting Programming Problem Difficulty
                                                       AANVI JHA (24114002)


Overview:
Online coding platforms such Codeforces, CodeChef assign difficulty labels and scores to programming problems.
These ratings are typically based on human judgment,which can be subjective and slow.
AutoJudge is a machine learning–based system that automatically predicts Difficulty Class: Easy / Medium / Hard and Difficulty Score: A numerical difficulty score using only the textual content of a problem statement.                                                           The system is trained on a labeled dataset and deployed through an interactive Streamlit web application.
Methodology:
1. Data Preprocessing                                                                                   Combined all textual fields into a single full_text feature.                                                 Handled missing values by replacing them with empty strings
2. Feature extraction:                                                                                                 To convert text into numerical features:
* TF-IDF Vectorization: Captures important words and phrase, using unigrams and bigrams
* Hand-crafted features: Text length, Count of mathematical symbols (+-*/=<>)
        These features are concatenated to form the final input vector.
3. Model — Classification:
Support Vector Machine (LinearSVC) is used to predict the 3 difficulty classes- Easy, Medium and Hard. SVM was chosen as it performs well on high-dimensional text data and is robust for classification tasks.
Evaluation Metrics
* Accuracy
* Confusion Matrix
* Precision / Recall / F1-Score
4. Model- Regression:
Random Forest Regressor is used to predict numerical difficulty score.                         It captures non-linear relationships and is robust to noisy and subjective labels
Evaluation Metrics
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)


Results
Achieved a classification accuracy of 48.48 % and MAE 1.7, RSME 2.04.
This shows reasonable MAE and RMSE given subjective difficulty scores.


Deployment 
The trained models are deployed using Streamlit, allowing users to:
1. Paste a new problem description
2. Click predict
3. Instantly see predicted difficulty class and predicted difficulty score
For Local run- run streamlit run app.py on your terminal.
