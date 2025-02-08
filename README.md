## This project is about SMS Spam Detection System

Abstract of project :

This project addresses the challenge of spam message proliferation across digital communication platforms by developing an advanced spam detection system leveraging machine learning. The primary objective is to create a scalable, user-friendly solution that ensures accurate classification of spam and legitimate messages, enhancing the digital communication experience.

The methodology begins with data preprocessing, including tokenization, removal of noise (e.g., stop words, punctuation), and stemming, followed by feature extraction using TF-IDF vectorization. A Multinomial Naive Bayes (MNB) classifier is trained on a labeled dataset of 5,575 SMS messages to identify patterns indicative of spam. The system's design emphasizes real-time processing, an intuitive user interface, and adaptability to varying user preferences.

Evaluation on the test dataset yielded robust performance, with metrics such as accuracy, precision, recall, and F1-score confirming the system's effectiveness. The spam detection system achieves precise identification of spam messages while minimizing false positives and negatives, providing users with a reliable tool for managing unwanted messages.

# Code :

First of all I install all the libraries and dependency required for the project

    streamlit
    scikit-learn
    numpy
    pandas
    scipy
    pickle
    tkinter

Data Pre-processing :

We drop the columns which contains NaN values in dataset

    data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

Then give the class columns values as Ham and Spam
 
    data['class'] = data['class'].map({'ham': 0, 'spam': 1})

Import Vectorizer and Split the data into train & test datasets

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split

    X=data['message']
    y=data['class']

Then use fit transform and create training and testing subsets

    cv=CountVectorizer()
    X=cv.fit_transform(X)
    x_train, x_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

From naive bayes use MultinomialNB model
    
    from sklearn.naive_bayes import MultinomialNB
    model=MultinomialNB()
    model.fit(x_train, y_train)

Then find the Score of model

    model.score(x_test, y_test)

The Score of the model is --> 0.97847533632287

# Then to convent this code into an Streamlit application :

First import the libraries
     
    import streamlit as st
    import pickle

Then load the pickle file which are spam.pkl & vectorizer.pkl

    model = pickle.load(open('spam.pkl','rb'))
    cv = pickle.load(open('vectorizer.pkl','rb'))

Give the title and subtitle of the application

    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify email as spam or ham.")
    user_input = st.text_area("Enter an email to classify",height=150)

Create the Classify button and show the output of the massage

    if st.button("Classify"):
        if user_input:
            data = [user_input]
            vectorized_data = cv.transform(data).toarray()
            result = model.predict(vectorized_data)
            if result[0] == 0:
                st.write("The email is Not Spam")
            else:
                st.write("The email is Spam")
        else:
            st.write("Please type Email to classify")


Model Workflow : 

![image](https://github.com/user-attachments/assets/ba78ac71-c5dc-46e1-865b-e58633884bcb)

Output and Deployment of model using Streamlit :

![image](https://github.com/user-attachments/assets/bd19c608-4a66-4824-82d7-a12d79bbf8cc)
The above image shows the interface of the Streamlit, in this some SMS massage is given to the model in the massage box then the model has classify the massage as Not Spam.

![image](https://github.com/user-attachments/assets/697586a2-4b04-4e04-8dba-c7d441f914bb)
This image shows the output of spam massage, in this some SMS massage is given to the model in the massage box then the model has classify the massage as Spam.
