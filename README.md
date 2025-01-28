## This project is about SMS Spam Detection System

Abstract of project :

This project addresses the challenge of spam message proliferation across digital communication platforms by developing an advanced spam detection system leveraging machine learning. The primary objective is to create a scalable, user-friendly solution that ensures accurate classification of spam and legitimate messages, enhancing the digital communication experience.

The methodology begins with data preprocessing, including tokenization, removal of noise (e.g., stop words, punctuation), and stemming, followed by feature extraction using TF-IDF vectorization. A Multinomial Naive Bayes (MNB) classifier is trained on a labeled dataset of 5,575 SMS messages to identify patterns indicative of spam. The system's design emphasizes real-time processing, an intuitive user interface, and adaptability to varying user preferences.

Evaluation on the test dataset yielded robust performance, with metrics such as accuracy, precision, recall, and F1-score confirming the system's effectiveness. The spam detection system achieves precise identification of spam messages while minimizing false positives and negatives, providing users with a reliable tool for managing unwanted messages.

Model Workflow : 

![image](https://github.com/user-attachments/assets/ba78ac71-c5dc-46e1-865b-e58633884bcb)

Output and Deployment of model using Streamlit :

![image](https://github.com/user-attachments/assets/bd19c608-4a66-4824-82d7-a12d79bbf8cc)
The above image shows the interface of the Streamlit, in this some SMS massage is given to the model in the massage box then the model has classify the massage as Not Spam.

![image](https://github.com/user-attachments/assets/697586a2-4b04-4e04-8dba-c7d441f914bb)
This image shows the output of spam massage, in this some SMS massage is given to the model in the massage box then the model has classify the massage as Spam.
