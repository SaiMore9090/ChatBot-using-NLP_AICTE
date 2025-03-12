import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issues for nltk downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file with better error handling
file_path = r"D:\Main Folder\Final Internship\intents.json"


if not os.path.exists(file_path):
    st.error(f"Error: The intents file was not found at: {file_path}")
    st.stop()

with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot response function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    
    return "Sorry, I didn't understand that."

# Initialize conversation counter
counter = 0

# Streamlit UI
def main():
    global counter
    st.title("Chatbot using NLP & Logistic Regression")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Type a message below to start chatting!")

        # Check and create chat log file if needed
        log_file = "chat_log.csv"
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, key=f"chatbot_response_{counter}")

            # Save the conversation log
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting! Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    # About Menu
    elif choice == "About":
        st.write("### Project Overview")
        st.write("This chatbot uses NLP and Logistic Regression to identify user intents and provide relevant responses.")
        st.write("The interface is built using Streamlit, allowing users to interact with the chatbot via a web-based UI.")

        st.subheader("Dataset:")
        st.write("The dataset consists of labeled intents, patterns, and responses stored in a JSON file.")

        st.subheader("How It Works:")
        st.write("""
        - The chatbot processes user input using TF-IDF vectorization.
        - A Logistic Regression model is trained to predict user intent.
        - Responses are selected based on the predicted intent.
        - The conversation history is logged for future reference.
        """)

        st.subheader("Conclusion:")
        st.write("This chatbot demonstrates the power of NLP and machine learning in building interactive AI assistants.")

# Run the Streamlit app
if __name__ == '__main__':
    main()
