import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/chat"


st.set_page_config(page_title=" AI Chatbot", page_icon="🤖")
st.title(" My Deep Learning Chatbot")
st.write("Ask me anything! I'm running on a custom PyTorch model.")


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user input
if prompt := st.chat_input("Type your message here..."):
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call the API
    try:
        with st.spinner("Thinking..."):
            response = requests.post(API_URL, json={"user_input": prompt})

            if response.status_code == 200:
                bot_reply = response.json()["response"]

                # 3. Display bot response
                with st.chat_message("assistant"):
                    st.markdown(bot_reply)

                # Save bot response to history
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            else:
                st.error(f"Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the Backend! Is 'app.py' running?")
