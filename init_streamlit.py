import streamlit as st

from llm_model import send_prompt


def init_data_app():
    st.title("Chat with Ollama")

    if "messages" not in st.session_state.keys(): 
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question !"}
        ]

    if prompt := st.chat_input("Your question"): 
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: 
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_prompt(prompt)
                print(response)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
        

