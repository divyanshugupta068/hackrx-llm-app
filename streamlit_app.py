import streamlit as st
import requests

st.title("📄 Ask Questions from Your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Input question
question = st.text_input("Enter your question")

# When button clicked
if st.button("Get Answer"):
    if uploaded_file and question:
        # Upload the file to file.io or any temporary host
        with st.spinner("Uploading PDF..."):
            files = {"file": uploaded_file}
            res = requests.post("https://file.io", files=files)
            if res.status_code != 200:
                st.error("Failed to upload file")
                st.stop()
            file_url = res.json().get("link")

        # Prepare the data to send to FastAPI
        payload = {
            "documents": file_url,
            "questions": [question]
        }

        headers = {
            "Authorization": "Bearer testtoken"  # or your actual token
        }

        with st.spinner("Getting answer from LLM..."):
            response = requests.post(
                "https://hackrx-fastapi.onrender.com/api/v1/hackrx/run",
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                answer = response.json()["answers"][0]
                st.success("Answer:")
                st.write(answer)
            else:
                st.error(f"Error: {response.text}")
    else:
        st.warning("Please upload a PDF and enter a question.")
