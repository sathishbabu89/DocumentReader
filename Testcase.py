import logging
import streamlit as st
import zipfile
import io
import os
from langchain.llms import HuggingFaceEndpoint

# Force PyTorch to use CPU
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = ""  # Add your Hugging Face API token

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

st.set_page_config(page_title="Spring Boot Test Case Generator", page_icon="🧪")

st.header("Spring Boot Test Case Generator 💻")

# File upload for Spring Boot project zip
uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")

# Function to generate test cases using LLM
def generate_test_cases(class_name, class_code, llm):
    try:
        prompt = f"""
        You are a highly skilled developer specializing in Java Spring Boot. Generate JUnit 5 test cases for the given class. 

        - Ensure all public methods are tested.
        - Cover edge cases, exceptions, and typical inputs.
        - Use Mockito for mocking dependencies and include necessary annotations like @MockBean.
        - Write meaningful assertions using AssertJ or JUnit's assert methods.
        - If the class has no methods to test, state: "// No public methods to test."

        Class to test:
        {class_code}
        """

        response = llm.invoke(prompt)
        if not response.strip():
            return f"// No test cases generated for {class_name}."
        return response
    except Exception as e:
        logger.error(f"Error generating test cases for {class_name}: {e}")
        return f"// Error occurred during test generation for {class_name}."

if uploaded_file:
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            project_dir = "/tmp/springboot_project"
            os.makedirs(project_dir, exist_ok=True)
            zip_ref.extractall(project_dir)
            st.success("Project extracted successfully!")
    except Exception as e:
        logger.error(f"Failed to extract ZIP file: {e}", exc_info=True)
        st.error("Invalid ZIP file. Please upload a valid Spring Boot project.")

    # LLM initialization
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        max_new_tokens=2048,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    )

    # Generate test cases for each class
    if st.button("Generate Test Cases"):
        test_cases = {}
        progress_bar = st.progress(0)
        files_processed = 0

        with st.spinner("Generating test cases..."):
            java_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(project_dir)
                for file in files if file.endswith(".java")
            ]
            total_files = len(java_files)

            for file_path in java_files:
                try:
                    with open(file_path, "r") as file:
                        class_content = file.read()
                    class_name = os.path.basename(file_path).replace(".java", "Test.java")

                    # Generate test case for the current class
                    test_case_code = generate_test_cases(class_name, class_content, llm)
                    test_cases[class_name] = test_case_code

                    # Display individual test case as downloadable content
                    st.download_button(
                        label=f"Download {class_name}",
                        data=test_case_code,
                        file_name=class_name,
                        mime="text/x-java-source",
                    )

                    # Update progress bar
                    files_processed += 1
                    progress_bar.progress(files_processed / total_files)
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")

        # Create a downloadable zip of all test cases
        if test_cases:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for test_class, test_content in test_cases.items():
                    test_file_path = f"src/test/java/{test_class}"
                    zip_file.writestr(test_file_path, test_content)

            zip_buffer.seek(0)
            st.download_button(
                label="Download All Test Cases (ZIP)",
                data=zip_buffer,
                file_name="springboot_test_cases.zip",
                mime="application/zip",
            )
            st.success("Test cases generated successfully!")
        else:
            st.error("No valid Java classes found in the project.")
