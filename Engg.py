import logging
import streamlit as st
import zipfile
import os
from langchain.llms import HuggingFaceEndpoint
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = "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"  # Add your Hugging Face API token

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

# Use the actual emoji instead of surrogate Unicode characters
st.set_page_config(page_title="Spring Boot Project Analyzer", page_icon="📊", layout="wide")

st.header("Spring Boot Project Analyzer 📊")


def generate_cicd_pipeline(project_dir: str, tool: str, llm):
    """Generate CI/CD pipeline for the project based on the selected tool."""
    try:
        prompt = f"""
        You are a DevOps expert. Generate a CI/CD pipeline configuration for a Spring Boot project located in {project_dir}.

        The pipeline should include the following stages:
        - Build: Compile the project using Maven.
        - Test: Run unit tests.
        - Package: Create a deployable artifact (e.g., JAR file).
        - Deploy: Deploy the application to a Kubernetes cluster.

        Provide the configuration in the format for the selected tool: {tool}.
        """
        response = llm.invoke(prompt)
        if not response.strip():
            return f"// No CI/CD pipeline generated for tool {tool}."
        return response
    except Exception as e:
        logger.error(f"Error generating CI/CD pipeline for tool {tool}: {e}")
        return f"// Error occurred during CI/CD pipeline generation for {tool}."


def generate_microservice_recommendations(project_dir: str, llm):
    """Generate microservice architecture recommendations for the project."""
    try:
        prompt = f"""
        You are a software architect. Analyze the Spring Boot project located in {project_dir} and provide recommendations to refactor or improve its microservice architecture.

        Your recommendations should include:
        - Service boundaries and separation of concerns.
        - Scalability and fault tolerance strategies.
        - Best practices for communication between services (e.g., REST, gRPC, messaging).
        - Deployment and monitoring considerations.
        - Suggestions for database per service or shared database designs.

        Provide your response in a structured format.
        """
        response = llm.invoke(prompt)
        if not response.strip():
            return "No recommendations were generated for the microservice architecture."
        return response
    except Exception as e:
        logger.error(f"Error generating microservice recommendations: {e}")
        return "An error occurred while generating microservice recommendations."


def cicd_pipeline_generator_page(uploaded_file):
    """CI/CD Pipeline Generator Page."""
    if uploaded_file:
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                project_dir = "/tmp/springboot_project_cicd"
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

        st.subheader("Select CI/CD Tool")
        tool = st.selectbox("Choose a tool for CI/CD pipeline generation", ("Jenkins", "GitHub Actions", "GitLab CI"))

        if st.button("Generate CI/CD Pipeline"):
            with st.spinner("Generating CI/CD pipeline..."):
                pipeline_config = generate_cicd_pipeline(project_dir, tool, llm)

            st.subheader(f"Generated {tool} Pipeline Configuration")
            st.code(pipeline_config, language="yaml")

            st.download_button(
                label="Download Pipeline Configuration",
                data=pipeline_config,
                file_name=f"{tool.lower()}_pipeline.yml",
                mime="text/yaml",
            )


def microservice_recommendations_page(uploaded_file):
    """Microservice Recommendations Page."""
    if uploaded_file:
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                project_dir = "/tmp/springboot_project_microservices"
                os.makedirs(project_dir, exist_ok=True)
                zip_ref.extractall(project_dir)
                st.success("Project extracted successfully!")
        except Exception as e:
            logger.error(f"Failed to extract ZIP file: {e}", exc_info=True)
            st.error("Invalid ZIP file. Please upload a valid Spring Boot project.")

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

        if st.button("Generate Microservice Recommendations"):
            with st.spinner("Generating microservice recommendations..."):
                recommendations = generate_microservice_recommendations(project_dir, llm)

            st.subheader("Microservice Architecture Recommendations")
            st.text_area("Recommendations", value=recommendations, height=300)

            st.download_button(
                label="Download Recommendations",
                data=recommendations,
                file_name="microservice_recommendations.txt",
                mime="text/plain",
            )


# Page selection
page = st.sidebar.selectbox(
    "Select a Page", ("Test Case Generator", "Project Analysis", "CI/CD Pipeline Generator", "Microservice Recommendations")
)

if page == "Test Case Generator":
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    # test_case_generator_page(uploaded_file)

elif page == "Project Analysis":
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    # Call project analysis function here

elif page == "CI/CD Pipeline Generator":
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    cicd_pipeline_generator_page(uploaded_file)

elif page == "Microservice Recommendations":
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    microservice_recommendations_page(uploaded_file)
