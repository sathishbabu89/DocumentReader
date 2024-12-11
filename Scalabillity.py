import logging
import streamlit as st
import zipfile
import io
import os
from langchain.llms import HuggingFaceEndpoint
import shutil

# Set page configuration first (must be the first Streamlit command)
st.set_page_config(page_title="Spring Boot Project Deployment", page_icon="🚀")

# Force PyTorch to use CPU
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = "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"  # Add your Hugging Face API token

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

# Create the dropdown for page selection
pages = ["Upload Project & Generate Config", "Docker & Kubernetes Config Generator"]
selected_page = st.selectbox("Select a page", pages)

if selected_page == "Upload Project & Generate Config":
    st.header("Upload Spring Boot Project and Generate Dockerfile & Kubernetes Config 🚀")

    # Step 1: Upload Spring Boot Project ZIP
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")

    if uploaded_file:
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                project_dir = "/tmp/springboot_project"
                os.makedirs(project_dir, exist_ok=True)
                zip_ref.extractall(project_dir)
                st.success("Spring Boot project extracted successfully!")
        except Exception as e:
            logger.error(f"Failed to extract ZIP file: {e}", exc_info=True)
            st.error("Invalid ZIP file. Please upload a valid Spring Boot project.")

        # Step 2: Button to Generate Dockerfile and Kubernetes Config
        if st.button("Generate Dockerfile & Kubernetes Config"):
            # Get the base name of the project (e.g., "yourproject")
            project_name = os.path.basename(project_dir)

            # LLM initialization for Dockerfile and Kubernetes config generation
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

            # Generate Dockerfile using LLM
            dockerfile_prompt = f"""
            Generate a Dockerfile for a Java Spring Boot microservice called "{project_name}".
            - The base image should be openjdk:17.
            - Expose port 8080.
            - The application JAR file is located in the target directory and is named "{project_name}.jar".
            - Ensure that the Dockerfile is optimized for production with a minimal image size and follows best practices.
            """

            # Generate Kubernetes YAML using LLM
            k8s_prompt = f"""
            Generate Kubernetes deployment and service YAML for a Spring Boot microservice called "{project_name}".
            - Kubernetes namespace: default
            - Replicas: 2
            - Container port: 8080
            - The service should be of type ClusterIP.
            - Ensure that the deployment and service are configured to expose the application correctly.
            """

            try:
                # Generate Dockerfile content
                dockerfile_content = llm.invoke(dockerfile_prompt)

                # Generate Kubernetes configuration content
                k8s_config_content = llm.invoke(k8s_prompt)

                # Step 3: Display Generated Files to the User

                # Show Dockerfile content
                st.subheader("Generated Dockerfile:")
                st.code(dockerfile_content, language="dockerfile")

                # Show Kubernetes configuration content
                st.subheader("Generated Kubernetes Configuration:")
                st.code(k8s_config_content, language="yaml")

                # Step 4: Provide Download Option

                # Create necessary folders in the project directory
                kubernetes_dir = os.path.join(project_dir, "kubernetes")
                os.makedirs(kubernetes_dir, exist_ok=True)

                # Paths for generated files
                dockerfile_path = os.path.join(project_dir, "Dockerfile")
                k8s_deployment_path = os.path.join(kubernetes_dir, "deployment.yaml")
                k8s_service_path = os.path.join(kubernetes_dir, "service.yaml")

                # Save Dockerfile and Kubernetes config to files
                with open(dockerfile_path, 'w') as dockerfile:
                    dockerfile.write(dockerfile_content)

                with open(k8s_deployment_path, 'w') as k8s_deployment_file:
                    k8s_deployment_file.write(k8s_config_content)

                with open(k8s_service_path, 'w') as k8s_service_file:
                    # Using the same content for simplicity, you could modify it if needed
                    k8s_service_file.write(k8s_config_content)

                # Step 5: Create a ZIP of the Entire Updated Project
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for root, dirs, files in os.walk(project_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zip_file.write(file_path, os.path.relpath(file_path, project_dir))

                zip_buffer.seek(0)
                # Provide the user a button to download the updated project zip
                st.download_button(
                    label="Download Updated Spring Boot Project with Dockerfile & Kubernetes Config",
                    data=zip_buffer,
                    file_name="springboot_with_docker_k8s.zip",
                    mime="application/zip"
                )
                st.success("Dockerfile and Kubernetes configuration generated and added to the project!")

            except Exception as e:
                logger.error(f"Error generating Dockerfile and Kubernetes config: {e}")
                st.error("Error generating configurations. Please try again.")
                
elif selected_page == "Docker & Kubernetes Config Generator":
    st.header("Docker & Kubernetes Configuration Generator 🌐")

    # This page can be used for other purposes if needed
    st.info("Use the 'Upload Project & Generate Config' page to upload your Spring Boot project and generate Docker & Kubernetes files.")
