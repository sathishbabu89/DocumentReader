import logging
import streamlit as st
import re
import torch
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import requests
import json
import pandas as pd
from typing import Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint


# Force PyTorch to use CPU
device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = "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"  # Add your Hugging Face API token

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

st.set_page_config(page_title="C++ to Java Conversion Tool", page_icon="💻")

page = st.sidebar.selectbox("Choose Page", ["Bug Detection", "Repo Summary"])

# Function to analyze repo summary based on the role selected
def repo_summary_page(uploaded_file, role):
    if uploaded_file:
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                project_dir = "C:\\Temp\\springboot_project"
                os.makedirs(project_dir, exist_ok=True)
                zip_ref.extractall(project_dir)
                st.success("Project extracted successfully!")
        except Exception as e:
            logger.error(f"Failed to extract ZIP file: {e}", exc_info=True)
            st.error("Invalid ZIP file. Please upload a valid Spring Boot project.")

        # LLM initialization for repo summary
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        )

        # Analyze the repo
        if st.button("Generate Repo Summary"):
            try:
                # Collect repo content
                all_files_content = []
                java_files = [
                    os.path.join(root, file)
                    for root, _, files in os.walk(project_dir)
                    for file in files if file.endswith(".java")
                ]
                for file_path in java_files:
                    with open(file_path, "r") as file:
                        all_files_content.append(file.read())

                # Create prompt based on the selected role
                prompt = f"""
                You are a {role} analyzing the following project. The project contains Java files and you are tasked with providing an overview of the code, identifying key aspects based on your role:
                
                Role: {role}
                Project files:
                {str(all_files_content)}

                Provide a detailed summary about the project based on your role. If you're a Developer, focus on code quality, architecture, and potential issues. If you're a Tester, highlight areas that may require thorough testing or may have bugs. If you're a Project Manager, focus on overall project health and team productivity.
                """

                # Request the summary from the LLM
                response = llm.invoke(prompt)
                st.subheader(f"Repo Summary for {role}")
                st.write(response)
            except Exception as e:
                logger.error(f"Error generating repo summary: {e}")
                st.error("Error generating repo summary.")
    else:
        st.warning("Please upload a Spring Boot project (ZIP).")

# Function for bug detection page (same as previous)
def bug_detection_page(uploaded_file):
    if uploaded_file:
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                project_dir = "C:\\Temp\\springboot_project"
                os.makedirs(project_dir, exist_ok=True)
                zip_ref.extractall(project_dir)
                st.success("Project extracted successfully!")
        except Exception as e:
            logger.error(f"Failed to extract ZIP file: {e}", exc_info=True)
            st.error("Invalid ZIP file. Please upload a valid Spring Boot project.")

        # LLM initialization
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=2048,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        )

        # Detect bugs in each class (same as previous)
        if st.button("Detect Bugs and Suggest Fixes"):
            bugs_and_fixes = {}
            severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
            fix_confidences = []
            total_files = 0

            with st.spinner("Detecting bugs..."):
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
                        class_name = os.path.basename(file_path).replace(".java", "")

                        # Detect bugs in the current class (same as previous)
                        bugs = detect_bugs_in_code(class_content)
                        if bugs:
                            bug_fixes = {}
                            for bug in bugs:
                                fix = suggest_bug_fix(bug, class_content, llm)
                                severity = categorize_bug_severity(bug)
                                severity_counts[severity] += 1

                                confidence = get_fix_confidence(fix)
                                fix_confidences.append(confidence)

                                bug_fixes[bug] = {
                                    "fix": fix,
                                    "severity": severity,
                                    "confidence": confidence,
                                }
                            bugs_and_fixes[class_name] = bug_fixes

                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")

            # Display detected bugs and suggested fixes (same as previous)
            if bugs_and_fixes:
                for class_name, bug_fixes in bugs_and_fixes.items():
                    st.subheader(f"Bugs in {class_name}:",divider=True)
                    for bug, details in bug_fixes.items():
                        st.write(f"Bug: {bug}")
                        st.write(f"Severity: {details['severity']}")
                        st.write(f"Suggested Fix: {details['fix']}")
                        st.write(f"Fix Confidence: {details['confidence']:.2f}%")

                # Generate metrics dashboard (same as previous)
                generate_metrics_dashboard(severity_counts, fix_confidences, total_files)

                # Provide download option (same as previous)
                st.subheader("Download Results")
                download_data = []

                # Prepare data for download (same as previous)
                for class_name, bug_fixes in bugs_and_fixes.items():
                    for bug, details in bug_fixes.items():
                        download_data.append({
                            "Class Name": class_name,
                            "Bug": bug,
                            "Severity": details['severity'],
                            "Suggested Fix": details['fix'],
                            "Fix Confidence (%)": f"{details['confidence']:.2f}",
                        })

                # Convert to DataFrame for CSV download (same as previous)
                df = pd.DataFrame(download_data)
                csv_data = df.to_csv(index=False)
                json_data = json.dumps(bugs_and_fixes, indent=4)

                # Allow user to download as CSV (same as previous)
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name="bug_fix_results.csv",
                    mime="text/csv",
                )

                # Allow user to download as JSON (same as previous)
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name="bug_fix_results.json",
                    mime="application/json",
                )

            else:
                st.write("No bugs detected in the provided code.")

# Page navigation
if page == "Bug Detection":
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    bug_detection_page(uploaded_file)
    
elif page == "Repo Summary":
    # Select role
    role = st.selectbox("Select your role", ["Developer", "Tester", "Project Manager"])
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    repo_summary_page(uploaded_file, role)

# Footer
st.sidebar.markdown("### About")
st.sidebar.write("This tool uses state-of-the-art AI models to assist with C++ to Java conversion, specifically tailored for Spring Boot applications.")
