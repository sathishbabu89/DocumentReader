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

page = st.sidebar.selectbox("Choose Page", ["File Upload Converter", "Inline Code Converter","Bug Detection","Business Rule Extraction"])

def generate_spring_boot_project(project_info):
    url = "https://start.spring.io/starter.zip"
    response = requests.post(url, params=project_info)
    if response.status_code == 200:
        return response.content
    else:
        st.error("Error generating Spring Boot project from Spring Initializr.")
        return None

def convert_cpp_to_java_spring_boot(cpp_code, filename, HUGGINGFACE_API_TOKEN, project_info):
    try:
        progress_bar = st.progress(0)
        progress_stage = 0

        with st.spinner("Processing and converting code..."):
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 1: Splitting the code into chunks... 💬")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(cpp_code)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 2: Generating embeddings... 📊")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embeddings)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 3: Loading the language model... 🚀")

            llm = HuggingFaceEndpoint(
                #repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
                repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                max_new_tokens=2048,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.03,
                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
            )

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 4: Converting C++ to Java Spring Boot... 🔄")

            prompt = f"""
Convert the following C++ code into Java Spring Boot. Generate separate classes only if needed by the logic of the C++ code, avoiding unnecessary layers.
Only generate a separate `Controller`, `Service`, and `Repository` if the C++ code includes logic for handling HTTP requests, database interactions, or business logic. If the code is simple (e.g., "Hello World"), convert it within a single `MainApplication` class.

Ensure to:
1. Conditionally create classes for each layer (e.g., 'Controller', 'Service', 'Entity', 'Repository') based on the complexity of the C++ code.
2. Include 'application.yaml' and 'pom.xml' with only the required dependencies.
3. Annotate each class appropriately (e.g., '@SpringBootApplication', '@RestController', '@Service', '@Entity', '@Repository', etc.).
4. Avoid generating duplicate code; ensure each logic appears only once.
5. Generate distinct downloadable files for each of the following: Controller, Service, Repository, Entity, the main Spring Boot application class (with the main method), pom.xml with needed dependencies, and application.yaml file if applicable.
6. Instead of a single Java file, generate multiple files for each service, controller, repository, entity, pom.xml, and application.yaml.

Additionally, incorporate deployment and scaling considerations:
1. Ensure the generated application is container-friendly by including a `Dockerfile` and a `docker-compose.yml` file for deployment.
2. Optimize application.yaml for production readiness, such as externalized configuration (e.g., environment variables) and default profiles for development and production.
3. Include comments or configurations that enable horizontal scaling (e.g., thread pool settings, database connection pool settings, etc.).
4. If applicable, add configurations for readiness and liveness probes to support Kubernetes deployments.
5. Generate guidelines or a deployment script (e.g., Bash or YAML) to deploy the application to Kubernetes, emphasizing scalability and resilience.

Here is the C++ code snippet:
{cpp_code}
"""
            response = llm.invoke(prompt)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 5: Conversion complete! 🎉")

            # Parsing response to identify components
            components = {}
            lines = response.splitlines()
            current_class = None

            for line in lines:
                if line.startswith("public class ") or line.startswith("class "):
                    current_class = line.split()[2].strip()  # Extract class name
                    components[current_class] = []

                    # Check if this class should be annotated as an entity
                    if "Model" in current_class or "Entity" in current_class:
                        components[current_class].append("import javax.persistence.Entity;")
                        components[current_class].append("@Entity")  # Adding the entity annotation
                if current_class:
                    components[current_class].append(line)

            # Step 6: Generate Spring Boot project
            st.success("Step 6: Generating Spring Boot project... 📦")
            spring_boot_project = generate_spring_boot_project(project_info)
            
            if spring_boot_project:
                zip_buffer = io.BytesIO()
                zip_filename = filename.rsplit('.', 1)[0] + '.zip'
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    # Save each converted Java class to the zip file
                    package_path = project_info['packageName'].replace('.', '/')

                    for class_name, class_lines in components.items():
                        class_code = "\n".join(class_lines)

                         # Determine the class type based on naming conventions
                        if "Controller" in class_name:
                            class_path = f"{class_name}.java"  # Place directly in the project root
                        elif "Service" in class_name:
                            class_path = f"{class_name}.java"  # Place directly in the project root
                        elif "Repository" in class_name:
                            class_path = f"{class_name}.java"  # Place directly in the project root
                        elif "Model" in class_name or "Entity" in class_name:
                            class_path = f"{class_name}.java"  # Place directly in the project root
                        else:
                            class_path = f"{class_name}.java"  # Default case

                        zip_file.writestr(class_path, class_code)

                        # Individual download for each class
                        st.download_button(
                            label=f"Download {class_name}.java",
                            data=class_code,
                            file_name=f"{class_name}.java",
                            mime="text/x-java-source"
                        )

                    # Add Spring Boot project zip content
                    zip_file.writestr("spring-boot-project.zip", spring_boot_project)

                zip_buffer.seek(0)  # Move to the beginning of the BytesIO buffer

                # Download button for the complete project zip file
                st.download_button(
                    label="Download Complete Spring Boot Project",
                    data=zip_buffer,
                    file_name=zip_filename,
                    mime="application/zip"
                )
            else:
                st.error("Failed to generate the Spring Boot project.")

            # Display the converted Java code
            st.code(response, language='java')

            if re.search(r'\berror\b|\bexception\b|\bsyntax\b|\bmissing\b', response.lower()):
                st.warning("The converted Java code may contain syntax or structural errors. Please review it carefully.")
            else:
                st.success("The Java code is free from basic syntax errors!")

    except Exception as e:
        logger.error(f"An error occurred while converting the code: {e}", exc_info=True)
        st.error("Unable to convert C++ code to Java.")

# Function to detect potential bugs in Java code (simplified example)
def detect_bugs_in_code(code: str):
    bugs_detected = []

    # Concurrency issues
    if "synchronized" not in code and "Thread" in code:
        bugs_detected.append("Potential concurrency issue detected: Missing synchronization on shared resources.")

    # Memory leaks
    if "new" in code and "close" not in code:
        bugs_detected.append("Potential memory leak detected: Object allocated without proper cleanup (missing close).")

    # Exception handling
    if "catch (Exception" in code:
        bugs_detected.append("Potential issue in exception handling: Catching general Exception. Consider catching more specific exceptions.")

    # Database issues
    if "DataSource" in code or "Connection" in code:
        if "close()" not in code:
            bugs_detected.append("Potential database connection leak: Ensure all connections are closed.")

    # REST API issues
    if "@RequestBody" in code or "@RequestParam" in code:
        if "@Valid" not in code:
            bugs_detected.append("Unvalidated input detected: Missing @Valid annotation on REST API parameters.")

    # Security issues
    if "password" in code or "secret" in code:
        bugs_detected.append("Security issue detected: Hardcoded credentials found.")

    return bugs_detected


# Function to suggest fixes for bugs
def suggest_bug_fix(bug_description: str, code: str, llm):
    try:
        # Form the prompt for the LLM to suggest fixes
        prompt = f"""
        You are an expert Java developer and a code quality analyst. The following potential issue was detected:

        Issue: {bug_description}

        The code is:
        {code}

        Suggest a fix or provide a patch to resolve the issue. If the issue requires a detailed explanation, Include detailed reasoning and alternative solutions.
        """

        # Request a fix suggestion from the LLM
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        logger.error(f"Error generating fix: {e}")
        return "Error occurred while generating fix."


# Function to categorize bug severity
def categorize_bug_severity(bug_description: str) -> str:
    if "memory leak" in bug_description.lower():
        return "Critical"
    elif "security" in bug_description.lower():
        return "High"
    elif "unvalidated input" in bug_description.lower():
        return "Medium"
    elif "formatting" in bug_description.lower():
        return "Low"
    return "Medium"


# Function to simulate confidence score for a fix
def get_fix_confidence(llm_response: str) -> float:
    # Simulated logic: longer responses may indicate higher confidence
    return min(100, len(llm_response) / 10.0)


# Function to generate metrics dashboard
def generate_metrics_dashboard(severity_counts: dict, fix_confidences: list, total_files: int):
    # Metrics summary
    st.subheader("Metrics Summary")
    st.write(f"Total Files Processed: {total_files}")
    st.write(f"Total Bugs Detected: {sum(severity_counts.values())}")
    st.write(f"Average Fix Confidence: {sum(fix_confidences) / len(fix_confidences):.2f}%")

    # Plot severity distribution
    st.subheader("Bug Severity Distribution")
    fig, ax = plt.subplots()
    ax.bar(severity_counts.keys(), severity_counts.values(), color=["red", "orange", "yellow", "green"])
    ax.set_title("Bug Severity Distribution")
    ax.set_ylabel("Number of Bugs")
    ax.set_xlabel("Severity Level")
    st.pyplot(fig)

    # Plot confidence scores
    st.subheader("Fix Confidence Distribution")
    fig, ax = plt.subplots()
    sns.histplot(fix_confidences, kde=True, bins=10, ax=ax, color="blue")
    ax.set_title("Fix Confidence Distribution")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


# Bug detection page logic
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

        # Detect bugs in each class
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

                        # Detect bugs in the current class
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

            # Display detected bugs and suggested fixes
            if bugs_and_fixes:
                for class_name, bug_fixes in bugs_and_fixes.items():
                    st.subheader(f"Bugs in {class_name}:",divider=True)
                    for bug, details in bug_fixes.items():
                        st.write(f"Bug: {bug}")
                        st.write(f"Severity: {details['severity']}")
                        st.write(f"Suggested Fix: {details['fix']}")
                        st.write(f"Fix Confidence: {details['confidence']:.2f}%")

                # Generate metrics dashboard
                generate_metrics_dashboard(severity_counts, fix_confidences, total_files)

                # Provide download option
                st.subheader("Download Results")
                download_data = []

                # Prepare data for download
                for class_name, bug_fixes in bugs_and_fixes.items():
                    for bug, details in bug_fixes.items():
                        download_data.append({
                            "Class Name": class_name,
                            "Bug": bug,
                            "Severity": details['severity'],
                            "Suggested Fix": details['fix'],
                            "Fix Confidence (%)": f"{details['confidence']:.2f}",
                        })

                # Convert to DataFrame for CSV download
                df = pd.DataFrame(download_data)
                csv_data = df.to_csv(index=False)
                json_data = json.dumps(bugs_and_fixes, indent=4)

                # Allow user to download as CSV
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name="bug_fix_results.csv",
                    mime="text/csv",
                )

                # Allow user to download as JSON
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name="bug_fix_results.json",
                    mime="application/json",
                )

            else:
                st.write("No bugs detected in the provided code.")


# Function to extract business rules from Java class code using LLM
def extract_business_rules(class_name, class_code, llm):
    try:
        prompt = f"""
        You are an expert in extracting business rules from Java code. Analyze the following Java class and extract the business rules or important business logic.

        Class to analyze:
        {class_code}

        Extract the rules and logic in simple, clear statements. Provide each business rule separately.
        """
        
        response = llm.invoke(prompt)
        if not response.strip():
            return f"// No business rules found for {class_name}."
        return response
    except Exception as e:
        logger.error(f"Error extracting business rules for {class_name}: {e}")
        return f"// Error occurred during business rule extraction for {class_name}."

# Page 2: Business rule extraction logic (new code)
def business_rule_extraction_page(uploaded_file):
    if uploaded_file:
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                project_dir = "C:\\Temp\\springboot_project_rules"
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

        # Extract business rules for each Java class
        if st.button("Extract Business Rules"):
            business_rules = {}
            progress_bar = st.progress(0)
            files_processed = 0

            with st.spinner("Extracting business rules..."):
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
                        class_name = os.path.basename(file_path).replace(".java", ".txt")

                        # Extract business rules for the current class
                        business_rule_text = extract_business_rules(class_name, class_content, llm)
                        business_rules[class_name] = business_rule_text

                        # Display individual business rule as downloadable content
                        st.download_button(
                            label=f"Download Business Rules for {class_name}",
                            data=business_rule_text,
                            file_name=class_name,
                            mime="text/plain",
                        )

                        # Update progress bar
                        files_processed += 1
                        progress_bar.progress(files_processed / total_files)
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")

            # Create a downloadable zip of all business rules
            if business_rules:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for rule_class, rule_content in business_rules.items():
                        rule_file_path = f"business_rules/{rule_class}"
                        zip_file.writestr(rule_file_path, rule_content)

                zip_buffer.seek(0)
                st.download_button(
                    label="Download All Business Rules (ZIP)",
                    data=zip_buffer,
                    file_name="business_rules.zip",
                    mime="application/zip",
                )
                st.success("Business rules extracted successfully!")
            else:
                st.error("No valid Java classes found in the project.")


# Page 1: File Upload Converter
if page == "File Upload Converter":
    st.header("C++ to Java Conversion Tool with LLM 💻")

    with st.sidebar:        
      
        with st.expander("Spring Initializr", expanded=True):
            st.subheader("Spring Boot Project Metadata")
            group_id = st.text_input("Group ID", "com.example")
            artifact_id = st.text_input("Artifact ID", "demo")
            name = st.text_input("Project Name", "Demo Project")
            packaging = st.selectbox("Packaging", ["jar", "war"])
            dependencies = st.multiselect("Select Dependencies", ["web", "data-jpa", "mysql", "h2", "thymeleaf"])
        
        with st.expander("Upload Your C++ Code", expanded=True):
            file = st.file_uploader("Upload a C++ file (.cpp) to start analyzing", type="cpp")

            if file is not None:
                try:
                    code_content = file.read().decode("utf-8")
                    st.subheader("C++ Code Preview")
                    st.code(code_content[:5000], language='cpp')
                except Exception as e:
                    logger.error(f"An error occurred while reading the code file: {e}", exc_info=True)
                    st.warning("Unable to display code preview.")

    with st.expander("Tutorials & Tips", expanded=True):
        st.write("""### Welcome to the C++ to Java Conversion Tool!
        Here are some tips to help you use this tool effectively:
        - **Code Formatting:** Ensure your C++ code is properly formatted.
        - **Chunking:** Break large files into smaller parts.
        - **Annotations:** Ensure the Java conversion includes necessary annotations like `@RestController`.
        - **Testing:** Test the Java code after conversion.
        - **Documentation:** Familiarize with C++ and Java Spring Boot docs.
        """)

    if file is not None:
        if st.button("Convert C++ to Java Spring Boot"):            
            project_info = {
                'type': 'maven-project',
                'groupId': group_id,
                'artifactId': artifact_id,
                'name': name,
                'packageName': group_id,
                'version': '0.0.1-SNAPSHOT',
                'packaging': packaging,
                'dependencies': ','.join(dependencies)
            }
            convert_cpp_to_java_spring_boot(code_content, file.name, HUGGINGFACE_API_TOKEN, project_info)

# Page 2: Inline Code Converter
if page == "Inline Code Converter":
    st.header("Inline C++ to Java Code Converter 💻")

    cpp_code_input = st.text_area("Enter C++ Code to Convert to Java Spring Boot", height=300)

    with st.sidebar:
        st.subheader("Spring Boot Project Metadata")
        group_id = st.text_input("Group ID", "com.example")
        artifact_id = st.text_input("Artifact ID", "demo")
        name = st.text_input("Project Name", "Demo Project")
        packaging = st.selectbox("Packaging", ["jar", "war"])
        dependencies = st.multiselect("Select Dependencies", ["web", "data-jpa", "mysql", "h2", "thymeleaf"])
        

    if st.button("Convert Inline C++ to Java Spring Boot"):
        if cpp_code_input.strip():
            project_info = {
                'type': 'maven-project',
                'groupId': group_id,
                'artifactId': artifact_id,
                'name': name,
                'packageName': group_id,
                'version': '0.0.1-SNAPSHOT',
                'packaging': packaging,
                'dependencies': ','.join(dependencies)
            }
            convert_cpp_to_java_spring_boot(cpp_code_input, "inline_code_conversion.cpp", HUGGINGFACE_API_TOKEN, project_info)
        else:
            st.warning("Please enter some C++ code to convert.")

#Page 3
elif page == "Bug Detection":
    # Existing bug detection page
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    bug_detection_page(uploaded_file)

#Page 4
elif page == "Business Rule Extraction":
    # New business rule extraction page
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    business_rule_extraction_page(uploaded_file)
         
# Footer
st.sidebar.markdown("### About")
st.sidebar.write("This tool uses state-of-the-art AI models to assist with C++ to Java conversion, specifically tailored for Spring Boot applications.")
