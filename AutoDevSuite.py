import logging
import streamlit as st
import zipfile
import io
import os
import networkx as nx
import matplotlib.pyplot as plt
from langchain.llms import HuggingFaceEndpoint
import ast
from typing import List, Dict

# Force PyTorch to use CPU
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = "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"  # Add your Hugging Face API token

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

st.set_page_config(page_title="Spring Boot Project Analyzer", page_icon="📊", layout="wide")

st.header("Spring Boot Project Analyzer 📊")

# Function to analyze the Java files and extract class/interface relationships
def analyze_project_classes_and_interfaces(project_dir: str) -> Dict:
    classes = {}
    interfaces = {}
    relationships = []

    # Walk through the project directory to find all Java files
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".java"):
                class_name = os.path.basename(file).replace(".java", "")
                with open(os.path.join(root, file), "r") as f:
                    content = f.read()

                # Detect if it's a class or interface and gather relationships
                if "class" in content:
                    classes[class_name] = content
                elif "interface" in content:
                    interfaces[class_name] = content

                # Check for references to other classes or interfaces
                if "implements" in content:
                    # Extract interface names that are implemented
                    implemented_interfaces = [word.strip() for word in content.split("implements")[1].split(",")]
                    for iface in implemented_interfaces:
                        relationships.append((class_name, iface))

                if "extends" in content:
                    # Extract the class names that are extended
                    extended_classes = [word.strip() for word in content.split("extends")[1].split(",")]
                    for ext_class in extended_classes:
                        relationships.append((class_name, ext_class))

    return classes, interfaces, relationships

# Function to generate a class/interface relationship diagram using matplotlib
def generate_class_diagram(classes: Dict[str, str], interfaces: Dict[str, str], relationships: List[tuple]):
    G = nx.DiGraph()

    # Add nodes for classes and interfaces
    for class_name in classes:
        G.add_node(class_name, type='class')
    for interface_name in interfaces:
        G.add_node(interface_name, type='interface')

    # Add edges to represent relationships (implements/extends)
    for from_class, to_class in relationships:
        G.add_edge(from_class, to_class)

    # Create a plot
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")
    plt.title("Class and Interface Relationships", size=15)

    # Save the plot to a file
    img_path = "/tmp/class_diagram.png"
    plt.savefig(img_path, format="PNG")
    plt.close()  # Close the plot to free memory

    return img_path

# Function to generate business summary using LLM
def generate_business_summary(classes: Dict[str, str], interfaces: Dict[str, str], llm):
    try:
        # Creating a prompt for the business summary based on classes and interfaces
        class_names = ', '.join(classes.keys())
        interface_names = ', '.join(interfaces.keys())

        prompt = f"""
        You are a software architect. Provide a business summary for the following Spring Boot project. 
        Consider the classes and interfaces as key components of the application.

        Classes: {class_names}
        Interfaces: {interface_names}

        Based on the classes and interfaces, generate a business overview of what this project does.
        """

        # Send the prompt to the LLM
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating business summary: {e}")
        return "Error occurred while generating the business summary."

# Display project summary and relationships
def display_project_summary(classes: Dict[str, str], interfaces: Dict[str, str], relationships: List[tuple]):
    # Display total counts of classes and interfaces
    st.subheader(f"Classes: {len(classes)}")
    st.subheader(f"Interfaces: {len(interfaces)}")

    # Generate a summary of the overall project functionality
    st.subheader("Project Functionality Overview:")
    project_overview = "This Spring Boot project contains the following key components:\n\n"
    project_overview += f"Classes: {', '.join(classes.keys())}\n"
    project_overview += f"Interfaces: {', '.join(interfaces.keys())}\n\n"
    project_overview += "The relationships between classes and interfaces are displayed below."
    st.write(project_overview)

    # Generate and display the class/interface relationship diagram
    img_path = generate_class_diagram(classes, interfaces, relationships)
    st.image(img_path, use_column_width=True)

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


# Page 1: Test case generation logic (existing code)
def test_case_generator_page(uploaded_file):
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


# Function to generate API documentation using LLM
def generate_api_documentation(classes: Dict[str, str], llm):
    try:
        # Creating a prompt for API documentation based on classes
        class_names = ', '.join(classes.keys())
        prompt = f"""
        You are a software documentation expert. Generate comprehensive API documentation for the following Spring Boot classes:

        Classes: {class_names}

        For each class, provide the following:
        - Class description
        - Public methods with their descriptions, parameters, and return types
        - Any annotations used (e.g., @RestController, @Service)
        - Example usage if applicable
        """

        # Send the prompt to the LLM
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating API documentation: {e}")
        return "Error occurred while generating API documentation."

# Function to generate inline comments for the code using LLM
def generate_inline_comments(class_code: str, llm):
    try:
        prompt = f"""
        You are a Java Spring Boot expert. Please add inline comments to the following code to explain its functionality, especially for public methods, important logic, and annotations:

        {class_code}
        """

        # Send the prompt to the LLM
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating inline comments: {e}")
        return "Error occurred while generating inline comments."

# Function to generate a README file using LLM
def generate_readme(classes: Dict[str, str], llm):
    try:
        # Creating a prompt for generating a README file
        class_names = ', '.join(classes.keys())
        prompt = f"""
        You are an expert in software documentation. Generate a comprehensive README file for the following Spring Boot project. 
        The README should include:
        - Project Overview
        - Setup Instructions
        - Usage Instructions
        - API Documentation Overview
        - List of Classes: {class_names}

        Ensure the README is clear and well-organized, suitable for developers and end users.
        """

        # Send the prompt to the LLM
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating README file: {e}")
        return "Error occurred while generating README file."

# Function to extract Java files from the ZIP and generate documentation
def generate_documentation(uploaded_file):
    if uploaded_file:
        # Initialize progress bar
        progress_bar = st.progress(0)
        try:    
            # Extract the ZIP file
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                project_dir = "/tmp/springboot_project"
                os.makedirs(project_dir, exist_ok=True)
                zip_ref.extractall(project_dir)
                st.success("Project extracted successfully!")
        except Exception as e:
            logger.error(f"Failed to extract ZIP file: {e}", exc_info=True)
            st.error("Invalid ZIP file. Please upload a valid Spring Boot project.")

        # Update progress bar after extraction
        progress_bar.progress(30)

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

        # Extract Java classes and generate documentation
        java_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(project_dir)
            for file in files if file.endswith(".java")
        ]
        classes = {}

        for file_path in java_files:
            try:
                with open(file_path, "r") as file:
                    class_content = file.read()
                class_name = os.path.basename(file_path).replace(".java", "")
                classes[class_name] = class_content
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

        # Update progress bar after analysis
        progress_bar.progress(60)
        # Generate the API documentation, inline comments, and README
        api_docs = generate_api_documentation(classes, llm)
        inline_comments = {}
        readme_content = generate_readme(classes, llm)

        for class_name, class_code in classes.items():
            comments = generate_inline_comments(class_code, llm)
            inline_comments[class_name] = comments

        # Display and allow downloading the generated documentation
        st.subheader("Generated API Documentation:")
        st.write(api_docs)

        st.subheader("Generated Inline Comments:")
        for class_name, comments in inline_comments.items():
            st.subheader(f"Inline Comments for {class_name}:")
            st.write(comments)

        st.subheader("Generated README File:")
        st.write(readme_content)

        # Allow downloading the generated files as a ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Write the README file
            zip_file.writestr("README.md", readme_content)

            # Write each class file with inline comments
            for class_name, comments in inline_comments.items():
                zip_file.writestr(f"src/main/java/{class_name}.java", comments)

        zip_buffer.seek(0)
        st.download_button(
            label="Download Documentation (ZIP)",
            data=zip_buffer,
            file_name="springboot_project_documentation.zip",
            mime="application/zip",
        )
        # Update progress bar after summary generation
        progress_bar.progress(100)
        st.success("Documentation generated successfully!")

# Select a page for the user
page = st.sidebar.selectbox("Select a Page", ("Test Case Generator", "Project Analysis" , "Docker & Kubernetes Config Generator", "Documentation Generator"))

if page == "Project Analysis":
    # Upload project ZIP for analysis
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")

    if uploaded_file:
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Extract the project ZIP file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            project_dir = "/tmp/springboot_project_analysis"
            os.makedirs(project_dir, exist_ok=True)
            zip_ref.extractall(project_dir)

        # Update progress bar after extraction
        progress_bar.progress(30)
        status_text.text("Project ZIP extracted. Analyzing project...")

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

        # Step 2: Analyze the project (classes, interfaces, relationships)
        classes, interfaces, relationships = analyze_project_classes_and_interfaces(project_dir)
        
        # Update progress bar after analysis
        progress_bar.progress(60)
        status_text.text("Project analysis complete. Generating business summary...")

        # Display project summary
        display_project_summary(classes, interfaces, relationships)

        # Step 3: Generate the business summary
        st.subheader("Business Summary:")
        business_summary = generate_business_summary(classes, interfaces, llm)

        # Update progress bar after summary generation
        progress_bar.progress(100)
        status_text.text("Business summary generated.")

        # Display the business summary
        st.write(business_summary)

elif page == "Test Case Generator":
    # Existing test case generation page
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    test_case_generator_page(uploaded_file)

elif page == "Documentation Generator":
    # Upload project ZIP for documentation generation
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    generate_documentation(uploaded_file)

elif page == "Docker & Kubernetes Config Generator":
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
