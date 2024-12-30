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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_API_TOKEN = "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"  # Add your Hugging Face API token

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

# Use the actual emoji instead of surrogate Unicode characters
st.set_page_config(page_title="Spring Boot Project Analyzer", page_icon="📊", layout="wide")

st.header("Spring Boot Project Analyzer 📊")


# Page 1: Test case generation logic (existing code) Function to generate test cases using LLM
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
                #project_dir = "/tmp/springboot_project"             
                project_dir = "C:\\Temp\\springboot_project"
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

#Page 2

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



# Page 2: Project Analysis

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

# Page 2: Project Analysis
def project_analysis_page(uploaded_file):
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Extract the project ZIP file
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        project_dir = "C:\\Temp\\springboot_project_analysis"
        os.makedirs(project_dir, exist_ok=True)
        zip_ref.extractall(project_dir)

    # Update progress bar after extraction
    progress_bar.progress(30)
    status_text.text("Project ZIP extracted. Analyzing project...")

    # LLM initialization
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
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
                project_dir = "C:\\Temp\\springboot_project"
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
                project_dir = "C:\\Temp\\springboot_project_microservices"                
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
    test_case_generator_page(uploaded_file)

elif page == "Project Analysis":
    # Upload project ZIP for analysis
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    
    if uploaded_file:
        project_analysis_page(uploaded_file)

elif page == "CI/CD Pipeline Generator":
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    cicd_pipeline_generator_page(uploaded_file)

elif page == "Microservice Recommendations":
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    microservice_recommendations_page(uploaded_file)
