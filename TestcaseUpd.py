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

# Select a page for the user
page = st.sidebar.selectbox("Select a Page", ("Test Case Generator", "Project Analysis"))

if page == "Project Analysis":
    # Upload project ZIP for analysis
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")

    if uploaded_file:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            project_dir = "/tmp/springboot_project_analysis"
            os.makedirs(project_dir, exist_ok=True)
            zip_ref.extractall(project_dir)

        # Analyze the project and display the summary
        classes, interfaces, relationships = analyze_project_classes_and_interfaces(project_dir)
        display_project_summary(classes, interfaces, relationships)

elif page == "Test Case Generator":
    # Existing test case generation page
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    test_case_generator_page(uploaded_file)
