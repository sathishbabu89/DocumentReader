import logging
import streamlit as st
import re
import torch
import zipfile
import io
import os
import requests
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

st.set_page_config(page_title="COBOL to Java Conversion Tool", page_icon="💻")

page = st.sidebar.selectbox("Choose Page", ["File Upload Converter", "Inline Code Converter"])

def generate_spring_boot_project(project_info):
    url = "https://start.spring.io/starter.zip"
    response = requests.post(url, params=project_info)
    if response.status_code == 200:
        return response.content
    else:
        st.error("Error generating Spring Boot project from Spring Initializr.")
        return None

def convert_cobol_to_java_spring_boot(cobol_code, filename, HUGGINGFACE_API_TOKEN, project_info):
    try:
        progress_bar = st.progress(0)
        progress_stage = 0

        with st.spinner("Processing and converting code..."):
            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 1: Splitting the code into chunks... 💬")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(cobol_code)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 2: Generating embeddings... 📊")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embeddings)

            progress_stage += 20
            progress_bar.progress(progress_stage)
            st.info("Step 3: Loading the language model... 🚀")

            llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
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
            st.info("Step 4: Converting COBOL to Java Spring Boot... 🔄")

            prompt = f"""
Convert the following COBOL code into Java Spring Boot. Generate separate classes only if needed by the logic of the COBOL code, avoiding unnecessary layers.
Only generate a separate `Controller`, `Service`, and `Repository` if the COBOL code includes logic for handling HTTP requests, database interactions, or business logic. If the code is simple (e.g., "Hello World"), convert it within a single `MainApplication` class.

Ensure to:
1. Conditionally create classes for each layer(e.g., 'Controller', 'Service', ' Entity', 'Repository' ) based on the complexity of COBOL code.
2. Include 'application.yaml' and 'pom.xml' with only the required dependencies
3. Annotate each class appropriately (e.g., '@SpringBootApplication' , '@RestController' , '@Service' , '@Entity' ,'@Repository', etc.)
4. Avoid generating duplicate codes; ensure each logic appears only once
5. Generate distinct downloadable files for each of the following: Controller, Service, Repository, Entity, the main SpringBoot application class which has main method, pom.xml with needed dependencies and application.yaml file if applicable
Here is the COBOL code snippet:
{cobol_code}
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
        st.error("Unable to convert COBOL code to Java.")


# Page 1: File Upload Converter
if page == "File Upload Converter":
    st.header("COBOL to Java Conversion Tool with LLM 💻")

    with st.sidebar:        
      
        with st.expander("Spring Initializr", expanded=True):
            st.subheader("Spring Boot Project Metadata")
            group_id = st.text_input("Group ID", "com.example")
            artifact_id = st.text_input("Artifact ID", "demo")
            name = st.text_input("Project Name", "Demo Project")
            packaging = st.selectbox("Packaging", ["jar", "war"])
            dependencies = st.multiselect("Select Dependencies", ["web", "data-jpa", "mysql", "h2", "thymeleaf"])
        
        with st.expander("Upload Your COBOL Code", expanded=True):
            file = st.file_uploader("Upload a COBOL file (.cob) to start analyzing", type="cob")

            if file is not None:
                try:
                    code_content = file.read().decode("utf-8")
                    st.subheader("COBOL Code Preview")
                    st.code(code_content[:5000], language='cobol')
                except Exception as e:
                    logger.error(f"An error occurred while reading the code file: {e}", exc_info=True)
                    st.warning("Unable to display code preview.")

    with st.expander("Tutorials & Tips", expanded=True):
        st.write("""### Welcome to the COBOL to Java Conversion Tool!
        Here are some tips to help you use this tool effectively:
        - **Code Formatting:** Ensure your COBOL code is properly formatted.
        - **Chunking:** Break large files into smaller parts.
        - **Annotations:** Ensure the Java conversion includes necessary annotations like `@RestController`.
        - **Testing:** Test the Java code after conversion.
        - **Documentation:** Familiarize with COBOL and Java Spring Boot docs.
        """)

    if file is not None:
        if st.button("Convert COBOL to Java Spring Boot"):            
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
            convert_cobol_to_java_spring_boot(code_content, file.name, HUGGINGFACE_API_TOKEN, project_info)

# Page 2: Inline Code Converter
if page == "Inline Code Converter":
    st.header("Inline COBOL to Java Code Converter 💻")

    cobol_code_input = st.text_area("Enter COBOL Code to Convert to Java Spring Boot", height=300)

    with st.sidebar:
        st.subheader("Spring Boot Project Metadata")
        group_id = st.text_input("Group ID", "com.example")
        artifact_id = st.text_input("Artifact ID", "demo")
        name = st.text_input("Project Name", "Demo Project")
        packaging = st.selectbox("Packaging", ["jar", "war"])
        dependencies = st.multiselect("Select Dependencies", ["web", "data-jpa", "mysql", "h2", "thymeleaf"])

    if st.button("Convert Inline COBOL to Java Spring Boot"):
        if cobol_code_input.strip():
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
            convert_cobol_to_java_spring_boot(cobol_code_input, "inline_code_conversion.cob", HUGGINGFACE_API_TOKEN, project_info)
        else:
            st.warning("Please enter some COBOL code to convert.")

# Footer
st.sidebar.markdown("### About")
st.sidebar.write("This tool uses state-of-the-art AI models to assist with COBOL to Java conversion, specifically tailored for Spring Boot applications.")
