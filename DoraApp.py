import logging
import streamlit as st
import os
import io
import zipfile
from langchain.llms import HuggingFaceEndpoint
import matplotlib.pyplot as plt
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace API Token
HUGGINGFACE_API_TOKEN = "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"  # Add your Hugging Face API token

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set the Hugging Face API token.")

st.set_page_config(page_title="DORA Metrics Analyzer", page_icon="📊", layout="wide")

# Header
st.header("DORA Metrics Analyzer 📊")

# LLM Initialization
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

# Function to generate insights using LLM
def generate_insights(metrics: dict, llm):
    try:
        prompt = f"""
        Analyze the following DORA metrics for a Spring Boot project and provide detailed recommendations to improve performance, reliability, and deployment frequency. 

        Metrics:
        {metrics}
        """
        response = llm.invoke(prompt)
        return response.strip() if response else "No insights generated."
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        return "Error occurred while generating insights."

# Page logic for DORA analysis
def dora_analysis_page(uploaded_file):
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

        # Dummy metrics (replace with actual metrics calculation logic)
        metrics = {
            "Deployment Frequency": "Once a week",
            "Lead Time for Changes": "2 days",
            "Change Failure Rate": "5%",
            "Time to Restore Service": "1 hour",
        }

        # Display metrics
        st.subheader("DORA Metrics")
        st.json(metrics)

        # Generate insights using LLM
        if st.button("Generate Insights"):
            with st.spinner("Generating insights..."):
                insights = generate_insights(metrics, llm)
                st.subheader("Insights and Recommendations")
                st.write(insights)

# Main page logic
page = st.sidebar.selectbox("Select a Page", ("DORA Analysis", "Metrics Visualization"))

if page == "DORA Analysis":
    uploaded_file = st.file_uploader("Upload Spring Boot Project (ZIP)", type="zip")
    dora_analysis_page(uploaded_file)

elif page == "Metrics Visualization":
    st.write("Metrics visualization page coming soon!")
