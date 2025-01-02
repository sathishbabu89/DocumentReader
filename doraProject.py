import os
import shutil
import streamlit as st
from github import Github
from git import Repo
from datetime import datetime
from langchain.llms import HuggingFaceEndpoint

# Hugging Face API Token (replace with your own token)
HUGGINGFACE_API_TOKEN = "your_huggingface_token"

def extract_repository_data(repo_url, github_token):
    try:
        # Clone the repository
        clone_dir = "/tmp/github_repo"
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        Repo.clone_from(repo_url, clone_dir)

        # Initialize GitHub API
        g = Github(github_token)
        repo_name = repo_url.rstrip("/").split("/")[-1]
        repo = g.get_repo(repo_name)

        # Extract data from the repository
        tags = repo.get_tags()
        commits = repo.get_commits()
        issues = repo.get_issues(state="closed")

        # Format data for LLM input
        tag_dates = [tag.commit.commit.author.date.strftime("%Y-%m-%d") for tag in tags]
        commit_messages = [commit.commit.message for commit in commits]
        issue_summaries = [
            f"Issue: {issue.title}\nBody: {issue.body}\nCreated: {issue.created_at}\nClosed: {issue.closed_at}"
            for issue in issues
        ]

        return tag_dates, commit_messages, issue_summaries

    except Exception as e:
        st.error(f"Error extracting repository data: {e}")
        return None, None, None

def analyze_dora_metrics_with_llm(tag_dates, commit_messages, issue_summaries):
    try:
        # Initialize LLM
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-2-7B-Instruct",
            max_new_tokens=2048,
            temperature=0.7,
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        )

        # Prepare the prompt
        prompt = f"""
        You are an expert DevOps engineer. Analyze the following repository data to calculate DORA metrics:

        1. **Deployment Frequency**: Calculate the average number of deployments per month based on the tag dates.
        2. **Lead Time for Changes**: Calculate the average time between a commit and its deployment.
        3. **Change Failure Rate**: Estimate the percentage of deployments that fail based on the provided issues.
        4. **Mean Time to Recovery (MTTR)**: Calculate the average time to resolve incidents based on issue resolution times.

        **Tag Dates**: {tag_dates}

        **Commit Messages**:
        {commit_messages[:10]}  # Display only the first 10 messages for brevity.

        **Issue Summaries**:
        {issue_summaries[:5]}  # Display only the first 5 issues for brevity.

        Provide a detailed analysis of each metric with calculations.
        """

        # Get LLM response
        response = llm.invoke(prompt)

        return response

    except Exception as e:
        st.error(f"Error analyzing DORA metrics: {e}")
        return None

# Streamlit App
st.title("DORA Metrics Calculator with LLM")
repo_url = st.text_input("Enter GitHub Repository URL:")
github_token = st.text_input("Enter GitHub PAT Token:", type="password")

if st.button("Calculate DORA Metrics"):
    if not repo_url or not github_token:
        st.error("Please provide both the repository URL and GitHub PAT token.")
    else:
        with st.spinner("Extracting repository data..."):
            tag_dates, commit_messages, issue_summaries = extract_repository_data(repo_url, github_token)

        if tag_dates and commit_messages and issue_summaries:
            with st.spinner("Analyzing DORA metrics with LLM..."):
                metrics_analysis = analyze_dora_metrics_with_llm(tag_dates, commit_messages, issue_summaries)

            if metrics_analysis:
                st.success("DORA Metrics Analysis Completed!")
                st.write(metrics_analysis)
