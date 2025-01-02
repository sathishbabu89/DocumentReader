import os
import time
import streamlit as st
from github import Github
from git import Repo
from datetime import datetime

def calculate_dora_metrics(repo_url, github_token):
    try:
        # Clone the repository
        clone_dir = "/tmp/github_repo"
        if os.path.exists(clone_dir):
            os.system(f"rm -rf {clone_dir}")
        Repo.clone_from(repo_url, clone_dir)

        # Initialize GitHub API
        g = Github(github_token)
        repo_name = repo_url.rstrip("/").split("/")[-1]
        repo = g.get_repo(repo_name)

        # Calculate Deployment Frequency
        tags = repo.get_tags()
        deployment_dates = [tag.commit.commit.author.date for tag in tags]
        deployment_dates.sort()

        deployment_frequency = len(deployment_dates) / len(set([date.strftime("%Y-%m") for date in deployment_dates]))

        # Calculate Lead Time for Changes
        commits = repo.get_commits()
        lead_times = []
        for commit in commits:
            commit_date = commit.commit.author.date
            deployment_date = next((d for d in deployment_dates if d >= commit_date), None)
            if deployment_date:
                lead_times.append((deployment_date - commit_date).total_seconds())

        avg_lead_time = sum(lead_times) / len(lead_times) if lead_times else 0

        # Estimate Change Failure Rate
        issues = repo.get_issues(state="closed")
        failure_count = sum(1 for issue in issues if "failure" in issue.title.lower() or "failure" in issue.body.lower())
        change_failure_rate = failure_count / len(issues) if issues.totalCount > 0 else 0

        # Calculate Mean Time to Recovery (MTTR)
        recovery_times = [
            (issue.closed_at - issue.created_at).total_seconds()
            for issue in issues if "incident" in issue.title.lower() or "incident" in issue.body.lower()
        ]
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0

        # Display Metrics
        st.success("DORA Metrics Calculated Successfully!")
        st.write(f"**Deployment Frequency:** {deployment_frequency:.2f} deployments per month")
        st.write(f"**Lead Time for Changes:** {avg_lead_time / 3600:.2f} hours")
        st.write(f"**Change Failure Rate:** {change_failure_rate:.2%}")
        st.write(f"**Mean Time to Recovery (MTTR):** {avg_recovery_time / 3600:.2f} hours")

    except Exception as e:
        st.error(f"Error calculating DORA metrics: {e}")

# Streamlit App
st.title("DORA Metrics Calculator")
repo_url = st.text_input("Enter GitHub Repository URL:")
github_token = st.text_input("Enter GitHub PAT Token:", type="password")

if st.button("Calculate DORA Metrics"):
    if not repo_url or not github_token:
        st.error("Please provide both the repository URL and GitHub PAT token.")
    else:
        calculate_dora_metrics(repo_url, github_token)
