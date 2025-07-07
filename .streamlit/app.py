import streamlit as st
from main import collect_reddit_data, clean_text, analyze_text_with_groq, cluster_pain_points, generate_funnels_with_groq, create_marketing_snapshot
import json
import os

# Streamlit UI
st.set_page_config(page_title="Reddit Funnel Visualizer", layout="centered")
st.title("📊 Reddit Funnel Visualizer")
st.markdown("Generate a 3-step marketing funnel from Reddit comments.")

# Inputs
subreddit = st.text_input("Enter subreddit name:", value="deeplearning")
limit = st.slider("Number of posts to fetch:", min_value=5, max_value=50, value=10)

# Run pipeline
if st.button("Generate Funnel"):
    with st.spinner("Fetching and analyzing Reddit data..."):
        data = collect_reddit_data(subreddit_name=subreddit, limit=limit)

        if not data:
            st.error("No data collected from Reddit. Try another subreddit.")
        else:
            max_words = 100
            cleaned = []
            for item in data:
                text = clean_text(item["title"] + " " + item["body"])
                if text:
                    truncated = " ".join(text.split()[:max_words])
                    cleaned.append(truncated)

            analyzed = analyze_text_with_groq(cleaned)
            pain_points = cluster_pain_points(analyzed)
            funnels = generate_funnels_with_groq(pain_points)
            create_marketing_snapshot(funnels)

            st.success("Funnel generated!")

            # Show Image
            if os.path.exists("output/marketing_snapshot.png"):
                st.image("output/marketing_snapshot.png", caption="Generated Funnel Snapshot")
            else:
                st.warning("Snapshot image not found.")
