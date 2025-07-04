import os
import json
import re
import time
import praw
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ========== SETUP ==========

reddit = praw.Reddit(
    client_id="NnQeVRnkCh9_cs2NoPuBgQ",
    client_secret="WA05tqa5oo6vGHoORJJoMZgQPb8HSQ",
    user_agent="chat_scrapper by u/YourUsername"
)

groq_client = Groq(api_key="")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ========== FUNNEL SCHEMA ==========

class FunnelStep(BaseModel):
    title: str = Field(description="Catchy step title")
    description: str = Field(description="Brief persuasive description")
    cta: str = Field(description="Call to action")

class Funnel(BaseModel):
    L0: FunnelStep
    L1: FunnelStep
    L2: FunnelStep

parser = PydanticOutputParser(pydantic_object=Funnel)

# ========== FUNCTIONS ==========

def collect_reddit_data(subreddit_name="machinelearning", limit=10):
    """Collect posts and comments from a specified subreddit."""
    try:
        data = []
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            post.comments.replace_more(limit=0)
            data.append({"id": post.id, "title": post.title, "body": post.selftext, "source": "post"})
            for comment in post.comments.list():
                data.append({"id": comment.id, "title": "", "body": comment.body, "source": "comment"})
        os.makedirs("data", exist_ok=True)
        with open("data/raw_data.json", "w") as f:
            json.dump(data, f, indent=2)
        return data
    except Exception as e:
        print(f"[Reddit Error] Failed to collect data: {e}")
        return []

def clean_text(text):
    """Clean and preprocess text for analysis."""
    if not isinstance(text, str) or len(text) < 10:
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens) if len(tokens) > 10 else ""

def analyze_text_with_groq(texts, batch_size=10, initial_delay=3, max_retries=3):
    """Analyze texts using Groq API for sentiment and key phrases with exponential backoff."""
    analyzed_texts = []
    os.makedirs("data", exist_ok=True)
    with open("data/raw_groq_responses.json", "w") as log_file:
        log_file.write("[]\n")  # Initialize log file
    raw_responses = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for text in batch:
            if not text:
                continue
            prompt = f"""
You are an AI assistant analyzing Reddit comments about machine learning challenges. 
Your task is to analyze the following comment and return a JSON object with:
- sentiment: "positive", "negative", or "neutral"
- key_phrases: a list of exactly 3 key phrases (each 2-3 words long)

**Instructions**:
- **IMPORTANT**: Return the response in valid JSON format, enclosed in ```json``` code blocks.
- Ensure the JSON object contains only the "sentiment" and "key_phrases" fields.
- If the text is unclear or insufficient, return "neutral" sentiment and an empty key_phrases list.
- Do not include any explanations or additional text outside the JSON.

**Text to analyze**:
\"\"\"{text[:500]}\"\"

**Example response**:
```json
{{
  "sentiment": "negative",
  "key_phrases": ["model overfitting", "data scarcity", "training time"]
}}
```
"""
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    response = groq_client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.7
                    )
                    raw = response.choices[0].message.content.strip()
                    raw_responses.append({"text": text[:60], "response": raw, "attempt": attempt + 1})
                    
                    # Save raw responses to file for debugging
                    with open("data/raw_groq_responses.json", "w") as log_file:
                        json.dump(raw_responses, log_file, indent=2)
                    
                    # Extract JSON from code block or directly if not in code block
                    json_match = re.search(r'```json\n(.*?)\n```', raw, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                    else:
                        # Attempt to find JSON without code block
                        json_str = re.search(r'\{.*\}', raw, re.DOTALL)
                        if json_str:
                            json_str = json_str.group(0)
                        else:
                            raise ValueError("No valid JSON found in response")
                    
                    result = json.loads(json_str)
                    analyzed_texts.append({
                        "text": text,
                        "sentiment": result.get("sentiment", "neutral"),
                        "key_phrases": result.get("key_phrases", [])
                    })
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"[Groq Error] {text[:60]}... → {e} (Attempt {attempt + 1}/{max_retries})")
                        analyzed_texts.append({"text": text, "sentiment": "neutral", "key_phrases": []})
                    else:
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
            time.sleep(initial_delay)
    
    os.makedirs("data", exist_ok=True)
    with open("data/analyzed_texts.json", "w") as f:
        json.dump(analyzed_texts, f, indent=2)
    return analyzed_texts

def cluster_pain_points(analyzed_texts, n_clusters=3):
    """Cluster texts based on key phrases to identify pain points."""
    texts = [item["text"] for item in analyzed_texts if item["key_phrases"]]
    key_phrases = [" ".join(item["key_phrases"]) for item in analyzed_texts if item["key_phrases"]]
    if len(key_phrases) < n_clusters:
        print(f"[Clustering Error] Not enough data for {n_clusters} clusters. Found {len(key_phrases)} items.")
        return []
    try:
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        X = vectorizer.fit_transform(key_phrases)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        pain_points = []
        for i in range(n_clusters):
            cluster_texts = [texts[j] for j in range(len(texts)) if labels[j] == i]
            cluster_phrases = [phrase for j in range(len(texts)) if labels[j] == i for phrase in analyzed_texts[j]["key_phrases"]]
            top_phrases = pd.Series(cluster_phrases).value_counts().head(3).index.tolist()
            pain_points.append({"cluster": i, "label": " ".join(top_phrases), "texts": cluster_texts})
        return pain_points
    except Exception as e:
        print(f"[Clustering Error] Failed to cluster data: {e}")
        return []

def generate_funnels_with_groq(pain_points):
    """Generate marketing funnels based on identified pain points."""
    funnels = []
    for point in pain_points:
        label = point["label"]
        prompt = f"""
You are an AI assistant creating a 3-step webinar funnel (Awareness, Consideration, Decision) for a machine learning pain point: "{label}"

**Instructions**:
- **IMPORTANT**: Return the response in valid JSON format, enclosed in ```json``` code blocks.
- Follow the exact schema provided below.
- Ensure each step has a catchy title, a brief persuasive description, and a call to action (CTA).
- Do not include any explanations or additional text outside the JSON.

**Schema**:
{parser.get_format_instructions()}

**Example response**:
```json
{{
  "L0": {{"title": "Discover Model Overfitting", "description": "Learn why models fail to generalize", "cta": "Join Webinar"}},
  "L1": {{"title": "Master Overfitting Solutions", "description": "Explore techniques to improve models", "cta": "Register Now"}},
  "L2": {{"title": "Implement Robust Models", "description": "Apply proven strategies to succeed", "cta": "Enroll Today"}}
}}
```
"""
        delay = 3
        for attempt in range(3):
            try:
                response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.9
                )
                raw = response.choices[0].message.content.strip()
                json_match = re.search(r'```json\n(.*?)\n```', raw, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    json_str = re.search(r'\{.*\}', raw, re.DOTALL)
                    if json_str:
                        json_str = json_str.group(0)
                    else:
                        raise ValueError("No valid JSON found in response")
                
                parsed = parser.parse(json_str)
                funnels.append({"pain_point": label, **parsed.dict()})
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[Funnel Error] {label} → {e} (Attempt {attempt + 1}/3)")
                    funnels.append({
                        "pain_point": label,
                        "L0": {"title": f"Overcome {label.title()}", "description": f"Learn to tackle {label} in ML", "cta": "Join Webinar"},
                        "L1": {"title": f"Master {label.title()}", "description": f"Deep dive into solving {label}", "cta": "Register Now"},
                        "L2": {"title": f"Implement {label.title()} Solutions", "description": f"Step-by-step plan for {label}", "cta": "Enroll Today"}
                    })
                time.sleep(delay)
                delay *= 2
    os.makedirs("data", exist_ok=True)
    with open("data/funnels.json", "w") as f:
        json.dump(funnels, f, indent=2)
    return funnels

def create_marketing_snapshot(funnels):
    """Create a visual snapshot of the first funnel."""
    if not funnels:
        print("[Snapshot Error] No funnels to visualize.")
        return
    funnel = funnels[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    text = (
        f"Pain Point: {funnel['pain_point'].title()}\n\n"
        f"Awareness\n{funnel['L0']['title']}\n{funnel['L0']['description']}\nCTA: {funnel['L0']['cta']}\n\n"
        f"Consideration\n{funnel['L1']['title']}\n{funnel['L1']['description']}\nCTA: {funnel['L1']['cta']}\n\n"
        f"Decision\n{funnel['L2']['title']}\n{funnel['L2']['description']}\nCTA: {funnel['L2']['cta']}"
    )
    ax.text(0.1, 0.95, text, fontsize=12, va="top", wrap=True)
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/marketing_snapshot.png", bbox_inches="tight", dpi=150)
    plt.close()

# ========== MAIN ==========

def main():
    print("✅ Collecting data from Reddit...")
    data = collect_reddit_data(subreddit_name="machinelearning", limit=50)
    if not data:
        print("❌ No data collected. Exiting.")
        return

    print("✅ Cleaning text...")
    max_words = 100
    cleaned = []
    for item in data:
        text = clean_text(item["title"] + " " + item["body"])
        if text:
            truncated = " ".join(text.split()[:max_words])
            cleaned.append(truncated)
    if not cleaned:
        print("❌ No valid texts after cleaning. Exiting.")
        return
    os.makedirs("data", exist_ok=True)
    with open("data/cleaned_texts.json", "w") as f:
        json.dump(cleaned, f, indent=2)

    print("✅ Analyzing texts with Groq...")
    analyzed = analyze_text_with_groq(cleaned)
    if not analyzed:
        print("❌ No texts analyzed. Exiting.")
        return

    print("✅ Clustering pain points...")
    pain_points = cluster_pain_points(analyzed)
    if not pain_points:
        print("❌ No pain points identified. Exiting.")
        return

    print("✅ Generating webinar funnels...")
    funnels = generate_funnels_with_groq(pain_points)

    print("✅ Creating marketing snapshot...")
    create_marketing_snapshot(funnels)

    print("✅ Done! Check the `data/` and `output/` folders.")

if __name__ == "__main__":
    main()
