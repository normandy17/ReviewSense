import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import ast
import time
import math
from pathlib import Path
from datetime import datetime

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="ReviewSense",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --primary:   #1B4FD8;
    --primary-light: #EEF2FF;
    --accent:    #F59E0B;
    --positive:  #10B981;
    --neutral:   #6B7280;
    --negative:  #EF4444;
    --surface:   #F8FAFC;
    --border:    #E2E8F0;
    --text:      #0F172A;
    --text-muted:#64748B;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--text);
    padding-top: 1rem;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stButton button {
    # background: transparent;
    # border: 1px solid rgba(255,255,255,0.15);
    color: white !important;
    text-align: left;
    width: 100%;
    border-radius: 8px;
    margin-bottom: 4px;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.1);
    border-color: rgba(255,255,255,0.3);
}

/* Cards */
.rs-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    transition: box-shadow 0.2s, transform 0.2s;
    cursor: pointer;
}
.rs-card:hover {
    box-shadow: 0 8px 24px rgba(27,79,216,0.12);
    transform: translateY(-2px);
}

/* Badges */
.badge-positive { background:#D1FAE5; color:#065F46; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-neutral  { background:#F3F4F6; color:#374151; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-negative { background:#FEE2E2; color:#991B1B; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

/* Stars */
.stars { color: #F59E0B; font-size: 16px; letter-spacing: 1px; }

/* Section headers */
.rs-section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: white;
    margin-bottom: 0.2rem;
}
.rs-section-sub {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* Product image */
.product-img {
    width: 100%;
    height: 200px;
    object-fit: contain;
    background: var(--surface);
    border-radius: 10px;
    border: 1px solid var(--border);
}

/* Review card */
.review-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.review-meta {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 8px;
}

/* AI summary box */
.ai-summary {
    background: linear-gradient(135deg, #EEF2FF 0%, #F0FDF4 100%);
    border: 1px solid #C7D2FE;
    border-radius: 14px;
    padding: 20px 24px;
    color: black;
    margin-bottom: 20px;
}
.ai-summary h4 {
    color: var(--primary);
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}

/* Pipeline step */
.pipeline-step {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.pipeline-step .step-icon { font-size: 2rem; margin-bottom: 8px; }
.pipeline-step .step-title { font-weight: 600; font-size: 15px; }
.pipeline-step .step-desc { color: var(--text-muted); font-size: 13px; margin-top: 4px; }

/* Buying guide */
.article-body { 
    font-size: 15px; 
    line-height: 1.8; 
    color: #1e293b;
    max-width: 820px;
}
.article-body h1, .article-body h2 { font-family: 'DM Serif Display', serif; }

/* Pagination */
.page-indicator {
    background: var(--primary-light);
    color: var(--primary);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    display: inline-block;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    # ── Update this path to your CSV ──────────────────────────────────────────
    CSV_PATH = "./datasets/final_reviews_with_categories.csv"
    df = pd.read_csv(CSV_PATH, low_memory=False)
    return df

@st.cache_resource
def load_sentiment_model():
    from transformers import pipeline as hf_pipeline
    MODEL_PATH = "./Sentiment Analyzer 3.0"
    classifier = hf_pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=-1,      # CPU; set to 0 for GPU
        truncation=True,
        max_length=512,
    )
    return classifier

def label_to_sentiment(label):
    mapping = {"LABEL_2": "positive", "LABEL_1": "neutral", "LABEL_0": "negative"}
    return mapping.get(label, "neutral")

def sentiment_badge(sentiment):
    badges = {
        "positive": '<span class="badge-positive">🟢 Positive</span>',
        "neutral":  '<span class="badge-neutral">⚪ Neutral</span>',
        "negative": '<span class="badge-negative">🔴 Negative</span>',
    }
    return badges.get(sentiment, badges["neutral"])

def predict_sentiment(text, classifier):
    try:
        result = classifier(text[:512])[0]
        label = label_to_sentiment(result["label"])
        score = round(result["score"] * 100, 1)
        return label, score
    except Exception:
        return "neutral", 50.0

def get_image_url(images_raw):
    """Extract first large image URL from the images JSON field."""
    PLACEHOLDER = "https://via.placeholder.com/300x300?text=No+Image"
    if not images_raw:
        return PLACEHOLDER
    try:
        if isinstance(images_raw, str):
            images = json.loads(images_raw.replace("'", '"'))
        else:
            images = images_raw
        if isinstance(images, list) and images:
            url = images[0].get("large") or images[0].get("thumb")
            return url if url else PLACEHOLDER
    except Exception:
        pass
    return PLACEHOLDER

def render_stars(rating, max_stars=5):
    rating = float(rating) if rating else 0
    full  = int(rating)
    empty = max_stars - full
    return "★" * full + "☆" * empty

def format_count(n):
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(int(n))


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARIZER  (reuses your existing functions)
# ══════════════════════════════════════════════════════════════════════════════

import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client_oai = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# def get_balanced_priority_reviews(group, n_per_rating=8):
#     all_selected = []
#     for r in [1, 2, 3, 4, 5]:
#         mask   = group["rating"] == r
#         bucket = group[mask].sort_values(
#             by=["verified_purchase", "helpful_vote"], ascending=[False, False]
#         ).head(n_per_rating)
#         if not bucket.empty:
#             all_selected.append(bucket)
#     if not all_selected:
#         return ""
#     combined = pd.concat(all_selected)
#     return " [NEXT REVIEW] ".join(combined["text"].astype(str))

def get_balanced_priority_reviews(group, n_per_rating=8):
    all_selected = []
    for r in [1, 2, 3, 4, 5]:
        mask = group["rating"] == r
        bucket = group[mask]
        
        if not bucket.empty:
            # Sort by Verified first, then Helpful votes
            sorted_bucket = bucket.sort_values(
                by=["verified_purchase", "helpful_vote"], 
                ascending=[False, False]
            ).head(n_per_rating)
            
            # CRITICAL: Label the rating so the AI knows which is which!
            labeled_reviews = sorted_bucket.apply(
                lambda x: f"[{x['rating']} Stars]: {x['text']}", axis=1
            )
            all_selected.append(labeled_reviews)
            
    if not all_selected:
        return ""
        
    # Join everything together
    return " [NEXT REVIEW] ".join(pd.concat(all_selected))

def summarize_product(name, reviews_text):
    if not client_oai:
        return "⚠️ OpenAI API key not set. Add OPENAI_API_KEY to your environment."
    prompt = f"""Summarize the reviews for '{name}'.
The reviews provided are balanced across all rating levels (1-5 stars).
Please provide:
- Overall Sentiment
- Top 3 Pros (what loyal fans love)
- Top 3 Cons (common complaints from 1-2 star reviews)
- Best For: (who should buy this?)

REVIEWS:
{reviews_text}"""
    response = client_oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "page":             "home",
        "selected_category": None,
        "selected_product":  None,
        "review_page":       0,
        "product_summary_cache": {},   # product_name → summary text
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def nav(page, category=None, product=None):
    st.session_state.page = page
    if category is not None:
        st.session_state.selected_category = category
    if product is not None:
        st.session_state.selected_product = product
        st.session_state.review_page = 0


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 24px;'>
        <p style='font-family:DM Serif Display,serif;font-size:1.6rem;margin:0;color:white;'>ReviewSense</p>
        <p style='font-size:12px;opacity:0.5;margin:0;'>AI-powered review intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        ("🏠", "Home",                "home"),
        ("🔍", "Sentiment Analyzer",  "sentiment"),
        ("🗂️", "Categories",          "categories"),
        ("📰", "Buying Guide",        "buying_guide"),
    ]
    for icon, label, key in pages:
        active = "→ " if st.session_state.page == key else "   "
        if st.button(f"{icon}  {active}{label}", key=f"nav_{key}"):
            nav(key)
            st.rerun()

    # Breadcrumb
    if st.session_state.page == "category":
        st.markdown("---")
        st.markdown(f"<p style='font-size:12px;opacity:0.5;'>📂 {st.session_state.selected_category}</p>", unsafe_allow_html=True)
    if st.session_state.page == "product":
        st.markdown("---")
        if st.button("← Back to category"):
            nav("category")
            st.rerun()
        st.markdown(f"<p style='font-size:12px;opacity:0.5;'>🛍️ {str(st.session_state.selected_product)[:30]}...</p>", unsafe_allow_html=True)

    st.markdown("<div style='position:fixed;bottom:30px;left:20px;font-size:11px;opacity:0.3;'>Built with Streamlit · GPT-4o-mini · DistilBERT</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}. Check CSV_PATH in the app.")
    st.stop()

try:
    sentiment_clf = load_sentiment_model()
    MODEL_LOADED = True
except Exception as e:
    st.warning(f"Sentiment model not loaded: {e}")
    sentiment_clf = None
    MODEL_LOADED = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "home":
    # Hero
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1B4FD8 0%,#1e40af 60%,#1d4ed8 100%);
                border-radius:20px;padding:48px 40px;margin-bottom:32px;color:white;'>
        <p style='font-family:DM Serif Display,serif;font-size:3rem;margin:0;line-height:1.1;'>
            ReviewSense
        </p>
        <p style='font-size:1.15rem;opacity:0.85;margin:12px 0 24px;max-width:600px;'>
            From raw reviews to real intelligence. 
            NLP-powered product analysis, category discovery, and AI-generated buying guides.
        </p>
        <div style='display:flex;gap:12px;flex-wrap:wrap;'>
            <span style='background:rgba(255,255,255,0.15);padding:6px 16px;border-radius:20px;font-size:13px;'>📦 Amazon Beauty Dataset(It was the first in the list!!!)</span>
            <span style='background:rgba(255,255,255,0.15);padding:6px 16px;border-radius:20px;font-size:13px;'>🤖 DistilBERT · BERTopic · GPT-4o-mini</span>
            <span style='background:rgba(255,255,255,0.15);padding:6px 16px;border-radius:20px;font-size:13px;'>{:,} reviews analysed</span>
        </div>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("📝", "Total Reviews",    f"{len(df):,}",                          col1),
        ("🛍️", "Products",         f"{df['product_name'].nunique():,}",      col2),
        ("🗂️", "Categories",       f"{df['bertopic_category_name'].nunique():,}", col3),
        ("⭐", "Avg Rating",        f"{df['average_rating'].mean():.2f} / 5", col4),
    ]
    for icon, label, value, col in metrics:
        with col:
            st.markdown(f"""
            <div style='background:white;border:1px solid #E2E8F0;border-radius:14px;padding:20px;text-align:center;'>
                <div style='font-size:1.8rem;'>{icon}</div>
                <div style='font-size:1.5rem;font-weight:700;margin:4px 0;color: #0F172A;'>{value}</div>
                <div style='color:#64748B;font-size:13px;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline
    st.markdown('<p class="rs-section-title">How it works</p>', unsafe_allow_html=True)
    st.markdown('<p class="rs-section-sub">Three models, one pipeline</p>', unsafe_allow_html=True)

    steps = [
        ("📥", "Data Ingestion",      "Amazon Beauty reviews loaded, cleaned, and preprocessed"),
        ("🏷️", "Sentiment Analysis",  "DistilBERT classifies each review as positive, neutral, or negative"),
        ("🗂️", "Category Clustering", "BERTopic + GPT-4o-mini groups products into meaningful meta-categories"),
        ("✍️", "Article Generation",  "GPT-4o-mini writes buying guides using a map-reduce summarization pipeline"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class='pipeline-step'>
                <div class='step-icon'>{icon}</div>
                <div class='step-title' style='color: #0F172A;'>{title}</div>
                <div class='step-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick links
    st.markdown('<p class="rs-section-title">Explore</p>', unsafe_allow_html=True)
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("🔍  Try the Sentiment Analyzer", use_container_width=True):
            nav("sentiment"); st.rerun()
    with qc2:
        if st.button("🗂️  Browse Categories", use_container_width=True):
            nav("categories"); st.rerun()
    with qc3:
        if st.button("📰  Read Buying Guides", use_container_width=True):
            nav("buying_guide"); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "sentiment":
    st.markdown('<p class="rs-section-title">🔍 Sentiment Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="rs-section-sub">Paste any product review to classify it as positive, neutral, or negative using our fine-tuned DistilBERT model.</p>', unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("Sentiment model could not be loaded. Check the model path: `./Sentiment Analyzer 3.0`")
    else:
        review_input = st.text_area(
            "Review text",
            placeholder="e.g. This moisturizer is absolutely amazing — my skin has never felt better after just two weeks of use...",
            height=160,
        )

        col_btn, col_ex = st.columns([1, 3])
        with col_btn:
            analyze = st.button("Analyze →", type="primary", use_container_width=True)
        with col_ex:
            if st.button("Try an example", use_container_width=True):
                review_input = "I bought this serum expecting miracles but honestly it broke me out after just three days. Returned it immediately."
                analyze = True

        if analyze and review_input.strip():
            with st.spinner("Running DistilBERT inference..."):
                sentiment, confidence = predict_sentiment(review_input, sentiment_clf)

            # Result
            color_map = {"positive": "#10B981", "neutral": "#6B7280", "negative": "#EF4444"}
            icon_map  = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}
            color = color_map[sentiment]

            st.markdown(f"""
            <div style='background:white;border:2px solid {color};border-radius:16px;padding:28px;margin-top:16px;'>
                <div style='font-size:3rem;margin-bottom:8px;'>{icon_map[sentiment]}</div>
                <div style='font-size:2rem;font-weight:700;color:{color};text-transform:capitalize;'>{sentiment}</div>
                <div style='color:#64748B;margin-top:4px;font-size:15px;'>Confidence: <strong>{confidence}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(confidence / 100)

        elif analyze:
            st.warning("Please enter a review first.")

        # Dataset sentiment distribution
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Dataset sentiment distribution")
        if "sentiment" in df.columns:
            dist = df["sentiment"].value_counts()
        else:
            # Approximate from ratings
            def rating_to_sentiment(r):
                if r >= 4: return "positive"
                if r == 3: return "neutral"
                return "negative"
            dist = df["rating"].apply(rating_to_sentiment).value_counts()

        for sent, count in dist.items():
            pct = count / len(df) * 100
            color = {"positive":"#10B981","neutral":"#6B7280","negative":"#EF4444"}.get(sent,"#6B7280")
            st.markdown(f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'><span style='width:80px;font-size:13px;text-transform:capitalize;'>{sent}</span><div style='flex:1;background:#F1F5F9;border-radius:8px;height:20px;'><div style='background:{color};width:{pct:.0f}%;height:100%;border-radius:8px;'></div></div><span style='font-size:13px;color:#64748B;width:80px;text-align:right;'>{count:,} ({pct:.1f}%)</span></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "categories":
    st.markdown('<p class="rs-section-title">🗂️ Product Categories</p>', unsafe_allow_html=True)
    st.markdown('<p class="rs-section-sub">Discovered by BERTopic — click a category to explore its products.</p>', unsafe_allow_html=True)

    cats = df.groupby("bertopic_category_name").agg(
        n_products=("product_name", "nunique"),
        n_reviews=("text", "count"),
        avg_rating=("average_rating", "mean"),
    ).reset_index().sort_values("n_reviews", ascending=False)

    # Filter out noise topic
    cats = cats[cats["bertopic_category_name"] != "-1"].reset_index(drop=True)

    COLS = 3
    rows = math.ceil(len(cats) / COLS)

    for row_i in range(rows):
        cols = st.columns(COLS)
        for col_i, col in enumerate(cols):
            idx = row_i * COLS + col_i
            if idx >= len(cats):
                break
            cat_row = cats.iloc[idx]
            cat_name = cat_row["bertopic_category_name"]
            with col:
                if st.button(
                    f"**{cat_name}**\n\n"
                    f"🛍️ {int(cat_row['n_products'])} products  ·  "
                    f"📝 {format_count(int(cat_row['n_reviews']))} reviews  ·  "
                    f"⭐ {cat_row['avg_rating']:.1f}",
                    key=f"cat_{idx}",
                    use_container_width=True,
                ):
                    nav("category", category=cat_name)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CATEGORY DETAIL
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "category":
    # print("qwe123",df.iloc[0]["original_images"])
    cat = st.session_state.selected_category
    cat_df = df[df["bertopic_category_name"] == cat].copy()

    st.markdown(f'<p class="rs-section-title">📂 {cat}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="rs-section-sub">{cat_df["product_name"].nunique()} products · {len(cat_df):,} reviews</p>', unsafe_allow_html=True)

    # Product stats
    products = cat_df.groupby("product_name").agg(
        avg_rating=("average_rating", "first"),
        rating_number=("rating_number", "first"),
        n_reviews=("text", "count"),
        original_images=("original_images", "first"),
    ).reset_index().sort_values("avg_rating", ascending=False)

    credible_products = products[
    (products['avg_rating'] >= 4.5) &
    (products['rating_number'] >= 100) &
    (products['n_reviews'] >= 50)
]

    # ── TOP 10 ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(90deg,#FEF3C7,#FFF);border-left:4px solid #F59E0B;
                border-radius:0 12px 12px 0;padding:12px 18px;margin-bottom:20px;'>
        <span style='font-weight:600;color:#92400E;'>🏆 Top 10 Best Sellers</span>
        <span style='font-size:13px;color:#B45309;margin-left:8px;'>Ranked by average rating</span>
    </div>
    """, unsafe_allow_html=True)

    top10 = credible_products.sort_values("avg_rating", ascending=False).head(10)
    PROD_COLS = 5
    for row_i in range(math.ceil(len(top10) / PROD_COLS)):
        cols = st.columns(PROD_COLS)
        for col_i, col in enumerate(cols):
            idx = row_i * PROD_COLS + col_i
            if idx >= len(top10):
                break
            p = top10.iloc[idx]
            img_url = get_image_url(p["original_images"])
            with col:
                st.markdown(f"""
                <div style='background:white;border:1px solid #E2E8F0;border-radius:12px;
                            padding:14px;text-align:center;margin-bottom:8px;'>
                    <img src="{img_url}" style='width:100%;height:120px;object-fit:contain;border-radius:8px;background:#F8FAFC;'
                         onerror="this.src='https://via.placeholder.com/120x120?text=No+Image'"/>
                    <div style='font-size:12px;font-weight:600;margin-top:8px;color:#1e293b;
                                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
                         title='{p["product_name"]}'>
                        #{idx+1} {p["product_name"][:28]}{'...' if len(p["product_name"])>28 else ''}
                    </div>
                    <div style='color:#F59E0B;font-size:14px;margin-top:4px;'>
                        {"★" * int(float(p["avg_rating"]))}{"☆" * (5-int(float(p["avg_rating"])))}
                    </div>
                    <div style='font-size:11px;color:#94A3B8;'>{float(p["avg_rating"]):.1f} · {format_count(int(p["rating_number"] or 0))} ratings</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View →", key=f"top10_p_{idx}", use_container_width=True):
                    nav("product", product=p["product_name"])
                    st.rerun()

    # ── ALL PRODUCTS ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### All products ({len(products)})")

    # search = st.text_input("Search products", placeholder="Filter by name...")
    # if search:
    #     products = products[products["product_name"].str.contains(search, case=False, na=False)]

    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])

    with f1:
        search = st.text_input("Search products", placeholder="Filter by name...")
    with f2:
        min_rating = st.selectbox("Min rating", [0.0, 3.0, 3.5, 4.0, 4.5], 
                               format_func=lambda x: f"⭐ {x}+" if x > 0 else "Any rating")
    with f3:
        min_reviews = st.selectbox("Min reviews", [0, 10, 50, 100, 500],
                                format_func=lambda x: f"📝 {x}+" if x > 0 else "Any count")
    with f4:
        sort_by = st.selectbox("Sort by", 
                            ["avg_rating", "rating_number", "n_reviews"],
                            format_func=lambda x: {
                                "avg_rating":    "⭐ Avg Rating",
                                "rating_number": "🔢 Total Ratings",
                                "n_reviews":     "📝 Reviews in Dataset"
                            }[x])

    if search:
        products = products[products["product_name"].str.contains(search, case=False, na=False)]
    if min_rating > 0:
        products = products[products["avg_rating"] >= min_rating]
    if min_reviews > 0:
        products = products[products["n_reviews"] >= min_reviews]

    products = products.sort_values(sort_by, ascending=False)

    for row_idx, (_, p) in enumerate(products.iterrows()):
        img_url = get_image_url(p["original_images"])
        c1, c2, c3 = st.columns([1, 5, 1])
        with c1:
            st.markdown(f"""<img src="{img_url}" style="width:70px;height:70px;object-fit:contain;border-radius:8px;background:#F8FAFC;border:1px solid #E2E8F0;" onerror="this.src='https://via.placeholder.com/70?text=?'"/>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style='padding:4px 0;'>
                <div style='font-weight:600;font-size:15px;'>{p["product_name"]}</div>
                <div style='color:#F59E0B;font-size:14px;display:inline;'>{"★" * int(float(p["avg_rating"] or 0))}{"☆" * (5-int(float(p["avg_rating"] or 0)))}</div>
                <span style='color:#64748B;font-size:13px;margin-left:8px;'>{float(p["avg_rating"] or 0):.1f} · {format_count(int(p["rating_number"] or 0))} ratings · {int(p["n_reviews"])} reviews</span>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            if st.button("View", key=f"allp_{row_idx}", use_container_width=True):
                nav("product", product=p["product_name"])
                st.rerun()
        st.markdown("<hr style='margin:6px 0;border-color:#F1F5F9;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCT DETAIL
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "product":
    product_name = st.session_state.selected_product
    prod_df = df[df["product_name"] == product_name].copy()

    if prod_df.empty:
        st.error("Product not found.")
        st.stop()

    meta = prod_df.iloc[0]
    img_url = get_image_url(meta.get("original_images"))
    avg_rating    = float(meta.get("average_rating") or 0)
    rating_number = int(meta.get("rating_number") or 0)
    category      = meta.get("bertopic_category_name", "—")

    # ── Product header ────────────────────────────────────────────────────────
    col_img, col_info = st.columns([1, 3])
    with col_img:
        st.markdown(f"""<img src="{img_url}" style="width:100%;max-width:280px;height:280px;object-fit:contain;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;padding:12px;" onerror="this.src='https://via.placeholder.com/280?text=No+Image'"/>""", unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div style='padding: 8px 0;'>
            <span style='background:#EEF2FF;color:#3730A3;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;'>{category}</span>
            <h2 style='font-family:DM Serif Display,serif;font-size:1.8rem;margin:12px 0 8px;line-height:1.2;'>{product_name}</h2>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>
                <span style='color:#F59E0B;font-size:22px;'>{"★" * int(avg_rating)}{"☆" * (5-int(avg_rating))}</span>
                <span style='font-size:1.2rem;font-weight:700;'>{avg_rating:.1f}</span>
                <span style='color:#64748B;font-size:14px;'>{format_count(rating_number)} ratings · {len(prod_df):,} reviews in dataset</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Rating breakdown
        for stars in [5, 4, 3, 2, 1]:
            count = len(prod_df[prod_df["rating"] == stars])
            pct   = count / len(prod_df) * 100 if len(prod_df) > 0 else 0
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
                <span style='font-size:12px;width:40px;color:#64748B;'>{stars}★</span>
                <div style='flex:1;background:#F1F5F9;border-radius:6px;height:12px;'>
                    <div style='background:#F59E0B;width:{pct:.0f}%;height:100%;border-radius:6px;'></div>
                </div>
                <span style='font-size:12px;color:#64748B;width:35px;text-align:right;'>{pct:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI Summary ───────────────────────────────────────────────────────────
    st.markdown("#### 🤖 AI Review Summary")
    if product_name in st.session_state.product_summary_cache:
        summary = st.session_state.product_summary_cache[product_name]
        st.markdown(f'<div class="ai-summary"><h4>✨ GPT-4o-mini Summary</h4><div style="white-space:pre-wrap;font-size:14px;line-height:1.7;">{summary}</div></div>', unsafe_allow_html=True)
    else:
        if st.button("Generate AI Summary", type="primary"):
            with st.spinner("GPT-4o-mini is reading the reviews..."):
                reviews_text = get_balanced_priority_reviews(prod_df)
                summary = summarize_product(product_name, reviews_text)
                st.session_state.product_summary_cache[product_name] = summary
            st.rerun()

    # ── Paginated reviews ────────────────────────────────────────────────────
    st.markdown("#### 💬 Customer Reviews")

    REVIEWS_PER_PAGE = 10
    reviews = prod_df.sort_values("helpful_vote", ascending=False).reset_index(drop=True)
    total_pages = math.ceil(len(reviews) / REVIEWS_PER_PAGE)
    current_page = st.session_state.review_page
    start = current_page * REVIEWS_PER_PAGE
    end   = start + REVIEWS_PER_PAGE
    page_reviews = reviews.iloc[start:end]

    st.markdown(f'<div class="page-indicator">Showing {start+1}–{min(end,len(reviews))} of {len(reviews):,} reviews</div>', unsafe_allow_html=True)

    for _, rev in page_reviews.iterrows():
        # Live sentiment prediction
        rev_text = str(rev.get("text", ""))
        if MODEL_LOADED and rev_text.strip():
            sentiment, confidence = predict_sentiment(rev_text, sentiment_clf)
        else:
            sentiment, confidence = "neutral", 50.0

        badge   = sentiment_badge(sentiment)
        stars   = render_stars(rev.get("rating", 3))
        title   = str(rev.get("title", "")).strip() or "Review"
        helpful = int(rev.get("helpful_vote", 0) or 0)
        verified = rev.get("verified_purchase", False)
        ts      = str(rev.get("timestamp", ""))[:10] if rev.get("timestamp") else ""

        st.markdown(f"""
        <div class="review-card">
            <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;'>
                <div>
                    <span style='color:#F59E0B;font-size:15px;'>{stars}</span>
                    <strong style='margin-left:8px;font-size:15px;'>{title}</strong>
                </div>
                {badge}
            </div>
            <p style='margin:0;font-size:14px;line-height:1.7;color:#334155;'>{rev_text[:600]}{'...' if len(rev_text)>600 else ''}</p>
            <div class='review-meta'>
                {'✅ Verified Purchase · ' if verified else ''}{f'👍 {helpful} found helpful · ' if helpful else ''}{ts}
                <span style='margin-left:8px;font-size:11px;color:#94A3B8;'>Sentiment confidence: {confidence}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Pagination controls
    pc1, pc2, pc3 = st.columns([1, 2, 1])
    with pc1:
        if current_page > 0:
            if st.button("← Previous", use_container_width=True):
                st.session_state.review_page -= 1
                st.rerun()
    with pc2:
        st.markdown(f"<div style='text-align:center;color:#64748B;font-size:13px;padding-top:8px;'>Page {current_page+1} of {total_pages}</div>", unsafe_allow_html=True)
    with pc3:
        if current_page < total_pages - 1:
            if st.button("Next →", use_container_width=True):
                st.session_state.review_page += 1
                st.rerun()


# # ══════════════════════════════════════════════════════════════════════════════
# # PAGE: BUYING GUIDE
# # ══════════════════════════════════════════════════════════════════════════════

# elif st.session_state.page == "buying_guide":
#     st.markdown('<p class="rs-section-title">📰 Buying Guides</p>', unsafe_allow_html=True)
#     st.markdown('<p class="rs-section-sub">AI-generated product recommendation articles — one per category, written by GPT-4o-mini using a map-reduce summarization pipeline.</p>', unsafe_allow_html=True)

#     SUMMARIES_DIR = Path("./summaries")
#     article_files = list(SUMMARIES_DIR.glob("Article_*.txt")) if SUMMARIES_DIR.exists() else []

#     if not article_files:
#         st.warning(f"No article files found in `{SUMMARIES_DIR}`. Run the blogpost generation notebook first.")
#     else:
#         # Category selector
#         cat_names = [f.stem.replace("Article_", "").replace("_", " ") for f in article_files]
#         selected  = st.selectbox("Select a category", cat_names)
#         sel_file  = SUMMARIES_DIR / f"Article_{selected.replace(' ', '_')}.txt"

#         if sel_file.exists():
#             article_text = sel_file.read_text(encoding="utf-8")

#             # Meta bar
#             cat_df_guide = df[df["bertopic_category_name"].str.replace("_"," ") == selected]
#             n_prods    = cat_df_guide["product_name"].nunique() if not cat_df_guide.empty else "—"
#             n_revs     = len(cat_df_guide) if not cat_df_guide.empty else "—"

#             col_m1, col_m2, col_m3 = st.columns(3)
#             col_m1.metric("Products analysed", n_prods)
#             col_m2.metric("Reviews analysed",  f"{n_revs:,}" if isinstance(n_revs, int) else n_revs)
#             col_m3.metric("Generated by", "GPT-4o-mini")

#             st.markdown("<br>", unsafe_allow_html=True)

#             # Render article
#             st.markdown(f'<div class="article-body">', unsafe_allow_html=True)

#             # Convert basic markdown to readable format
#             for line in article_text.split("\n"):
#                 if line.startswith("# "):
#                     st.markdown(f"## {line[2:]}")
#                 elif line.startswith("## "):
#                     st.markdown(f"### {line[3:]}")
#                 elif line.startswith("### "):
#                     st.markdown(f"**{line[4:]}**")
#                 else:
#                     st.markdown(line)

#             st.markdown("</div>", unsafe_allow_html=True)

#             # Download
#             st.download_button(
#                 "⬇️ Download article (.txt)",
#                 data=article_text,
#                 file_name=sel_file.name,
#                 mime="text/plain",
#             )
#         else:
#             st.error(f"File not found: {sel_file}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BUYING GUIDE
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "buying_guide":
    st.markdown('<p class="rs-section-title">📰 Buying Guides</p>', unsafe_allow_html=True)
    st.markdown('<p class="rs-section-sub">AI-generated product recommendation articles — one per category, written by GPT-4o-mini using a map-reduce summarization pipeline.</p>', unsafe_allow_html=True)

    SUMMARIES_DIR = Path("./summaries")
    article_files = list(SUMMARIES_DIR.glob("Article_*.txt")) if SUMMARIES_DIR.exists() else []

    if not article_files:
        st.warning(f"No article files found in `{SUMMARIES_DIR}`. Run the blogpost generation notebook first.")
    else:
        # Category selector
        cat_names = [f.stem.replace("Article_", "").replace("_", " ") for f in article_files]
        selected  = st.selectbox("Select a category", cat_names)
        sel_file  = SUMMARIES_DIR / f"Article_{selected.replace(' ', '_')}.txt"

        if sel_file.exists():
            article_text = sel_file.read_text(encoding="utf-8")

            # Meta bar
            cat_df_guide = df[df["bertopic_category_name"].str.replace("_", " ") == selected]
            n_prods = cat_df_guide["product_name"].nunique() if not cat_df_guide.empty else "—"
            n_revs  = len(cat_df_guide) if not cat_df_guide.empty else "—"

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Products analysed", n_prods)
            col_m2.metric("Reviews analysed", f"{n_revs:,}" if isinstance(n_revs, int) else n_revs)
            col_m3.metric("Generated by", "GPT-4o-mini")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Parse product names from Summary file ─────────────────────────
            summary_file  = SUMMARIES_DIR / f"Summary_{selected.replace(' ', '_')}.txt"
            guide_products = []
            if summary_file.exists():
                summary_text = summary_file.read_text(encoding="utf-8")
                for block in summary_text.split("\n\n"):
                    first_line = block.strip().split("\n")[0]
                    if first_line.startswith("PRODUCT: "):
                        p_name = first_line.replace("PRODUCT: ", "").strip()
                        guide_products.append(p_name)

            # ── Product quick-jump buttons ────────────────────────────────────
            if guide_products:
                st.markdown("""
                <div style='background:linear-gradient(90deg,#EEF2FF,#FFF);border-left:4px solid #1B4FD8;
                            border-radius:0 12px 12px 0;padding:12px 18px;margin-bottom:16px;'>
                    <span style='font-weight:600;color:#1e3a8a;'>🛍️ Products in this guide</span>
                    <span style='font-size:13px;color:#3730A3;margin-left:8px;'>Click to view product details</span>
                </div>
                """, unsafe_allow_html=True)

                # Fetch image + rating for each product for richer cards
                PROD_COLS = 5
                for row_i in range(math.ceil(len(guide_products) / PROD_COLS)):
                    cols = st.columns(PROD_COLS)
                    for col_i, col in enumerate(cols):
                        idx = row_i * PROD_COLS + col_i
                        if idx >= len(guide_products):
                            break
                        p_name = guide_products[idx]
                        p_rows = df[df["product_name"] == p_name]
                        if not p_rows.empty:
                            p_meta    = p_rows.iloc[0]
                            img_url   = get_image_url(p_meta.get("original_images"))
                            avg_r     = float(p_meta.get("average_rating") or 0)
                            rat_count = int(p_meta.get("rating_number") or 0)
                        else:
                            img_url   = "https://via.placeholder.com/120x120?text=No+Image"
                            avg_r     = 0.0
                            rat_count = 0

                        with col:
                            st.markdown(f"""
                            <div style='background:white;border:1px solid #E2E8F0;border-radius:12px;
                                        padding:14px;text-align:center;margin-bottom:8px;'>
                                <img src="{img_url}" style='width:100%;height:100px;object-fit:contain;
                                     border-radius:8px;background:#F8FAFC;'
                                     onerror="this.src='https://via.placeholder.com/120x120?text=No+Image'"/>
                                <div style='font-size:11px;font-weight:600;margin-top:8px;color:#1e293b;
                                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
                                     title='{p_name}'>
                                    #{idx+1} {p_name[:24]}{'...' if len(p_name) > 24 else ''}
                                </div>
                                <div style='color:#F59E0B;font-size:13px;margin-top:4px;'>
                                    {"★" * int(avg_r)}{"☆" * (5 - int(avg_r))}
                                </div>
                                <div style='font-size:11px;color:#94A3B8;'>{avg_r:.1f} · {format_count(rat_count)} ratings</div>
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button("View →", key=f"guide_p_{idx}", use_container_width=True):
                                nav("product", product=p_name)
                                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Render article ────────────────────────────────────────────────
            st.markdown('<div class="article-body">', unsafe_allow_html=True)
            for line in article_text.split("\n"):
                if line.startswith("# "):
                    st.markdown(f"## {line[2:]}")
                elif line.startswith("## "):
                    st.markdown(f"### {line[3:]}")
                elif line.startswith("### "):
                    st.markdown(f"**{line[4:]}**")
                else:
                    st.markdown(line)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            
            # ── Download ──────────────────────────────────────────────────────
            st.download_button(
                "⬇️ Download article (.txt)",
                data=article_text,
                file_name=sel_file.name,
                mime="text/plain",
            )
        else:
            st.error(f"File not found: {sel_file}")