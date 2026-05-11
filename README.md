# ReviewSense 🔍
### From raw reviews to real intelligence

> An NLP-powered product review intelligence system that classifies sentiment, discovers product categories, and generates AI-written buying guides — deployed as an Amazon-style Streamlit web app.

**GitHub:** `<!-- ADD YOUR GITHUB URL HERE -->`  
**Live App:** `<!-- ADD YOUR HOSTED APP URL HERE -->`

---

## Overview

ReviewSense is a full end-to-end NLP project built on the [McAuley Amazon Beauty dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews). It tackles three tasks:

| Task | Description | Model |
|---|---|---|
| 1 — Sentiment Analysis | Classify reviews as Positive, Neutral, or Negative | DistilBERT fine-tuned |
| 2 — Category Clustering | Group sparse product categories into meaningful meta-categories | BERTopic + K-Means + GPT-4o-mini |
| 3 — Review Summarization | Generate "Top 10 Best" buying guide articles per category | GPT-4o-mini (3-step pipeline) |

All three models are integrated into a Streamlit web app with an Amazon-style product browsing experience.

---

## Project Structure

```
ReviewSense/
│
├── notebooks/
│   ├── 01_EDA.ipynb                    ← Exploratory data analysis
│   ├── 02_Sentiment_Classifier.ipynb   ← DistilBERT training & evaluation
│   ├── 03_Product_Clustering.ipynb     ← BERTopic clustering pipeline
│   └── 04_Summarizer.ipynb             ← GPT-4o-mini article generation
│
├── Sentiment Analyzer 3.0/             ← Saved HuggingFace sentiment model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
│
├── summaries/                          ← AI-generated article .txt files
│   ├── Article_CategoryName.txt
│   └── Summary_CategoryName.txt
│
├── data/
│   └── reviews_final.csv               ← Final joined dataset
│
├── app.py                              ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- An OpenAI API key (for the summarizer and Streamlit live summaries)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

On Windows:
```bash
set OPENAI_API_KEY=sk-...
```

---

## Dataset

The project uses the **McAuley Amazon Beauty Reviews** dataset.

1. Download from: [https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)
2. Select the **Beauty** category
3. Place the downloaded file(s) in the `data/` folder
4. Run `01_EDA.ipynb` to clean and export `reviews_final.csv`

---

## Running the Notebooks

Run the notebooks **in order** — each one produces outputs consumed by the next.

### 01 — EDA
```bash
jupyter notebook notebooks/01_EDA.ipynb
```
- Loads and inspects the raw dataset
- Plots category distribution, rating balance, text length analysis, word clouds
- Exports `data/reviews_final.csv`

### 02 — Sentiment Classifier
```bash
jupyter notebook notebooks/02_Sentiment_Classifier.ipynb
```
- Maps star ratings → Positive (4–5★) / Neutral (3★) / Negative (1–2★)
- Trains DistilBERT (`distilbert-base-uncased`) on balanced, downsampled data
- Input: review `title` + review `text` combined
- Saves the model to `./Sentiment Analyzer 3.0/`
- **Final accuracy: 83% — balanced F1 across all 3 classes**

### 03 — Product Clustering
```bash
jupyter notebook notebooks/03_Product_Clustering.ipynb
```
- Embeds product names using `distilbert-base-nli-stsb-mean-tokens`
- Runs BERTopic with K-Means to discover natural product groupings
- Uses GPT-4o-mini to assign meaningful category names to each cluster
- Adds `bertopic_category_name` column to the dataset

### 04 — Summarizer
```bash
jupyter notebook notebooks/04_Summarizer.ipynb
```
- For each meta-category, selects the top 10 products by rating and volume
- **Step 1 — Curate:** picks 8 reviews per star rating (1–5), prioritising verified purchases and helpful votes
- **Step 2 — Summarize:** one GPT-4o-mini call per product → Overall Sentiment, Top 3 Pros, Top 3 Cons, Best For
- **Step 3 — Write Article:** feeds all 10 summaries to GPT → generates full "Top 10 Best {category}" article with a Category Winner and Buying Guide
- Saves outputs to `summaries/Article_{category}.txt` and `summaries/Summary_{category}.txt`

---

## Running the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### App pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, pipeline diagram, dataset statistics |
| 🔍 Sentiment Analyzer | Paste any review text → live DistilBERT prediction with confidence score |
| 🗂️ Categories | Grid of all BERTopic meta-categories with product and review counts |
| 📦 Category Page | Top 10 products + full searchable product list with images |
| 🛍️ Product Page | Amazon-style view: images, star breakdown, paginated reviews (10 at a time) with sentiment badges, live GPT-4o-mini summary on demand |
| 📰 Buying Guide | AI-generated buying articles rendered per category with download option |

### Notes
- Product AI summaries are cached in session state — GPT is only called once per product per session
- Sentiment is run live on every review card using the saved DistilBERT model
- Product images load from Amazon CDN URLs with automatic placeholder fallback

---

## Model Training Details

### Task 1 — Sentiment Classifier

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Labels | 3 (Positive / Neutral / Negative) |
| Input | Review title + review text (concatenated) |
| Dataset size | ~170,000 reviews |
| Balancing strategy | Downsampled to match smallest class |
| Training epochs | 3 |
| **Overall accuracy** | **83%** |
| Negative F1 | 0.82 |
| Neutral F1 | 0.75 |
| Positive F1 | 0.92 |

**Key decision:** The first model reached 95% accuracy but had poor F1 on Negative and Neutral classes due to severe class imbalance. Switching to a larger balanced dataset and downsampling to the smallest class resolved this.

### Task 2 — Category Clustering

| Parameter | Value |
|---|---|
| Embedding model | `distilbert-base-nli-stsb-mean-tokens` |
| Clustering | BERTopic with K-Means |
| Category naming | GPT-4o-mini |
| Input | Product name |
| Output | 4–6 meta-categories |

**Key decision:** Standalone K-Means forced equal cluster sizes regardless of natural data distribution. BERTopic with K-Means respects semantic groupings — one category legitimately holding 70% of products was confirmed correct on inspection.

### Task 3 — Review Summarizer

| Parameter | Value |
|---|---|
| LLM | GPT-4o-mini |
| Reviews per product | 8 per star rating (1–5) = up to 40 reviews |
| Review priority | `verified_purchase` + `helpful_vote` |
| GPT calls per category | 10 (one per product) + 1 (final article) |
| Output | Per-product summary + full category article |

---

## Deployment

The app is deployed on **Streamlit Community Cloud**.

**Live URL:** `<!-- ADD YOUR HOSTED APP URL HERE -->`

To deploy your own instance:
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `app.py`
4. Add `OPENAI_API_KEY` under the Secrets panel
5. Deploy — a public URL is generated automatically

---

## Tech Stack

- **NLP models:** HuggingFace Transformers, BERTopic, sentence-transformers
- **LLM:** OpenAI GPT-4o-mini
- **Data:** pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **App:** Streamlit
- **Training:** PyTorch

---

## Acknowledgements

Dataset: [McAuley Lab — Amazon Reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)  
Base model: [DistilBERT — Hugging Face](https://huggingface.co/distilbert-base-uncased)  
Topic modelling: [BERTopic](https://maartengr.github.io/BERTopic/)
