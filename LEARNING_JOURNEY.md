# ReviewSense — Learning Journey 📖

> A personal account of what I learned at each stage of building ReviewSense — the decisions made, the failures hit, and the lessons that came out of each one.

---

## Stage 1 — Project Planning & Setup

### What I did
Created a GitHub repository with structured milestones, one per task. Used Claude to generate a project plan with daily checkpoints and a realistic delivery timeline.

### What I learned
- **Milestone-driven development works.** Breaking a multi-model ML project into daily checkpoints made it less overwhelming and easier to track progress.
- **Planning with AI is a skill.** Getting a useful project plan out of Claude required being specific about constraints — dataset size, deadline, hardware limits, no Colab until training exceeded 3–4 minutes locally.
- **Defining deliverables upfront keeps you honest.** Knowing from day one that the output was notebooks + blogposts + a Streamlit app + a PPT meant every technical decision was made with the end product in mind.

---

## Stage 2 — Exploratory Data Analysis

### What I did
Loaded the HuggingFace Amazon Beauty dataset and ran EDA: category distribution, rating distribution, text length analysis, null checks, and word clouds per sentiment class.

### What I learned
- **EDA is not optional — it predicts your problems.** The heavy positive skew in ratings (most reviews were 4–5 stars) was visible immediately in a simple countplot. This single finding foreshadowed every class imbalance issue that came later in model training.
- **Understanding your data before modelling saves time.** Going forward "anyways" with the skewed data was a conscious decision — but making it consciously, with the EDA evidence in hand, is very different from not knowing about it.
- **Category sparsity is a real problem.** The original dataset had hundreds of product categories, most with fewer than 10 reviews. This made it immediately clear that a clustering step was necessary — you can't train or analyse on 200 sparse groups.
- **Word clouds are for presentations, TF-IDF bar charts are for insights.** Word clouds look good in slides but top unigrams per class are more actionable for validating label mappings.

---

## Stage 3 — Sentiment Model v1 (95% accuracy, poor F1)

### What I did
Fine-tuned `distilbert-base-uncased` on the HuggingFace dataset using only the review text as input. Achieved 95% overall accuracy.

### What I learned
- **Accuracy is a misleading metric on imbalanced data.** A model that predicts "positive" for every review in a dataset that is 90% positive will achieve 90% accuracy while being completely useless. This was the most important practical lesson of the project.
- **Always look at per-class F1 scores.** The confusion matrix showed the model was almost entirely predicting positive — negative and neutral F1 scores were near zero. The 95% headline number was hiding a broken model.
- **Class imbalance is a data problem, not a model problem.** The instinct is to tune hyperparameters or add complexity. The right move is to fix the data first.
- **Three options exist for class imbalance — know all of them:**
  - Add class weights to the loss function (quick, stays with the same data)
  - Downsample the majority class (loses data but balances distribution)
  - Get more data for the minority classes (best quality, most effort)
  - Each has trade-offs — understanding when to use which is a real skill.

---

## Stage 4 — Sentiment Model v2 (78.6% accuracy, balanced F1)

### What I did
Sourced a larger dataset with ~170,000 reviews and downsampled all three classes to match the size of the smallest class (negative reviews). Retrained DistilBERT on the balanced dataset.

### What I learned
- **Downsampling + a larger dataset is a powerful combination.** Getting more data first, then downsampling, gives you a balanced dataset without artificially inflating minority classes with repeated examples.
- **Once data is balanced, class weights become unnecessary.** The model learned all three classes properly without any loss function modification — confirming the imbalance was the root cause, not the model architecture.
- **There is a real accuracy–balance trade-off.** Dropping from 95% to 78.6% felt like regression, but the model was now actually useful. A balanced 78.6% beats a misleading 95% every time.
- **Balanced F1 is the right metric for multi-class imbalanced problems.** Macro-averaged F1 weights all classes equally — it penalises you for ignoring minority classes regardless of overall accuracy.

---

## Stage 5 — Sentiment Model v3 — Final (83% accuracy)

### What I did
Added the review `title` as an additional input alongside the review `text`, concatenated both fields, and retrained.

### What I learned
- **Feature engineering still matters in deep learning.** The assumption that transformer models "figure it out" from raw text is only true if you give them the right raw text. Review titles are short, opinionated, and information-dense — they carry strong sentiment signal in very few words.
- **Small input changes can move the needle meaningfully.** Adding one column — `title` — moved accuracy from 78.6% to 83% with no other changes. This is a stronger result than most hyperparameter searches would produce.
- **Concatenation order matters for transformers.** Putting the title first, then a separator, then the body text gives the model a natural reading order that mirrors how a human would scan a review.
- **Know when to stop tuning.** 83% with balanced F1 across all three classes is a good result for a 3-class sentiment problem on noisy user-generated text. The marginal gain from further tuning would not be worth the time given the project deadline.

---

## Stage 6 — Clustering v1 (K-Means failure)

### What I did
Used `distilbert-base-nli-stsb-mean-tokens` to embed product names, used the elbow method to find k=4, ran K-Means, and tried TF-IDF to extract meaningful category names. Tried three input combinations: product name alone, product name + description, product name + description + category string.

### What I learned
- **Garbage in, garbage out applies to embeddings too.** When the top TF-IDF words were identical across all clusters ("skin", "cream", "face", "oil"), it meant the clusters were not semantically distinct — the embedding or the clustering wasn't separating the space meaningfully.
- **K-Means forces balance even when balance is wrong.** K-Means minimises within-cluster variance and will split a large natural group into multiple clusters to fill its k buckets. If your data has one dominant group (Beauty products are mostly skincare), K-Means will fracture it artificially.
- **The elbow method gives you k, not quality.** Finding k=4 from the elbow plot means 4 is the point of diminishing returns for inertia — it says nothing about whether those 4 clusters are semantically meaningful.
- **Trying multiple input combinations is the right approach.** Systematically testing product name alone, then adding description, then adding category string eliminates variables and shows which features carry the relevant signal.

---

## Stage 7 — Clustering v2 (BERTopic alone — also failed)

### What I did
Switched to BERTopic with various embedding combinations. BERTopic returned 9 classes with the majority of products assigned to class -1 (outliers).

### What I learned
- **BERTopic's -1 class is a signal, not a failure.** Class -1 means "did not fit into any topic with sufficient confidence." A large -1 class means either the data is genuinely noisy, the embedding space is not separating topics well, or the minimum topic size parameter is too large.
- **Default BERTopic settings are conservative.** The default `min_topic_size` is designed for document-level text (articles, papers). For short product names, it needs to be tuned down significantly.
- **Unsupervised clustering requires more iteration than supervised learning.** There is no loss curve to watch, no validation metric to optimise. You have to look at the outputs, reason about them, and adjust — which is a slower and more qualitative process.
- **Knowing what a tool does poorly is as valuable as knowing what it does well.** BERTopic's strength is identifying coherent topics from text corpora. Its weakness is handling very short strings with high inter-topic vocabulary overlap — exactly the challenge with beauty product names.

---

## Stage 8 — Clustering v3 — Final (BERTopic + K-Means + GPT-4o-mini)

### What I did
Used BERTopic with K-Means as the clustering backend and GPT-4o-mini as the representation model to generate meaningful category names. One category held 70% of products — alarming initially, but confirmed correct on manual inspection.

### What I learned
- **Combining models can solve what each fails to solve alone.** BERTopic provides semantic topic modelling. K-Means provides controllable cluster count. GPT-4o-mini provides human-readable naming. None of the three solves the problem alone — together they do.
- **Questioning a result is not the same as the result being wrong.** The 70% concentration in one category felt like a bug. Cross-checking the actual products in that cluster showed it was correct — the Beauty dataset is genuinely dominated by skincare. K-Means had been hiding this reality by forcing artificial balance.
- **Forced balance in clustering is a form of bias.** If the real world has an unequal distribution, a model that imposes equal distribution is introducing systematic error. Letting the data speak is harder to explain but more honest.
- **Using an LLM for labelling is a legitimate technique.** Asking GPT-4o-mini to name a cluster given its top representative products produces better labels than any TF-IDF approach, because it understands context and can generalise — "these products are all hair styling tools" rather than "top words: brush, hair, hold, spray."

---

## Stage 9 — Review Summarizer

### What I did
Designed a 3-step pipeline: curate reviews per product (8 per star rating, prioritising verified purchases and helpful votes), summarize each product individually with GPT-4o-mini, then generate a final "Top 10 Best" category article from all 10 product summaries.

### What I learned
- **Metadata is as valuable as the text itself.** `verified_purchase` and `helpful_vote` are signals that a review is genuine and useful. Using them to weight review selection improves the quality of what the LLM sees — better inputs produce better summaries.
- **Balancing across star ratings prevents prompt bias.** If you only send the most helpful reviews, they will be skewed toward 5-star reviews (helpful votes correlate with positivity). Deliberately selecting 8 reviews per star rating ensures the LLM sees the full sentiment spectrum and can accurately report pros and cons.
- **Breaking a complex generation task into smaller steps produces better output.** One prompt asking GPT to write a full article from 400 raw reviews produces generic, shallow output. Two prompts — first summarize each product, then write the article from summaries — produces structured, specific, reliable output.
- **Prompt structure determines output structure.** Asking for "Overall Sentiment, Top 3 Pros, Top 3 Cons, Best For" produces exactly that, consistently, across all products. Structured prompts make downstream processing and display predictable.
- **LLMs are tools, not magic.** The quality of the final articles was directly proportional to the quality of the reviews fed in. The LLM synthesised well — but it could not compensate for noisy or irrelevant input reviews.

---

## Stage 10 — Streamlit Deployment

### What I did
Built a multi-page Streamlit app replicating an Amazon-style product browsing experience — categories, product pages, paginated reviews with live sentiment badges, on-demand AI summaries, and buying guide articles.

### What I learned
- **st.session_state is the foundation of multi-page Streamlit apps.** Streamlit reruns the entire script on every interaction — session state is how you persist which page the user is on, which product they selected, and which summaries have already been generated.
- **Caching is essential for ML apps.** `@st.cache_resource` for the DistilBERT model (loaded once, reused) and `@st.cache_data` for the CSV (loaded once, shared) are the difference between a usable app and one that reloads the model on every button click.
- **Real-world data is messy in unexpected ways.** Amazon image URLs used a quote structure that broke when embedded inside Python f-strings inside HTML passed to `st.markdown()`. A bug that has nothing to do with ML and everything to do with knowing how strings are parsed.
- **Unique widget keys are not optional in Streamlit.** Any loop that generates buttons must use a unique key per button. Using the first 20 characters of a product name as a key fails when two products share a common prefix — an integer index is always safer.
- **Building for showcase changes what you prioritise.** A research notebook prioritises correctness. A showcase app prioritises clarity, speed of understanding, and visual appeal. Knowing which mode you are in shapes every design decision.

---

## Overall Takeaways

| Lesson | Where it came from |
|---|---|
| Fix the data before tuning the model | Sentiment v1 → v2 |
| Accuracy alone is not a useful metric for imbalanced classes | Sentiment v1 |
| Feature engineering still matters in transformer models | Sentiment v2 → v3 |
| Forced clustering balance introduces bias | K-Means failure |
| Combining models solves what each fails alone | BERTopic + K-Means + GPT |
| Structured prompts produce structured, reliable outputs | Summarizer design |
| Metadata quality improves LLM input quality | Review curation |
| Session state and caching are non-negotiable in Streamlit | Deployment |
| A result that looks wrong is worth investigating before changing | 70% cluster concentration |
| Planning with clear deliverables keeps every decision grounded | Project setup |
