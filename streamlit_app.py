import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random

# download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

#ABBREVIATION DICTIONARY-------------------------------------------------------
ABBREVIATION_DICT = {
    "auth": "authentication",
    "authn": "authentication",
    "authz": "authorization",
    "acl": "access control list",
    "api": "application programming interface",
    "db": "database",
    "dbms": "database management system",
    "ui": "user interface",
    "ux": "user experience",
    "os": "operating system",
    "sso": "single sign on",
    "mfa": "multi factor authentication",
    "2fa": "two factor authentication",
    "rbac": "role based access control",
    "pii": "personally identifiable information",
    "gdpr": "general data protection regulation",
    "tls": "transport layer security",
    "ssl": "secure sockets layer",
    "https": "hypertext transfer protocol secure",
    "ddos": "distributed denial of service",
    "dos": "denial of service",
    "xss": "cross site scripting",
    "csrf": "cross site request forgery",
    "sql": "structured query language",
    "jwt": "json web token",
    "vpn": "virtual private network",
    "ip": "internet protocol",
    "nfr": "non functional requirement",
    "fr": "functional requirement",
    "srs": "software requirements specification",
    "qos": "quality of service",
    "sla": "service level agreement",
    "rpo": "recovery point objective",
    "rto": "recovery time objective",
    "iam": "identity and access management",
    "ca": "certificate authority",
    "pki": "public key infrastructure",
    "aes": "advanced encryption standard",
    "rsa": "rivest shamir adleman",
    "audit log": "audit log",
    "e2e": "end to end",
    "crud": "create read update delete",
}

#data: NFR dataset------------------------------------------------------------------------
PROMISE_NFR_PATH = "data/nfr.txt"
 
@st.cache_data
def load_promise_nfr_dataset(filepath=PROMISE_NFR_PATH):
    """
    Loads the PROMISE NFR dataset, keeps only Functional (F) and Security (SE) labelled rows, and returns a list of (sentence, phase1_label, phase2_label) tuples matching the format used by evaluate_model(). phase2_label is always "N/A" since this source has no CIA Triad sub-labels.
    """
    label_map = {"F": "Functional", "SE": "Security"}
    dataset = []
 
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []
 
    for line in lines:
        match = re.match(r'^([A-Za-z]+):\s*(.*)', line)
        if not match:
            continue
        raw_label, text = match.group(1), match.group(2).strip()
        if raw_label in label_map and text:
            dataset.append((text, label_map[raw_label], "N/A"))
 
    return dataset
 
#Word-Net Synonym Augmentation (Class Imbalance Handling)--------------------------------
def get_synonym(word):
    """Get a synonym for a word using WordNet, preserving meaning."""
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    synonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    if synonyms:
        return random.choice(list(synonyms))
    return word
 
def synonym_augment(sentence, replace_ratio=0.3):
    """
    Generate an augmented version of a sentence by replacing a portion of words with WordNet synonyms, preserving original semantic meaning.
    """
    words = sentence.split()
    n_to_replace = max(1, int(len(words) * replace_ratio))
    indices = random.sample(range(len(words)), min(n_to_replace, len(words)))
 
    new_words = words.copy()
    for idx in indices:
        word = re.sub(r"[^\w]", "", words[idx])
        if word.lower() in STOP_WORDS or len(word) < 4:
            continue
        synonym = get_synonym(word)
        new_words[idx] = synonym
 
    return " ".join(new_words)
 
def augment_minority_class(dataset, minority_label="Security", label_index=1):
    """
    Doubles the minority class (Security requirements) using WordNet synonym replacement to address class imbalance, while preserving original semantic meaning.
    """
    augmented = list(dataset)
    minority_samples = [item for item in dataset if item[label_index] == minority_label]
 
    for sentence, p1_label, p2_label in minority_samples:
        aug_sentence = synonym_augment(sentence)
        augmented.append((aug_sentence, p1_label, p2_label))
 
    return augmented

#NLP Pre-processing pipeline-----------------------------------------------------
def expand_abbreviations(text):
    """Expand known software engineering abbreviations before tokenization."""
    words = text.split()
    expanded = []
    for word in words:
        clean = re.sub(r"[^\w]", "", word.lower())
        if clean in ABBREVIATION_DICT:
            expanded.append(ABBREVIATION_DICT[clean])
        else:
            expanded.append(word)
    return " ".join(expanded)
 
def clean_sentence(sentence):
    """Noise reduction: lowercase, remove special characters, remove stop words."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    tokens = [word for word in sentence.split() if word not in STOP_WORDS]
    return " ".join(tokens)
 
def preprocess_pipeline(combined_text):
    """Full pipeline: abbreviation expansion -> tokenization -> noise reduction."""
    expanded_text = expand_abbreviations(combined_text)
    sentences = sent_tokenize(expanded_text)
    cleaned_sentences = [clean_sentence(s) for s in sentences]
    # keep pairs aligned, drop empty
    pairs = [(orig, clean) for orig, clean in zip(sentences, cleaned_sentences) if clean]
    sentences = [p[0] for p in pairs]
    cleaned_sentences = [p[1] for p in pairs]
    return sentences, cleaned_sentences

#MODEL--------------------------------------------------------------------------------------------
# SECURITY-CENTRIC CLASSIFICATION ENGINE
# Phase I  → fine-tuned RoBERTa-base (naa18/srs-security-classifier-phase1)
# Phase II → zero-shot RoBERTa (no fine-tuned CIA model yet)

# Hugging Face repo for the fine-tuned Phase I model
PHASE1_MODEL_REPO = "naa18/srs-security-classifier-phase1"
 
# Label mapping must match what was used during training
PHASE1_ID2LABEL = {0: "Functional", 1: "Security"}

@st.cache_resource
def load_models():
    # Phase I: load fine-tuned model from Hugging Face Hub
    phase1_classifier = pipeline(
        "text-classification",
        model=PHASE1_MODEL_REPO,
        return_all_scores=True   # gives confidence for both classes
    )
    # Phase II: still zero-shot (no fine-tuned CIA model yet)
    phase2_classifier = pipeline(
        "zero-shot-classification",
        model="roberta-large-mnli"
    )
    return phase1_classifier, phase2_classifier
 
def phase1_classify(sentence, classifier):
    """
    Phase I: Binary classification using fine-tuned RoBERTa-base.
    Trained on PROMISE NFR dataset (F vs SE labels).
    Returns predicted label and confidence score for that label.
    """
    results = classifier(sentence)[0]  # list of {label, score} dicts
    # Find the winning label
    best = max(results, key=lambda x: x["score"])
    # Map model output label (LABEL_0/LABEL_1 or Functional/Security)
    raw_label = best["label"]
    if raw_label in PHASE1_ID2LABEL.values():
        label = raw_label
    else:
        # Handle LABEL_0 / LABEL_1 format
        label_id = int(raw_label.split("_")[-1])
        label = PHASE1_ID2LABEL[label_id]
    return label, best["score"]
 
def phase2_classify(sentence, classifier):
    """Phase II: Fine-grained CIA Triad classification (Security sentences only)."""
    candidate_labels = [
        "confidentiality - data access control and privacy",
        "integrity - data accuracy and prevention of unauthorized modification",
        "availability - system and data accessibility and uptime"
    ]
    result = classifier(sentence, candidate_labels)
    label_map = {
        "confidentiality - data access control and privacy": "Confidentiality",
        "integrity - data accuracy and prevention of unauthorized modification": "Integrity",
        "availability - system and data accessibility and uptime": "Availability"
    }
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    return label_map[top_label], top_score
 
 
def classify_requirements(sentences):
    phase1_classifier, phase2_classifier = load_models()
    results = []
    progress = st.progress(0)
    status = st.empty()
 
    for i, sentence in enumerate(sentences):
        status.write(f"Analysing sentence {i+1} of {len(sentences)}...")
        progress.progress((i + 1) / len(sentences))
 
        if len(sentence.split()) < 3:
            continue
 
        phase1_label, phase1_score = phase1_classify(sentence, phase1_classifier)
 
        if phase1_label == "Security":
            phase2_label, phase2_score = phase2_classify(sentence, phase2_classifier)
        else:
            phase2_label = "N/A"
            phase2_score = None
 
        results.append({
            "Sentence": sentence,
            "Phase I (Type)": phase1_label,
            "Phase I Confidence": phase1_score,
            "Phase II (CIA)": phase2_label,
            "Phase II Confidence": phase2_score
        })
 
    progress.empty()
    status.empty()
    return pd.DataFrame(results)

#Recommendations part--------------------------------------------------------------
RECOMMENDATION_RULES = [
    {
        "trigger_keywords": ["password", "login", "log in", "sign in", "credential"],
        "category": "Confidentiality - Authentication",
        "recommendation": "The system shall enforce strong password policies and multi-factor authentication for all user logins."
    },
    {
        "trigger_keywords": ["upload", "file", "document", "attachment"],
        "category": "Integrity - File Validation",
        "recommendation": "The system shall validate and scan uploaded files for malicious content before processing."
    },
    {
        "trigger_keywords": ["payment", "transaction", "checkout", "billing", "credit card"],
        "category": "Confidentiality - Data Encryption",
        "recommendation": "The system shall encrypt all payment and transaction data both in transit and at rest."
    },
    {
        "trigger_keywords": ["report", "export", "download data", "generate report"],
        "category": "Confidentiality - Access Control",
        "recommendation": "The system shall restrict report generation and data export to authorized roles only."
    },
    {
        "trigger_keywords": ["update", "edit", "modify", "delete", "change record"],
        "category": "Integrity - Change Tracking",
        "recommendation": "The system shall maintain an audit log of all create, update, and delete operations on critical records."
    },
    {
        "trigger_keywords": ["api", "integration", "third-party", "external service"],
        "category": "Confidentiality - API Security",
        "recommendation": "The system shall authenticate and authorize all API requests using secure tokens (e.g., JWT/OAuth)."
    },
    {
        "trigger_keywords": ["server", "uptime", "availability", "performance", "load"],
        "category": "Availability - Resilience",
        "recommendation": "The system shall implement redundancy and failover mechanisms to ensure continuous availability."
    },
    {
        "trigger_keywords": ["user data", "personal information", "profile", "customer data"],
        "category": "Confidentiality - Data Privacy",
        "recommendation": "The system shall comply with data privacy regulations (e.g., GDPR) when storing personal information."
    },
    {
        "trigger_keywords": ["search", "query", "filter", "input"],
        "category": "Integrity - Input Validation",
        "recommendation": "The system shall sanitize and validate all user input to prevent injection attacks."
    },
    {
        "trigger_keywords": ["notification", "email", "sms", "alert"],
        "category": "Confidentiality - Communication Security",
        "recommendation": "The system shall ensure notification channels do not leak sensitive information to unintended recipients."
    },
]
 
def recommend_security_requirements(functional_sentences):
    """
    Analyzes functional requirements to detect 'hidden' security needs
    using keyword matching combined with the classification engine.
    """
    recommendations = []
    for sentence in functional_sentences:
        sentence_lower = sentence.lower()
        matched_rules = []
        for rule in RECOMMENDATION_RULES:
            if any(kw in sentence_lower for kw in rule["trigger_keywords"]):
                matched_rules.append(rule)
 
        for rule in matched_rules:
            recommendations.append({
                "Functional Requirement": sentence,
                "Detected Security Gap": rule["category"],
                "Recommended Security Requirement": rule["recommendation"]
            })
 
    return pd.DataFrame(recommendations).drop_duplicates() if recommendations else pd.DataFrame(
        columns=["Functional Requirement", "Detected Security Gap", "Recommended Security Requirement"]
    )


#File reading (csv, txt files)-----------------------------------------------
def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
 
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_string(index=False)
 
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
 
    return ""

#Evaluation---------------------------------------------------------------------------
TARGET_F1 = 0.92
 
def evaluate_model(dataset):
    phase1_classifier, phase2_classifier = load_models()
 
    true_phase1, pred_phase1 = [], []
    true_phase2, pred_phase2 = [], []
 
    progress = st.progress(0)
    status = st.empty()
 
    for i, (sentence, true_p1, true_p2) in enumerate(dataset):
        status.write(f"Evaluating sentence {i+1} of {len(dataset)}...")
        progress.progress((i + 1) / len(dataset))
 
        pred_p1, _ = phase1_classify(sentence, phase1_classifier)
        true_phase1.append(true_p1)
        pred_phase1.append(pred_p1)
 
        if true_p1 == "Security":
            pred_p2, _ = phase2_classify(sentence, phase2_classifier)
            true_phase2.append(true_p2)
            pred_phase2.append(pred_p2)
 
    progress.empty()
    status.empty()
 
    # Phase I metrics
    p1_accuracy  = accuracy_score(true_phase1, pred_phase1)
    p1_precision = precision_score(true_phase1, pred_phase1, pos_label="Security", zero_division=0)
    p1_recall    = recall_score(true_phase1, pred_phase1, pos_label="Security", zero_division=0)
    p1_f1        = f1_score(true_phase1, pred_phase1, pos_label="Security", zero_division=0)
 
    st.write("### Phase I — Binary Classification Metrics")
    st.caption("Security vs Functional")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{p1_accuracy:.2%}")
    col2.metric("Precision", f"{p1_precision:.2%}")
    col3.metric("Recall",    f"{p1_recall:.2%}")
    col4.metric("F1-Score",  f"{p1_f1:.2%}", delta=f"{(p1_f1-TARGET_F1):+.2%} vs target")
 
    if p1_f1 >= TARGET_F1:
        st.success(f"Phase I F1-Score ({p1_f1:.2%}) meets target of {TARGET_F1:.0%}")
    else:
        st.warning(f"Phase I F1-Score ({p1_f1:.2%}) is below target of {TARGET_F1:.0%}")
 
    with st.expander("Phase I Confusion Matrix"):
        cm1 = confusion_matrix(true_phase1, pred_phase1, labels=["Security", "Functional"])
        cm1_df = pd.DataFrame(cm1,
            index=["Actual: Security", "Actual: Functional"],
            columns=["Predicted: Security", "Predicted: Functional"])
        st.dataframe(cm1_df)
 
        fig, ax = plt.subplots()
        ax.imshow(cm1, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Security", "Functional"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Security", "Functional"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm1[i, j], ha="center", va="center", color="black")
        ax.set_title("Phase I Confusion Matrix")
        st.pyplot(fig)
 
    # Phase II metrics
    p2_accuracy  = accuracy_score(true_phase2, pred_phase2) if true_phase2 else 0
    p2_precision = precision_score(true_phase2, pred_phase2, average="macro", zero_division=0) if true_phase2 else 0
    p2_recall    = recall_score(true_phase2, pred_phase2, average="macro", zero_division=0) if true_phase2 else 0
    p2_f1        = f1_score(true_phase2, pred_phase2, average="macro", zero_division=0) if true_phase2 else 0
 
    st.write("### Phase II — CIA Triad Classification Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{p2_accuracy:.2%}")
    col2.metric("Precision", f"{p2_precision:.2%}")
    col3.metric("Recall",    f"{p2_recall:.2%}")
    col4.metric("F1-Score",  f"{p2_f1:.2%}", delta=f"{(p2_f1-TARGET_F1):+.2%} vs target")
 
    if p2_f1 >= TARGET_F1:
        st.success(f"Phase II F1-Score ({p2_f1:.2%}) meets target of {TARGET_F1:.0%}")
    else:
        st.warning(f"Phase II F1-Score ({p2_f1:.2%}) is below target of {TARGET_F1:.0%}")
 
    if true_phase2:
        with st.expander("Phase II Confusion Matrix"):
            cia_labels = ["Confidentiality", "Integrity", "Availability"]
            cm2 = confusion_matrix(true_phase2, pred_phase2, labels=cia_labels)
            cm2_df = pd.DataFrame(cm2,
                index=[f"Actual: {l}" for l in cia_labels],
                columns=[f"Predicted: {l}" for l in cia_labels])
            st.dataframe(cm2_df)
 
            fig2, ax2 = plt.subplots()
            ax2.imshow(cm2, cmap="Greens")
            ax2.set_xticks(range(3)); ax2.set_xticklabels(cia_labels, rotation=30)
            ax2.set_yticks(range(3)); ax2.set_yticklabels(cia_labels)
            ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
            for i in range(3):
                for j in range(3):
                    ax2.text(j, i, cm2[i, j], ha="center", va="center", color="black")
            ax2.set_title("Phase II Confusion Matrix")
            st.pyplot(fig2)
 
    # Summary
    st.write("###Overall Metrics Summary")
    summary_df = pd.DataFrame({
        "Phase": ["Phase I (Binary)", "Phase II (CIA Triad)"],
        "Accuracy":  [f"{p1_accuracy:.2%}",  f"{p2_accuracy:.2%}"],
        "Precision": [f"{p1_precision:.2%}", f"{p2_precision:.2%}"],
        "Recall":    [f"{p1_recall:.2%}",    f"{p2_recall:.2%}"],
        "F1-Score":  [f"{p1_f1:.2%}",        f"{p2_f1:.2%}"],
        "Target F1": [f"{TARGET_F1:.0%}",    f"{TARGET_F1:.0%}"]
    })
    st.dataframe(summary_df, use_container_width=True)
 
    fig3, ax3 = plt.subplots()
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    phase1_vals = [p1_accuracy, p1_precision, p1_recall, p1_f1]
    phase2_vals = [p2_accuracy, p2_precision, p2_recall, p2_f1]
    x = np.arange(len(metrics_names))
    width = 0.35
    ax3.bar(x - width/2, phase1_vals, width, label="Phase I")
    ax3.bar(x + width/2, phase2_vals, width, label="Phase II")
    ax3.axhline(y=TARGET_F1, color="red", linestyle="--", label=f"Target ({TARGET_F1:.0%})")
    ax3.set_xticks(x); ax3.set_xticklabels(metrics_names)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Model Performance vs Target")
    ax3.legend()
    st.pyplot(fig3)
    

#UI part------------------------------------------------------------------------------------------
st.title('Software Requirement Specification Identifier for Security-Related Requirements')

st.write('The system is to identify security-related requirements in the SRS')

# Sidebar: model info
with st.sidebar:
    st.write("### ⚙️ Model Info")
    st.write("**Phase I (Binary)**")
    st.code(PHASE1_MODEL_REPO, language=None)
    st.caption("Fine-tuned RoBERTa-base · PROMISE NFR dataset · F1: 91.67%")
    st.write("**Phase II (CIA Triad)**")
    st.code("roberta-large-mnli", language=None)
    st.caption("Zero-shot classification")
    st.write("**PyTorch backend**")
    import torch
    st.caption(f"v{torch.__version__} · {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
tab1, tab2, tab3, tab4 = st.tabs([
    "Input & Preprocessing",
    "Classification & Recommendations",
    "Model Evaluation",
    "Data Augmentation Demo"
])

#Tab 1: TEXT/FILE UPLOAD SECTION---------------------------------------------------------------
with tab1:
    st.write("##Text Input")
    txt = st.text_area("Text to identify the requirements")
    st.write(f"{len(txt)} characters | {len(txt.split())} words")
 
    st.write("##SRS Document Upload")
    st.caption("Supported formats: .csv, .txt")
    uploaded_file = st.file_uploader(
        "Choose file(s)", accept_multiple_files=True,
        type=["csv", "txt"]
    )
 
    all_text = []
    if txt.strip():
        all_text.append(txt)
 
    if uploaded_file:
        for uf in uploaded_file:
            try:
                content = read_uploaded_file(uf)
                all_text.append(content)
                st.success(f"Loaded: {uf.name}")
            except Exception as e:
                st.error(f"Failed to load {uf.name}: {e}")
 
    if all_text:
        combined_text = "\n\n".join(all_text)
        st.session_state["combined_text"] = combined_text
 
        with st.expander("Raw Text Preview"):
            st.text_area("Raw", combined_text[:1000] + "...", height=150)
 
        sentences, cleaned_sentences = preprocess_pipeline(combined_text)
        st.session_state["sentences"] = sentences
        st.session_state["cleaned_sentences"] = cleaned_sentences
 
        with st.expander("Before vs After Comparison (incl. abbreviation expansion)"):
            comparison_df = pd.DataFrame({
                "Original/Expanded Sentence": sentences,
                "Cleaned Sentence": cleaned_sentences
            })
            st.dataframe(comparison_df, use_container_width=True)
 
        st.write(f"**{len(cleaned_sentences)} sentences ready for analysis**")
    else:
        st.warning("Please enter text or upload a file to proceed.")
        st.session_state["sentences"] = []

#Tab 2: Classification and Recommendations------------------------------------------------
with tab2:
    sentences = st.session_state.get("sentences", [])
 
    if not sentences:
        st.info("Add input in the **Input & Preprocessing** tab first.")
    else:
        st.write("##Two-Phase Classification")
        if st.button("Run Classification"):
            with st.spinner("Loading RoBERTa model (first run may take a minute)..."):
                results_df = classify_requirements(sentences)
            st.session_state["results_df"] = results_df
 
        if "results_df" in st.session_state:
            results_df = st.session_state["results_df"]
 
            total = len(results_df)
            security_count = len(results_df[results_df["Phase I (Type)"] == "Security"])
            functional_count = total - security_count
 
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Sentences", total)
            col2.metric("Security Requirements", security_count)
            col3.metric("Functional Requirements", functional_count)
 
            # Visual analytics
            st.write("###Visual Analytics")
            viz_col1, viz_col2 = st.columns(2)
 
            with viz_col1:
                fig1, ax1 = plt.subplots()
                ax1.bar(["Security", "Functional"], [security_count, functional_count],
                        color=["#d62728", "#1f77b4"])
                ax1.set_title("Phase I: Security vs Functional")
                ax1.set_ylabel("Count")
                st.pyplot(fig1)
 
            with viz_col2:
                cia_df = results_df[results_df["Phase II (CIA)"] != "N/A"]
                if not cia_df.empty:
                    cia_counts = cia_df["Phase II (CIA)"].value_counts()
                    fig2, ax2 = plt.subplots()
                    ax2.pie(cia_counts.values, labels=cia_counts.index, autopct="%1.1f%%",
                            colors=["#2ca02c", "#ff7f0e", "#9467bd"])
                    ax2.set_title("Phase II: CIA Triad Breakdown")
                    st.pyplot(fig2)
                else:
                    st.info("No security requirements found to break down by CIA category.")
 
            # Display table with formatted confidence
            display_df = results_df.copy()
            display_df["Phase I Confidence"] = display_df["Phase I Confidence"].apply(lambda x: f"{x:.0%}")
            display_df["Phase II Confidence"] = display_df["Phase II Confidence"].apply(
                lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
 
            st.write("###Full Classification Results")
            st.dataframe(display_df, use_container_width=True)
 
            csv = display_df.to_csv(index=False)
            st.download_button("Download Results as CSV", csv, "classification_results.csv", "text/csv")
 
            # ── Recommendation Logic ────────────────────────────────────────
            st.write("---")
            st.write("##Security Requirements Recommendations")
            st.caption("Detecting hidden security needs in functional requirements")
 
            functional_sentences = results_df[results_df["Phase I (Type)"] == "Functional"]["Sentence"].tolist()
            rec_df = recommend_security_requirements(functional_sentences)
 
            if not rec_df.empty:
                st.write(f"**{len(rec_df)} potential security gaps detected**")
                st.dataframe(rec_df, use_container_width=True)
 
                rec_csv = rec_df.to_csv(index=False)
                st.download_button("⬇️ Download Recommendations as CSV", rec_csv,
                                    "security_recommendations.csv", "text/csv")
            else:
                st.info("No obvious hidden security gaps detected in the functional requirements.")
 
            # ── Summary Report Export ───────────────────────────────────────
            st.write("---")
            st.write("##Summary Report")
            report_lines = [
                "AI-ASSISTED SECURITY REQUIREMENTS ANALYSIS REPORT",
                "=" * 50,
                f"Total Sentences Analysed: {total}",
                f"Security Requirements: {security_count}",
                f"Functional Requirements: {functional_count}",
                "",
                "CIA TRIAD BREAKDOWN:",
            ]
            if not cia_df.empty:
                for cat, count in cia_df["Phase II (CIA)"].value_counts().items():
                    report_lines.append(f"  - {cat}: {count}")
            report_lines.append("")
            report_lines.append(f"SECURITY RECOMMENDATIONS GENERATED: {len(rec_df)}")
            report_lines.append("=" * 50)
 
            report_text = "\n".join(report_lines)
            st.text_area("Report Preview", report_text, height=250)
            st.download_button("Download Summary Report (.txt)", report_text, "summary_report.txt", "text/plain")

#Tab 3: Evaluation -------------------------------------------------------------------
with tab3:
    st.write("## Model Evaluation Metrics")
 
    promise_dataset = load_promise_nfr_dataset()
 
    if not promise_dataset:
        st.error(f"Could not load PROMISE NFR dataset from `{PROMISE_NFR_PATH}`. "
                 "Make sure `data/nfr.txt` is placed alongside `streamlit_app.py`.")
    else:
        f_count = len([d for d in promise_dataset if d[1] == "Functional"])
        se_count = len([d for d in promise_dataset if d[1] == "Security"])
        st.info(f"Evaluating on {len(promise_dataset)} PROMISE NFR sentences "
                f"({f_count} Functional, {se_count} Security). "
                f"Target F1-Score: {TARGET_F1:.0%}")
 
        use_augmented = st.checkbox("Evaluate using augmented dataset (class-balanced)", value=False)
 
        sample_size = st.slider(
            "Number of sentences to evaluate (larger = slower, more accurate)",
            min_value=20, max_value=len(promise_dataset),
            value=min(50, len(promise_dataset)), step=10
        )
 
        if st.button("Run Evaluation"):
            eval_dataset = promise_dataset
            eval_dataset = random.sample(eval_dataset, sample_size)
            if use_augmented:
                eval_dataset = augment_minority_class(eval_dataset)
            with st.spinner("Running evaluation..."):
                evaluate_model(eval_dataset)
 

#Tab 4: Demo----------------------------------------------------------------------
with tab4:
    st.write("##WordNet Synonym Augmentation Demo")
    st.caption("Demonstrates how the minority class (Security requirements) is augmented to handle class imbalance.")
 
    security_only = [item for item in SAMPLE_DATASET if item[1] == "Security"]
    functional_only = [item for item in SAMPLE_DATASET if item[1] == "Functional"]
 
    col1, col2 = st.columns(2)
    col1.metric("Original Security Count", len(security_only))
    col2.metric("Original Functional Count", len(functional_only))
 
    if st.button("Generate Augmented Samples"):
        augmented_dataset = augment_minority_class(SAMPLE_DATASET)
        new_security_count = len([item for item in augmented_dataset if item[1] == "Security"])
 
        st.success(f"Security class size: {len(security_only)} → {new_security_count} (after augmentation)")
 
        aug_preview = []
        for original, augmented in zip(security_only, augmented_dataset[len(SAMPLE_DATASET):]):
            aug_preview.append({
                "Original Sentence": original[0],
                "Augmented Sentence": augmented[0],
                "Label": augmented[1]
            })
 
        st.write("### Preview: Original vs Augmented")
        st.dataframe(pd.DataFrame(aug_preview), use_container_width=True)

#dumpppppppppppppppppp
# st.info('Text / SRS File Upload')

# #user input text
# st.write('Text Input')
# txt=st.text_area(
#   "Text to identify the requirements"
# )
# st.write(f" {len(txt)} characters | {len(txt.split())} words")

# #user file upload
# st.write('File Upload')
# uploaded_file=st.file_uploader(
#   "Choose file(s)", accept_multiple_files=True,type=["csv","txt"]
# )

# #for text input
# all_text=[]
# if txt.strip():
#   all_text.append(txt)
  
# #preprocess the files
# if uploaded_file:
  
#   for uploaded_files in uploaded_file:
#     st.write(f"{uploaded_files.name}")
#     #if text
#     if uploaded_files.name.endswith(".csv"):
#         df=pd.read_csv(uploaded_files)
#         string_data=df.to_string(index=False)
  
#     #if csv file
#     elif uploaded_files.name.endswith(".txt"):
#       string_data=uploaded_files.read().decode("utf-8",errors="ignore")

#     all_text.append(string_data)
#     st.success(f"Loaded: {uploaded_files.name}")

# if all_text:
#     combined_text="\n\n".join(all_text)
#     #view text
#     with st.expander("Raw Text Preview"):
#         st.text_area("Raw", combined_text[:1000] + "...", height=150)
        
#     #sentence tokenization
#     sentences = sent_tokenize(combined_text)
      
#   #noise reduction
#     def clean_sentence(sentence):
#         sentence = sentence.lower()
#         sentence = re.sub(r"[^a-z\s]", "", sentence)
#         sentence = re.sub(r"\s+", " ", sentence).strip()
#         tokens = sentence.split()
#         tokens = [word for word in tokens if word not in STOP_WORDS]
#         return " ".join(tokens)

#     cleaned_sentences = [clean_sentence(s) for s in sentences]
#     cleaned_sentences = [s for s in cleaned_sentences if s]

#       #before and after sentences comparison
#     with st.expander("Before vs After Comparison"):
#         comparison_df = pd.DataFrame({
#             "Original Sentence": sentences[:len(cleaned_sentences)],
#             "Cleaned Sentence": cleaned_sentences
#         })
#         st.dataframe(comparison_df)
        
#     st.write(f"{len(cleaned_sentences)} sentences ready for analysis")

#     #classification SECTION---------------------------------------------------------------
#     st.info("Classification of Requirements")
    
#     if st.button("Run"):
#         with st.spinner("Loading (It may take a few minutes...)"):
#             results_df = classify_requirements(sentences)
            
# else:
#   st.warning("Please enter text or upload file(s) to proceed.")



  
