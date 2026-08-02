import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from transformers import pipeline
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

st.set_page_config(
    page_title = "AI-Assisted Security Requirements Identifier",
    layout = "wide"
)

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
TARGET_F1 = 0.92
 
@st.cache_data
def load_promise_nfr_dataset(filepath=PROMISE_NFR_PATH):
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
 
#Word-Net Synonym Augmentation (Class Imbalance Handling)------------------------------
def get_synonym(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return word
    syns = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            c = lemma.name().replace("_", " ")
            if c.lower() != word.lower():
                syns.add(c)
    return random.choice(list(syns)) if syns else word
 
def synonym_augment(sentence, replace_ratio=0.3):
    words = sentence.split()
    n = max(1, int(len(words) * replace_ratio))
    idxs = random.sample(range(len(words)), min(n, len(words)))
    new = words.copy()
    for i in idxs:
        w = re.sub(r"[^\w]", "", words[i])
        if w.lower() in STOP_WORDS or len(w) < 4:
            continue
        new[i] = get_synonym(w)
    return " ".join(new)
 
def augment_minority_class(dataset, minority_label="Security", label_index=1):
    augmented = list(dataset)
    minority = [item for item in dataset if item[label_index] == minority_label]
    for sentence, p1, p2 in minority:
        augmented.append((synonym_augment(sentence), p1, p2))
    return augmented
 
#NLP Pre-processing pipeline-----------------------------------------------------
def expand_abbreviations(text):
    """Expands known SRS abbreviations before tokenization."""
    words = text.split()
    expanded = []
    for word in words:
        clean = re.sub(r"[^\w]", "", word.lower())
        expanded.append(ABBREVIATION_DICT.get(clean, word))
    return " ".join(expanded)
 
def split_requirements(text):
   
    lines = [l.strip() for l in text.splitlines() if l.strip()]
 
    # Single block of text pasted without newlines
    if len(lines) == 1 and len(text.split()) > 20:
        # Known abbreviations that should never be split on
        abbrevs = {"e.g", "i.e", "etc", "vs", "Dr", "Mr", "Ms", "No",
                   "Fig", "fig", "Vol", "vol", "dept", "approx"}
 
        # Find all candidate split positions manually (avoids regex lookbehind)
        candidates = []
        i = 0
        paren_depth = 0
        while i < len(text):
            ch = text[i]
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
            elif ch == "." and paren_depth == 0:
                # Check what comes after the period
                rest = text[i+1:]
                if rest and rest[0] == " " and len(rest) > 1 and rest[1].isupper():
                    # Check the word before the period is not an abbreviation
                    word_before = re.search(r"(\w+)$", text[:i])
                    if word_before and word_before.group(1) not in abbrevs:
                        candidates.append(i)
            i += 1
 
        if candidates:
            parts = []
            prev = 0
            for pos in candidates:
                parts.append(text[prev:pos].strip())
                prev = pos + 2  # skip ". "
            parts.append(text[prev:].strip())
            lines = [p for p in parts if p]
 
    return [r for r in lines if len(r.split()) >= 3]
 
def clean_for_display(sentence):
    
    sentence = sentence.strip()
    sentence = re.sub(r"\s+", " ", sentence)  # collapse whitespace only
    return sentence
 
def preprocess_pipeline(combined_text):
   
    expanded_text = expand_abbreviations(combined_text)
    sentences = split_requirements(expanded_text)
 
    # Clean only for display — model always gets the original
    cleaned_for_display = [clean_for_display(s) for s in sentences]
 
    # Filter out empty/very short sentences
    pairs = [
        (orig, disp)
        for orig, disp in zip(sentences, cleaned_for_display)
        if len(orig.split()) >= 3
    ]
 
    originals = [p[0] for p in pairs]
    display   = [p[1] for p in pairs]
    return originals, display
 
 
#MODEL--------------------------------------------------------------------------------------------
PHASE1_MODEL_REPO = "naa18/srs-security-classifier-phase1"
PHASE2_MODEL_REPO = "naa18/srs-cia-classifier-phase2"
 
PHASE1_ID2LABEL = {0: "Functional", 1: "Security"}
PHASE2_ID2LABEL = {0: "Confidentiality", 1: "Integrity", 2: "Availability"}
 
@st.cache_resource
def load_models():
    phase1 = pipeline(
        "text-classification",
        model=PHASE1_MODEL_REPO,
        return_all_scores=True
    )
    phase2 = pipeline(
        "text-classification",
        model=PHASE2_MODEL_REPO,
        return_all_scores=True
    )
    return phase1, phase2
 

#Recommendations part--------------------------------------------------------------
# RECOMMENDATION_RULES = [
#     {
#         "trigger_keywords": ["password", "login", "log in", "sign in", "credential"],
#         "category": "Confidentiality - Authentication",
#         "recommendation": "The system shall enforce strong password policies and multi-factor authentication for all user logins."
#     },
#     {
#         "trigger_keywords": ["upload", "file", "document", "attachment"],
#         "category": "Integrity - File Validation",
#         "recommendation": "The system shall validate and scan uploaded files for malicious content before processing."
#     },
#     {
#         "trigger_keywords": ["payment", "transaction", "checkout", "billing", "credit card"],
#         "category": "Confidentiality - Data Encryption",
#         "recommendation": "The system shall encrypt all payment and transaction data both in transit and at rest."
#     },
#     {
#         "trigger_keywords": ["report", "export", "download data", "generate report"],
#         "category": "Confidentiality - Access Control",
#         "recommendation": "The system shall restrict report generation and data export to authorized roles only."
#     },
#     {
#         "trigger_keywords": ["update", "edit", "modify", "delete", "change record"],
#         "category": "Integrity - Change Tracking",
#         "recommendation": "The system shall maintain an audit log of all create, update, and delete operations on critical records."
#     },
#     {
#         "trigger_keywords": ["api", "integration", "third-party", "external service"],
#         "category": "Confidentiality - API Security",
#         "recommendation": "The system shall authenticate and authorize all API requests using secure tokens (e.g., JWT/OAuth)."
#     },
#     {
#         "trigger_keywords": ["server", "uptime", "availability", "performance", "load"],
#         "category": "Availability - Resilience",
#         "recommendation": "The system shall implement redundancy and failover mechanisms to ensure continuous availability."
#     },
#     {
#         "trigger_keywords": ["user data", "personal information", "profile", "customer data"],
#         "category": "Confidentiality - Data Privacy",
#         "recommendation": "The system shall comply with data privacy regulations (e.g., GDPR) when storing personal information."
#     },
#     {
#         "trigger_keywords": ["search", "query", "filter", "input"],
#         "category": "Integrity - Input Validation",
#         "recommendation": "The system shall sanitize and validate all user input to prevent injection attacks."
#     },
#     {
#         "trigger_keywords": ["notification", "email", "sms", "alert"],
#         "category": "Confidentiality - Communication Security",
#         "recommendation": "The system shall ensure notification channels do not leak sensitive information to unintended recipients."
#     },
# ]
 
# def recommend_security_requirements(functional_sentences):
#     """
#     Analyzes functional requirements to detect 'hidden' security needs
#     using keyword matching combined with the classification engine.
#     """
#     recommendations = []
#     for sentence in functional_sentences:
#         sentence_lower = sentence.lower()
#         matched_rules = []
#         for rule in RECOMMENDATION_RULES:
#             if any(kw in sentence_lower for kw in rule["trigger_keywords"]):
#                 matched_rules.append(rule)
 
#         for rule in matched_rules:
#             recommendations.append({
#                 "Functional Requirement": sentence,
#                 "Detected Security Gap": rule["category"],
#                 "Recommended Security Requirement": rule["recommendation"]
#             })
 
#     return pd.DataFrame(recommendations).drop_duplicates() if recommendations else pd.DataFrame(
#         columns=["Functional Requirement", "Detected Security Gap", "Recommended Security Requirement"]
#     )

#Classification Functions----------------------------------------------------------------

def phase1_classify(sentence, classifier):
    try:
        raw = classifier(sentence)
        results = raw[0] if isinstance(raw[0], list) else raw
        best = max(results, key=lambda x: x["score"])
        raw_label = best["label"]
        if raw_label in PHASE1_ID2LABEL.values():
            label = raw_label
        else:
            label = PHASE1_ID2LABEL[int(raw_label.split("_")[-1])]
        return label, best["score"]
    except Exception:
        # Fallback: truncate to first 100 words and retry
        short = " ".join(sentence.split()[:100])
        try:
            raw = classifier(short)
            results = raw[0] if isinstance(raw[0], list) else raw
            best = max(results, key=lambda x: x["score"])
            raw_label = best["label"]
            if raw_label in PHASE1_ID2LABEL.values():
                label = raw_label
            else:
                label = PHASE1_ID2LABEL[int(raw_label.split("_")[-1])]
            return label, best["score"]
        except Exception:
            return "Functional", 0.0
 
def phase2_classify(sentence, classifier):
    try:
        raw = classifier(sentence)
        results = raw[0] if isinstance(raw[0], list) else raw
        best = max(results, key=lambda x: x["score"])
        raw_label = best["label"]
        if raw_label in PHASE2_ID2LABEL.values():
            label = raw_label
        else:
            label = PHASE2_ID2LABEL[int(raw_label.split("_")[-1])]
        return label, best["score"]
    except Exception:
        short = " ".join(sentence.split()[:100])
        try:
            raw = classifier(short)
            results = raw[0] if isinstance(raw[0], list) else raw
            best = max(results, key=lambda x: x["score"])
            raw_label = best["label"]
            if raw_label in PHASE2_ID2LABEL.values():
                label = raw_label
            else:
                label = PHASE2_ID2LABEL[int(raw_label.split("_")[-1])]
            return label, best["score"]
        except Exception:
            return "Confidentiality", 0.0
 
def classify_requirements(sentences):
    phase1_classifier, phase2_classifier = load_models()
    results = []
    progress = st.progress(0)
    status   = st.empty()
 
    for i, sentence in enumerate(sentences):
        status.write(f"Analysing sentence {i+1} of {len(sentences)}...")
        progress.progress((i + 1) / len(sentences))
 
        if len(sentence.split()) < 3:
            continue
 
        p1_label, p1_score = phase1_classify(sentence, phase1_classifier)
 
        if p1_label == "Security":
            p2_label, p2_score = phase2_classify(sentence, phase2_classifier)
        else:
            p2_label, p2_score = "N/A", None
 
        results.append({
            "Sentence":            sentence,
            "Phase I (Type)":      p1_label,
            "Phase I Confidence":  p1_score,
            "Phase II (CIA)":      p2_label,
            "Phase II Confidence": p2_score,
        })
 
    progress.empty()
    status.empty()
    return pd.DataFrame(results)
 
#Evaluation------------------------------------------------------------------------

def evaluate_model(dataset):
    phase1_classifier, phase2_classifier = load_models()
 
    true_p1, pred_p1 = [], []
    true_p2, pred_p2 = [], []
 
    progress = st.progress(0)
    status   = st.empty()
 
    for i, (sentence, true_l1, true_l2) in enumerate(dataset):
        status.write(f"Evaluating sentence {i+1} of {len(dataset)}...")
        progress.progress((i + 1) / len(dataset))
 
        p1, _ = phase1_classify(sentence, phase1_classifier)
        true_p1.append(true_l1)
        pred_p1.append(p1)
 
        if true_l1 == "Security" and true_l2 != "N/A":
            p2, _ = phase2_classify(sentence, phase2_classifier)
            true_p2.append(true_l2)
            pred_p2.append(p2)
 
    progress.empty()
    status.empty()
 
    # Phase I metrics 
    st.write("### Phase I — Binary Classification Metrics")
    st.caption("Security vs Functional")
 
    p1_acc  = accuracy_score(true_p1, pred_p1)
    p1_prec = precision_score(true_p1, pred_p1, pos_label="Security", zero_division=0)
    p1_rec  = recall_score(true_p1, pred_p1, pos_label="Security", zero_division=0)
    p1_f1   = f1_score(true_p1, pred_p1, pos_label="Security", zero_division=0)
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{p1_acc:.2%}")
    c2.metric("Precision", f"{p1_prec:.2%}")
    c3.metric("Recall",    f"{p1_rec:.2%}")
    c4.metric("F1-Score",  f"{p1_f1:.2%}", delta=f"{(p1_f1-TARGET_F1):+.2%} vs target")
 
    if p1_f1 >= TARGET_F1:
        st.success(f"Phase I F1 ({p1_f1:.2%}) meets target of {TARGET_F1:.0%}")
    else:
        st.warning(f"Phase I F1 ({p1_f1:.2%}) is below target of {TARGET_F1:.0%}")
 
    with st.expander("Phase I Confusion Matrix"):
        cm1 = confusion_matrix(true_p1, pred_p1, labels=["Security", "Functional"])
        cm1_df = pd.DataFrame(cm1,
            index=["Actual: Security", "Actual: Functional"],
            columns=["Predicted: Security", "Predicted: Functional"])
        st.dataframe(cm1_df)
        fig, ax = plt.subplots()
        ax.imshow(cm1, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_xticklabels(["Security","Functional"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["Security","Functional"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm1[i,j], ha="center", va="center")
        ax.set_title("Phase I Confusion Matrix")
        st.pyplot(fig)
 
    # Phase II metrics 
    st.write("### Phase II — CIA Triad Classification Metrics")
 
    if not true_p2:
        st.warning("No CIA ground-truth labels available — Phase II metrics cannot be computed.")
    else:
        p2_acc  = accuracy_score(true_p2, pred_p2)
        p2_prec = precision_score(true_p2, pred_p2, average="macro", zero_division=0)
        p2_rec  = recall_score(true_p2, pred_p2, average="macro", zero_division=0)
        p2_f1   = f1_score(true_p2, pred_p2, average="macro", zero_division=0)
 
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{p2_acc:.2%}")
        c2.metric("Precision", f"{p2_prec:.2%}")
        c3.metric("Recall",    f"{p2_rec:.2%}")
        c4.metric("F1-Score",  f"{p2_f1:.2%}", delta=f"{(p2_f1-TARGET_F1):+.2%} vs target")
 
        if p2_f1 >= TARGET_F1:
            st.success(f"Phase II F1 ({p2_f1:.2%}) meets target of {TARGET_F1:.0%}")
        else:
            st.warning(f"Phase II F1 ({p2_f1:.2%}) is below target of {TARGET_F1:.0%}")
 
        with st.expander("Phase II Confusion Matrix"):
            cia = ["Confidentiality","Integrity","Availability"]
            cm2 = confusion_matrix(true_p2, pred_p2, labels=cia)
            cm2_df = pd.DataFrame(cm2,
                index=[f"Actual: {l}" for l in cia],
                columns=[f"Predicted: {l}" for l in cia])
            st.dataframe(cm2_df)
            fig2, ax2 = plt.subplots()
            ax2.imshow(cm2, cmap="Greens")
            ax2.set_xticks(range(3)); ax2.set_xticklabels(cia, rotation=20)
            ax2.set_yticks(range(3)); ax2.set_yticklabels(cia)
            for i in range(3):
                for j in range(3):
                    ax2.text(j, i, cm2[i,j], ha="center", va="center")
            ax2.set_title("Phase II Confusion Matrix")
            st.pyplot(fig2)
 
    #Summary bar chart 
    st.write("### Overall Metrics Summary")
    summary_df = pd.DataFrame({
        "Phase":     ["Phase I (Binary)", "Phase II (CIA Triad)"],
        "Accuracy":  [f"{p1_acc:.2%}",  f"{p2_acc:.2%}"  if true_p2 else "N/A"],
        "Precision": [f"{p1_prec:.2%}", f"{p2_prec:.2%}" if true_p2 else "N/A"],
        "Recall":    [f"{p1_rec:.2%}",  f"{p2_rec:.2%}"  if true_p2 else "N/A"],
        "F1-Score":  [f"{p1_f1:.2%}",   f"{p2_f1:.2%}"  if true_p2 else "N/A"],
    })
    st.dataframe(summary_df, use_container_width=True)
 
    if true_p2:
        fig3, ax3 = plt.subplots()
        metrics = ["Accuracy","Precision","Recall","F1-Score"]
        v1 = [p1_acc, p1_prec, p1_rec, p1_f1]
        v2 = [p2_acc, p2_prec, p2_rec, p2_f1]
        x  = np.arange(len(metrics))
        ax3.bar(x - 0.175, v1, 0.35, label="Phase I")
        ax3.bar(x + 0.175, v2, 0.35, label="Phase II")
        ax3.axhline(y=TARGET_F1, color="red", linestyle="--", label=f"Target ({TARGET_F1:.0%})")
        ax3.set_xticks(x); ax3.set_xticklabels(metrics)
        ax3.set_ylim(0, 1.1); ax3.legend()
        ax3.set_title("Model Performance vs Target")
        st.pyplot(fig3)
 
#File reading (csv, txt files)-----------------------------------------------
def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
 
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_string(index=False)
 
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
 
    return ""

#UI part------------------------------------------------------------------------------------------
st.title('Software Requirement Specification Identifier for Security-Related Requirements')

st.write('The system is to identify security-related requirements in the SRS')

# Sidebar: model info
with st.sidebar:
    st.write("### Model Info")
    st.write("**Phase I — Binary Classification**")
    st.code(PHASE1_MODEL_REPO, language=None)
    st.caption("Fine-tuned RoBERTa-base · Functional vs Security")
    st.write("**Phase II — CIA Triad**")
    st.code(PHASE2_MODEL_REPO, language=None)
    st.caption("Fine-tuned RoBERTa-base | C/I/A")
    import torch
    st.write("**Backend**")
    st.caption(f"PyTorch {torch.__version__} · {'GPU' if torch.cuda.is_available() else 'CPU'}")
 
st.write("---")

#TEXT/FILE UPLOAD SECTION---------------------------------------------------------------

st.write("## Text Input")
 
txt = st.text_area(
    "Paste SRS requirements text here",
    height=150,
    placeholder="e.g. The system shall authenticate users before granting access..."
)
st.caption(f"{len(txt)} characters | {len(txt.split())} words")
 
uploaded_files = st.file_uploader(
    "Or upload SRS file(s)",
    accept_multiple_files=True,
    type=["csv", "txt"]
)
 
# Collect all input text
all_text = []
if txt.strip():
    all_text.append(txt)
if uploaded_files:
    for uf in uploaded_files:
        try:
            content = read_uploaded_file(uf)
            if content.strip():
                all_text.append(content)
                st.success(f"Loaded: {uf.name}")
        except Exception as e:
            st.error(f"Failed to load {uf.name}: {e}")

st.write("")
start = st.button("Start Identifying Requirements", type="primary", use_container_width=True)
 
if start:
    if not all_text:
        st.warning("Please enter text or upload a file first.")
    else:
        combined_text = "\n\n".join(all_text) 
        sentences, cleaned_sentences = preprocess_pipeline(combined_text)
 
        with st.spinner("Loading models and classifying..."):
            results_df = classify_requirements(sentences)

        #store all results in session_state
        st.session_state["results_df"]        = results_df
        st.session_state["sentences"]         = sentences
        st.session_state["cleaned_sentences"] = cleaned_sentences
        st.session_state["combined_text"]     = combined_text
        st.session_state["input_was_csv"]     = any(
            uf.name.lower().endswith(".csv") for uf in uploaded_files
        ) if uploaded_files else False

if "results_df" in st.session_state: 
    
    results_df      = st.session_state["results_df"]
    sentences       = st.session_state["sentences"]
    cleaned_sentences = st.session_state["cleaned_sentences"]
    combined_text   = st.session_state["combined_text"]
    input_was_csv   = st.session_state["input_was_csv"]
    
    # Preprocessing 
    st.write("---")
    st.write("## Preprocessing")
 
    with st.expander("View raw text"):
        st.text_area("Raw", combined_text[:1000] + ("..." if len(combined_text) > 1000 else ""), height=150)
 
    with st.expander("View before vs after cleaning"):
        st.dataframe(pd.DataFrame({
            "Original Sentence": sentences[:len(cleaned_sentences)],
            "Cleaned Sentence":  cleaned_sentences
        }), use_container_width=True)
 
    st.success(f"{len(sentences)} sentences extracted and preprocessed")
 
    # Classification Results 
    st.write("---")
    st.write("## Classification Results")
 
    total            = len(results_df)
    security_count   = len(results_df[results_df["Phase I (Type)"] == "Security"])
    functional_count = total - security_count
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sentences",         total)
    c2.metric("Security Requirements",   security_count)
    c3.metric("Functional Requirements", functional_count)
 
    # Charts 
    st.write("### Visual Analytics")
    col1, col2 = st.columns(2)
 
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.bar(["Security","Functional"], [security_count, functional_count],
                color=["#d62728","#1f77b4"])
        ax1.set_title("Phase I: Security vs Functional")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)
 
    with col2:
        cia_df = results_df[results_df["Phase II (CIA)"] != "N/A"]
        if not cia_df.empty:
            cia_counts = cia_df["Phase II (CIA)"].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(cia_counts.values, labels=cia_counts.index, autopct="%1.1f%%",
                    colors=["#2ca02c","#ff7f0e","#9467bd"])
            ax2.set_title("Phase II: CIA Triad Breakdown")
            st.pyplot(fig2)
        else:
            st.info("No security requirements found for CIA breakdown.")
 
    # Full results table 
    st.write("### Full Classification Table")
    display_df = results_df.copy()
    display_df["Phase I Confidence"]  = display_df["Phase I Confidence"].apply(lambda x: f"{x:.0%}")
    display_df["Phase II Confidence"] = display_df["Phase II Confidence"].apply(
        lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
 
    display_df["Type Label"] = display_df["Phase I (Type)"].map(
        {"Security": "SE", "Functional": "F"}
    )
    display_df["CIA Label"] = display_df["Phase II (CIA)"].map(
        {"Confidentiality": "C", "Integrity": "I", "Availability": "A"}
    ).fillna("-")
 
    display_df = display_df[[
        "Sentence",
        "Type Label", "Phase I (Type)", "Phase I Confidence",
        "CIA Label",  "Phase II (CIA)", "Phase II Confidence"
    ]]
    st.dataframe(display_df, use_container_width=True)
 
    # Format-aware download 
    st.write("#### Download Results")
 
    csv_out = display_df.to_csv(index=False)
    st.download_button(
        "Download as CSV",
        csv_out, "classification_results.csv", "text/csv"
    )
 
    if not input_was_csv:
        txt_lines = ["CLASSIFICATION RESULTS", "=" * 65,
                     f"{'#':<4} {'Label':<6} {'CIA':<5} Requirement",
                     "-" * 65]
        for idx, row in display_df.reset_index().iterrows():
            cia = row["CIA Label"] if row["CIA Label"] != "-" else "  -"
            txt_lines.append(
                f"{idx+1:<4} {row['Type Label']:<6} {cia:<5} {row['Sentence']}"
            )
        txt_out = "\n".join(txt_lines)
        st.download_button(
            "Download as TXT",
            txt_out, "classification_results.txt", "text/plain"
        )
 
    # Summary Report 
    st.write("---")
    st.write("## Summary Report")
 
    cia_breakdown = ""
    if not cia_df.empty:
        for cat, cnt in cia_df["Phase II (CIA)"].value_counts().items():
            cia_breakdown += f"  - {cat}: {cnt}\n"
    else:
        cia_breakdown = "  None detected\n"
 
    report = f"""AI-ASSISTED SECURITY REQUIREMENTS ANALYSIS REPORT
{"="*50}
Total Sentences Analysed : {total}
Security Requirements    : {security_count}
Functional Requirements  : {functional_count}
 
CIA TRIAD BREAKDOWN:
{cia_breakdown}
{"="*50}"""
 
    st.text_area("Report Preview", report, height=220)
    st.download_button("Download Report (.txt)", report, "summary_report.txt", "text/plain")

# AI Reommendation System------------------------------------------------------------

    st.write("## Security Requirements Recommendation System")
    st.caption("Analyses each security requirements for vagueness and suggests improvements based on CIA Triad best practices.")
    
    SECURITY_IMPROVEMENT_RULES = {

    # ========================================================
    # CONFIDENTIALITY
    # ========================================================

    "Confidentiality": [

        {
            "concept": "password storage",

            "strong_triggers": [
                "store password",
                "stored password",
                "password storage",
                "protect password",
                "secure password",
                "password database",
                "credential storage"
            ],

            "trigger_groups": [
                ["password", "store"],
                ["password", "database"],
                ["password", "protect"],
                ["credential", "store"]
            ],

            "exclude_if": [],

            "required_groups": {
                "password-protection method": [
                    "password hashing",
                    "password-hashing",
                    "hashed password",
                    "hash passwords",
                    "salted hash",
                    "salt",
                    "argon2",
                    "bcrypt",
                    "scrypt",
                    "pbkdf2"
                ],

                "plaintext-storage restriction": [
                    "not store in plaintext",
                    "not stored in plaintext",
                    "shall not store plaintext",
                    "never store plaintext",
                    "not use reversible encryption",
                    "not stored using reversible encryption"
                ]
            },

            "minimum_missing": 1,
            "priority": 10,

            "issue": (
                "does not fully specify how stored passwords are protected"
            ),

            "suggestion": (
                "The system shall store passwords using an "
                "organisation-approved salted password-hashing mechanism "
                "and shall not store passwords in plaintext or using "
                "reversible encryption."
            )
        },

        {
            "concept": "authentication",

            "strong_triggers": [
                "authenticate",
                "authentication",
                "login",
                "log in",
                "sign in",
                "verify identity",
                "identity verification",
                "credential verification"
            ],

            "trigger_groups": [
                ["grant access", "identity"],
                ["access system", "credential"],
                ["access account", "credential"],
                ["access application", "credential"]
            ],

            "exclude_if": [
                "display login page",
                "login page layout",
                "navigate to login",
                "login screen colour",
                "login screen color"
            ],

            "required_groups": {
                "authentication method": [
                    "password",
                    "passkey",
                    "mfa",
                    "multi-factor",
                    "multifactor",
                    "two-factor",
                    "2fa",
                    "biometric",
                    "token",
                    "one-time password",
                    "otp",
                    "digital certificate",
                    "single sign-on",
                    "single sign on",
                    "sso",
                    "oauth",
                    "saml",
                    "ldap"
                ],

                "protected resource or operation": [
                    "account",
                    "application",
                    "system",
                    "database",
                    "portal",
                    "service",
                    "administrative function",
                    "protected resource",
                    "restricted function",
                    "sensitive operation"
                ]
            },

            "minimum_missing": 1,
            "priority": 9,

            "issue": (
                "does not fully specify the authentication method or "
                "the protected resource"
            ),

            "suggestion": (
                "The system shall authenticate [specified user or actor] "
                "using [organisation-approved authentication method] "
                "before granting access to [specified protected resource "
                "or operation]."
            )
        },

        {
            "concept": "authorisation",

            "strong_triggers": [
                "authorise",
                "authorize",
                "authorisation",
                "authorization",
                "access control",
                "access rights",
                "access permission",
                "restricted access",
                "privileged access",
                "only authorised",
                "only authorized",
                "unauthorised access",
                "unauthorized access",
                "role-based access",
                "role based access",
                "rbac"
            ],

            "trigger_groups": [
                ["only", "access"],
                ["restrict", "access"],
                ["permission", "resource"],
                ["privilege", "function"],
                ["role", "access"]
            ],

            "exclude_if": [
                "wheelchair access",
                "physical access road",
                "accessibility requirement",
                "access the public page",
                "access the help page"
            ],

            "required_groups": {
                "authorised actor or role": [
                    "administrator",
                    "admin",
                    "manager",
                    "supervisor",
                    "operator",
                    "employee",
                    "customer",
                    "owner",
                    "authorised user",
                    "authorized user",
                    "assigned role",
                    "specified role",
                    "rbac",
                    "role-based",
                    "role based"
                ],

                "protected resource or operation": [
                    "record",
                    "account",
                    "file",
                    "database",
                    "function",
                    "operation",
                    "resource",
                    "report",
                    "configuration",
                    "personal information",
                    "customer data",
                    "patient data",
                    "administrative function"
                ],

                "access-control basis": [
                    "role-based",
                    "role based",
                    "rbac",
                    "permission",
                    "access policy",
                    "acl",
                    "least privilege",
                    "assigned role",
                    "authorisation policy",
                    "authorization policy"
                ]
            },

            "minimum_missing": 1,
            "priority": 9,

            "issue": (
                "does not fully identify the authorised role, protected "
                "resource, or access-control basis"
            ),

            "suggestion": (
                "The system shall enforce [approved access-control policy] "
                "and permit [specified operation] on [specified resource] "
                "only to users assigned the [authorised role or permission]."
            )
        },

        {
            "concept": "encryption",

            "strong_triggers": [
                "encrypt",
                "encrypted",
                "encryption",
                "cryptographic protection",
                "protect sensitive data",
                "protect confidential data",
                "secure personal data",
                "secure customer data",
                "secure patient data"
            ],

            "trigger_groups": [
                ["sensitive data", "protect"],
                ["personal data", "protect"],
                ["confidential information", "protect"],
                ["payment information", "protect"],
                ["customer information", "protect"]
            ],

            "exclude_if": [
                "password hashing",
                "hash password",
                "password hash",
                "display encrypted",
                "encryption icon"
            ],

            "required_groups": {
                "protected data": [
                    "personal data",
                    "personal information",
                    "customer data",
                    "patient data",
                    "payment data",
                    "financial data",
                    "credential",
                    "file",
                    "database",
                    "record",
                    "specified data",
                    "sensitive information"
                ],

                "protection state": [
                    "at rest",
                    "stored data",
                    "in storage",
                    "in transit",
                    "during transmission",
                    "transmitted data",
                    "end-to-end"
                ],

                "approved cryptographic basis": [
                    "approved encryption",
                    "approved cryptographic",
                    "organisation-approved",
                    "organization-approved",
                    "cryptographic policy",
                    "key management",
                    "encryption key",
                    "tls",
                    "https"
                ]
            },

            "minimum_missing": 1,
            "priority": 8,

            "issue": (
                "does not fully specify which data is protected, when "
                "encryption applies, or the approved cryptographic basis"
            ),

            "suggestion": (
                "The system shall encrypt [specified sensitive data] "
                "[at rest, in transit, or both] using an "
                "organisation-approved cryptographic mechanism and "
                "approved key-management procedures."
            )
        },

        {
            "concept": "data privacy",

            "strong_triggers": [
                "personal data",
                "personal information",
                "personally identifiable information",
                "pii",
                "privacy",
                "privacy policy",
                "patient information",
                "health information",
                "data consent",
                "data retention"
            ],

            "trigger_groups": [
                ["customer data", "privacy"],
                ["patient data", "privacy"],
                ["personal data", "collect"],
                ["personal information", "store"]
            ],

            "exclude_if": [
                "public information",
                "non-personal data",
                "anonymous public data"
            ],

            "required_groups": {
                "processing purpose": [
                    "specified purpose",
                    "defined purpose",
                    "stated purpose",
                    "purpose limitation",
                    "business purpose",
                    "lawful purpose"
                ],

                "access or disclosure restriction": [
                    "authorised",
                    "authorized",
                    "access restriction",
                    "disclosure restriction",
                    "only accessible",
                    "only disclosed",
                    "consent"
                ],

                "retention or deletion condition": [
                    "retention",
                    "retain",
                    "delete",
                    "deletion",
                    "purge",
                    "remove",
                    "no longer necessary",
                    "retention period"
                ]
            },

            "minimum_missing": 2,
            "priority": 7,

            "issue": (
                "does not fully define the purpose, access restrictions, "
                "or retention conditions for personal data"
            ),

            "suggestion": (
                "The system shall process [specified personal data] only "
                "for [defined purpose], restrict access or disclosure to "
                "[authorised parties], and retain the data for no longer "
                "than [approved retention period or condition]."
            )
        }
    ],

    # ========================================================
    # INTEGRITY
    # ========================================================

    "Integrity": [

        {
            "concept": "input validation",

            "strong_triggers": [
                "input validation",
                "validate input",
                "validate user input",
                "validate submitted data",
                "sanitize input",
                "sanitise input",
                "invalid input",
                "malicious input",
                "injection prevention"
            ],

            "trigger_groups": [
                ["user input", "validate"],
                ["submitted data", "validate"],
                ["uploaded data", "validate"],
                ["form field", "validate"],
                ["input", "reject"],
                ["input", "invalid"]
            ],

            "exclude_if": [
                "input device",
                "keyboard input",
                "audio input",
                "video input",
                "input screen layout"
            ],

            "required_groups": {
                "validation constraints": [
                    "data type",
                    "expected type",
                    "length",
                    "minimum length",
                    "maximum length",
                    "range",
                    "format",
                    "regular expression",
                    "regex",
                    "schema",
                    "allowed value",
                    "allowlist",
                    "whitelist"
                ],

                "invalid-input handling": [
                    "reject",
                    "rejected",
                    "validation error",
                    "error message",
                    "not accepted",
                    "prevent processing",
                    "do not process"
                ],

                "validation timing or location": [
                    "server-side",
                    "server side",
                    "before processing",
                    "before storage",
                    "before execution",
                    "on submission"
                ]
            },

            "minimum_missing": 1,
            "priority": 10,

            "issue": (
                "does not fully specify the validation constraints, "
                "validation timing, or handling of invalid input"
            ),

            "suggestion": (
                "The system shall validate [specified input] before "
                "[processing or storage] by enforcing its expected type, "
                "length, range, and format, and shall reject values that "
                "do not satisfy the defined validation rules."
            )
        },

        {
            "concept": "audit logging",

            "strong_triggers": [
                "audit log",
                "audit trail",
                "security log",
                "transaction log",
                "log security event",
                "record security event",
                "track changes",
                "change history",
                "activity history"
            ],

            "trigger_groups": [
                ["record", "modification"],
                ["log", "access"],
                ["log", "change"],
                ["track", "transaction"],
                ["record", "user action"],
                ["history", "action"]
            ],

            "exclude_if": [
                "application debug log",
                "developer debug log",
                "logarithm",
                "wood log"
            ],

            "required_groups": {
                "actor identity": [
                    "user id",
                    "user identity",
                    "username",
                    "account",
                    "actor",
                    "authenticated user",
                    "service identity",
                    "who performed"
                ],

                "event time": [
                    "timestamp",
                    "date and time",
                    "event time",
                    "when"
                ],

                "event and affected object": [
                    "action",
                    "operation",
                    "event type",
                    "affected record",
                    "affected resource",
                    "before value",
                    "after value",
                    "what was changed"
                ],

                "log protection": [
                    "immutable",
                    "tamper-resistant",
                    "tamper resistant",
                    "read-only",
                    "protected from modification",
                    "protected from deletion",
                    "access controlled"
                ],

                "retention condition": [
                    "retain",
                    "retention",
                    "retained for",
                    "days",
                    "months",
                    "years",
                    "approved duration"
                ]
            },

            "minimum_missing": 2,
            "priority": 9,

            "issue": (
                "does not fully define the audit event details, protection, "
                "or retention conditions"
            ),

            "suggestion": (
                "The system shall record the authenticated actor, timestamp, "
                "action, affected resource, and outcome for [specified "
                "security-relevant event], protect the audit records against "
                "unauthorised modification or deletion, and retain them for "
                "[approved duration]."
            )
        },

        {
            "concept": "data integrity checking",

            "strong_triggers": [
                "data integrity",
                "integrity check",
                "verify integrity",
                "detect tampering",
                "tamper detection",
                "detect corruption",
                "corruption detection",
                "message authentication code",
                "digital signature"
            ],

            "trigger_groups": [
                ["verify", "data"],
                ["verify", "file"],
                ["checksum", "verify"],
                ["hash", "verify"],
                ["tamper", "data"],
                ["integrity", "record"]
            ],

            "exclude_if": [
                "referential integrity",
                "visual integrity",
                "structural integrity"
            ],

            "required_groups": {
                "protected object": [
                    "file",
                    "message",
                    "record",
                    "transaction",
                    "database",
                    "document",
                    "software package",
                    "specified data"
                ],

                "integrity-control mechanism": [
                    "hmac",
                    "message authentication code",
                    "digital signature",
                    "signed",
                    "authenticated hash",
                    "sha-256",
                    "sha-384",
                    "sha-512",
                    "organisation-approved integrity",
                    "organization-approved integrity"
                ],

                "verification timing": [
                    "before processing",
                    "before installation",
                    "before storage",
                    "before use",
                    "on receipt",
                    "during transmission",
                    "after retrieval"
                ],

                "failure response": [
                    "reject",
                    "quarantine",
                    "alert",
                    "notify",
                    "stop processing",
                    "integrity failure"
                ]
            },

            "minimum_missing": 2,
            "priority": 8,

            "issue": (
                "does not fully specify the protected object, integrity "
                "mechanism, verification timing, or failure response"
            ),

            "suggestion": (
                "The system shall verify the integrity and authenticity of "
                "[specified data or object] using an organisation-approved "
                "integrity-control mechanism before [specified processing "
                "stage], and shall [reject, quarantine, or alert] when "
                "verification fails."
            )
        }
    ],

    # ========================================================
    # AVAILABILITY
    # ========================================================

    "Availability": [

        {
            "concept": "backup and recovery",

            "strong_triggers": [
                "backup",
                "data backup",
                "system backup",
                "restore data",
                "data restoration",
                "disaster recovery",
                "recovery point objective",
                "recovery time objective",
                "rpo",
                "rto"
            ],

            "trigger_groups": [
                ["recover", "data"],
                ["restore", "database"],
                ["restore", "system"],
                ["data loss", "recover"],
                ["snapshot", "restore"]
            ],

            "exclude_if": [
                "backup button colour",
                "backup button color",
                "backup icon",
                "backup copy displayed"
            ],

            "required_groups": {
                "protected data or service": [
                    "database",
                    "customer data",
                    "transaction data",
                    "configuration",
                    "system state",
                    "critical data",
                    "critical service",
                    "specified data"
                ],

                "backup schedule": [
                    "hourly",
                    "daily",
                    "weekly",
                    "monthly",
                    "frequency",
                    "every",
                    "scheduled backup",
                    "continuous backup"
                ],

                "recovery objective": [
                    "rto",
                    "recovery time objective",
                    "restore within",
                    "rpo",
                    "recovery point objective",
                    "maximum data loss"
                ],

                "restoration verification": [
                    "restore test",
                    "restoration test",
                    "recovery test",
                    "tested backup",
                    "verify backup",
                    "backup verification"
                ]
            },

            "minimum_missing": 2,
            "priority": 10,

            "issue": (
                "does not fully define what is backed up, the backup "
                "schedule, recovery objectives, or restoration testing"
            ),

            "suggestion": (
                "The system shall back up [specified critical data or "
                "configuration] every [approved frequency], support recovery "
                "within an RTO of [value] and an RPO of [value], and verify "
                "restoration through [approved testing schedule]."
            )
        },

        {
            "concept": "uptime / availability target",

            "strong_triggers": [
                "system availability",
                "service availability",
                "system uptime",
                "service uptime",
                "maximum downtime",
                "available 24/7",
                "available 24x7",
                "continuously available",
                "high availability",
                "service interruption"
            ],

            "trigger_groups": [
                ["available", "business hours"],
                ["available", "all times"],
                ["available", "year"],
                ["available", "day"],
                ["downtime", "system"],
                ["uptime", "system"]
            ],

            "exclude_if": [
                "available product",
                "available option",
                "available room",
                "available record",
                "available item",
                "available appointment",
                "available user"
            ],

            "required_groups": {
                "measurable target": [
                    "%",
                    "percent",
                    "uptime target",
                    "availability target",
                    "maximum downtime",
                    "service level"
                ],

                "measurement period": [
                    "per month",
                    "monthly",
                    "per year",
                    "annually",
                    "per quarter",
                    "quarterly",
                    "business hours",
                    "calendar month",
                    "measurement period"
                ],

                "scope or exclusions": [
                    "excluding scheduled maintenance",
                    "scheduled maintenance",
                    "planned maintenance",
                    "specified service",
                    "critical service",
                    "service level agreement",
                    "sla"
                ]
            },

            "minimum_missing": 1,
            "priority": 9,

            "issue": (
                "does not fully define a measurable availability target, "
                "measurement period, or applicable service scope"
            ),

            "suggestion": (
                "The system shall maintain an availability of "
                "[stakeholder-approved percentage] for [specified service] "
                "during [defined measurement period], excluding only "
                "[approved maintenance conditions]."
            )
        },

        {
            "concept": "failover and redundancy",

            "strong_triggers": [
                "failover",
                "automatic failover",
                "standby server",
                "backup server",
                "redundant server",
                "redundancy",
                "replica server",
                "secondary server",
                "cluster failover"
            ],

            "trigger_groups": [
                ["server failure", "switch"],
                ["primary server", "secondary"],
                ["outage", "standby"],
                ["failure", "replica"]
            ],

            "exclude_if": [
                "backup file",
                "backup data",
                "duplicate record"
            ],

            "required_groups": {
                "failure condition": [
                    "server failure",
                    "service failure",
                    "node failure",
                    "network failure",
                    "primary failure",
                    "health-check failure",
                    "specified failure"
                ],

                "failover destination": [
                    "standby server",
                    "secondary server",
                    "replica",
                    "backup site",
                    "alternate site",
                    "redundant node"
                ],

                "switchover objective": [
                    "within",
                    "seconds",
                    "minutes",
                    "rto",
                    "recovery time",
                    "switchover time"
                ],

                "failover behaviour": [
                    "automatic",
                    "automatically",
                    "manual failover",
                    "traffic redirected",
                    "switch over",
                    "switchover"
                ]
            },

            "minimum_missing": 1,
            "priority": 9,

            "issue": (
                "does not fully define the failure condition, failover "
                "destination, switchover behaviour, or recovery objective"
            ),

            "suggestion": (
                "The system shall [automatically or manually] fail over "
                "from [primary component] to [approved standby component] "
                "when [defined failure condition] occurs, completing the "
                "switchover within [approved recovery objective]."
            )
        },

        {
            "concept": "denial of service protection",

            "strong_triggers": [
                "denial of service",
                "distributed denial of service",
                "ddos",
                "dos attack",
                "traffic flooding",
                "request flooding",
                "traffic overload",
                "resource exhaustion"
            ],

            "trigger_groups": [
                ["malicious traffic", "block"],
                ["excessive requests", "limit"],
                ["request rate", "limit"],
                ["traffic", "mitigate"],
                ["traffic", "throttle"]
            ],

            "exclude_if": [
                "normal network traffic report",
                "traffic statistics display",
                "road traffic"
            ],

            "required_groups": {
                "detection condition": [
                    "threshold",
                    "requests per",
                    "traffic rate",
                    "abnormal traffic",
                    "excessive requests",
                    "resource threshold",
                    "detection rule"
                ],

                "mitigation action": [
                    "rate limit",
                    "rate-limit",
                    "throttle",
                    "traffic filtering",
                    "block",
                    "drop request",
                    "waf",
                    "firewall",
                    "mitigate"
                ],

                "protected service": [
                    "api",
                    "website",
                    "service",
                    "application",
                    "network",
                    "endpoint",
                    "specified service"
                ],

                "notification or monitoring": [
                    "alert",
                    "notify",
                    "monitor",
                    "security team",
                    "administrator",
                    "log event"
                ]
            },

            "minimum_missing": 2,
            "priority": 8,

            "issue": (
                "does not fully specify the detection condition, mitigation "
                "action, protected service, or security notification"
            ),

            "suggestion": (
                "The system shall detect abnormal request or traffic rates "
                "for [specified service], apply [approved rate-limiting or "
                "traffic-filtering control] when [defined threshold] is "
                "exceeded, and notify [responsible security role]."
            )
        },

        {
            "concept": "maintenance and patching",

            "strong_triggers": [
                "security patch",
                "software patch",
                "patch management",
                "vulnerability patch",
                "security update",
                "vulnerability update",
                "antivirus update",
                "malware definition update"
            ],

            "trigger_groups": [
                ["patch", "vulnerability"],
                ["update", "security vulnerability"],
                ["update", "malware definition"]
            ],

            "exclude_if": [
                "update profile",
                "update address",
                "update booking",
                "update customer",
                "update record",
                "update report",
                "update product",
                "update account information"
            ],

            "required_groups": {
                "deployment timeframe": [
                    "within",
                    "hours",
                    "days",
                    "severity",
                    "critical patch",
                    "high-severity",
                    "approved timeframe",
                    "patch schedule"
                ],

                "pre-deployment testing": [
                    "test",
                    "tested",
                    "staging",
                    "pre-production",
                    "validation environment"
                ],

                "controlled deployment": [
                    "maintenance window",
                    "approved deployment",
                    "change management",
                    "rollback",
                    "backout plan",
                    "deployment approval"
                ]
            },

            "minimum_missing": 1,
            "priority": 7,

            "issue": (
                "does not fully specify the patching timeframe, testing "
                "process, or controlled deployment procedure"
            ),

            "suggestion": (
                "The system shall apply approved security patches within "
                "[severity-based organisational timeframe], test the patches "
                "in [approved pre-production environment] before deployment, "
                "and provide an approved rollback procedure."
            )
        }
    ]
}


# ============================================================
# MATCHING FUNCTIONS
# ============================================================

def normalise_text(text):
    """
    Normalises text only for matching.
    It does not modify the original displayed requirement.
    """

    text = str(text).lower()
    text = text.replace("’", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_term(text, term):
    """
    Checks whether a word or phrase appears in the requirement.
    """

    text = normalise_text(text)
    term = normalise_text(term)

    if not term:
        return False

    # Phrase or symbol-based matching
    if (
        " " in term
        or "-" in term
        or "%" in term
        or "/" in term
    ):
        return term in text

    pattern = rf"\b{re.escape(term)}\b"

    return re.search(pattern, text) is not None


def any_term_present(text, terms):
    """
    Returns True if at least one listed term appears.
    """

    return any(
        contains_term(text, term)
        for term in terms
    )


def trigger_group_present(text, trigger_group):
    """
    Returns True if every term in one trigger group appears.
    """

    return all(
        contains_term(text, term)
        for term in trigger_group
    )


def find_missing_groups(requirement, required_groups):
    """
    Returns the names of detail groups not found in the requirement.
    """

    missing_groups = []

    for group_name, indicators in required_groups.items():

        group_found = any_term_present(
            requirement,
            indicators
        )

        if not group_found:
            missing_groups.append(group_name)

    return missing_groups


# ============================================================
# RULE EVALUATION
# ============================================================

def evaluate_rule(requirement, rule):
    """
    Evaluates one requirement against one rule.

    Returns a scored recommendation when:
    - the concept is relevant;
    - no exclusion phrase is found;
    - enough important details are missing.

    Otherwise, returns None.
    """

    requirement_lower = normalise_text(requirement)

    # Check exclusions
    if any_term_present(
        requirement_lower,
        rule.get("exclude_if", [])
    ):
        return None

    # Strong trigger matches
    strong_matches = [
        trigger
        for trigger in rule.get("strong_triggers", [])
        if contains_term(requirement_lower, trigger)
    ]

    # Trigger-group matches
    group_matches = [
        group
        for group in rule.get("trigger_groups", [])
        if trigger_group_present(requirement_lower, group)
    ]

    concept_relevant = (
        len(strong_matches) > 0
        or len(group_matches) > 0
    )

    if not concept_relevant:
        return None

    # Find missing details
    missing_details = find_missing_groups(
        requirement_lower,
        rule.get("required_groups", {})
    )

    minimum_missing = rule.get(
        "minimum_missing",
        1
    )

    if len(missing_details) < minimum_missing:
        return None

    # Score used to select one best recommendation
    score = (
        rule.get("priority", 0) * 10
        + len(strong_matches) * 5
        + len(group_matches) * 4
        + len(missing_details) * 3
    )

    return {
        "concept": rule["concept"],
        "issue": rule["issue"],
        "suggestion": rule["suggestion"],
        "missing_details": missing_details,
        "score": score
    }


# ============================================================
# DETECT ONE BEST RECOMMENDATION
# ============================================================

def detect_vagueness(requirement, cia_category):
    """
    Returns one best recommendation, or None.

    Each requirement receives at most one recommendation.
    """

    if not isinstance(requirement, str):
        return None

    requirement = requirement.strip()

    if not requirement:
        return None

    if cia_category not in [
        "Confidentiality",
        "Integrity",
        "Availability"
    ]:
        return None

    candidates = []

    rules = SECURITY_IMPROVEMENT_RULES.get(
        cia_category,
        []
    )

    for rule in rules:

        result = evaluate_rule(
            requirement,
            rule
        )

        if result is not None:
            candidates.append(result)

    if not candidates:
        return None

    # Select only one highest-scoring recommendation
    best_result = max(
        candidates,
        key=lambda item: item["score"]
    )

    return best_result


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(results_df):
    """
    Processes only requirements classified as Security.

    A requirement receives:
    - zero recommendations; or
    - one recommendation.
    """

    recommendations = []

    required_columns = [
        "Sentence",
        "Phase I (Type)",
        "Phase II (CIA)"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        st.error(
            "Recommendation system cannot run because these "
            f"columns are missing: {', '.join(missing_columns)}"
        )
        return recommendations

    security_rows = results_df[
        results_df["Phase I (Type)"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("security")
    ]

    for original_index, row in security_rows.iterrows():

        requirement = str(
            row["Sentence"]
        ).strip()

        cia_category = str(
            row["Phase II (CIA)"]
        ).strip()

        if not requirement:
            continue

        if cia_category in [
            "",
            "N/A",
            "NA",
            "None",
            "nan"
        ]:
            continue

        best_result = detect_vagueness(
            requirement,
            cia_category
        )

        # No recommendation is needed
        if best_result is None:
            continue

        recommendations.append({
            "req_number": original_index + 1,
            "requirement": requirement,
            "cia": cia_category,
            "concept": best_result["concept"],
            "issue": best_result["issue"],
            "missing_details": best_result["missing_details"],
            "suggestion": best_result["suggestion"]
        })

    return recommendations


# ============================================================
# RUN RECOMMENDATION SYSTEM
# ============================================================

recs = generate_recommendations(
    results_df
)


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

if not recs:

    st.success(
        "No security requirements require an improvement "
        "based on the current recommendation rules."
    )

else:

    st.info(
        f"**{len(recs)} improvement suggestion(s) found.** "
        "Each requirement receives at most one suggestion."
    )

    for recommendation_number, rec in enumerate(
        recs,
        start=1
    ):

        title = (
            f"Suggestion {recommendation_number} — "
            f"Requirement #{rec['req_number']} "
            f"[{rec['cia']}]"
        )

        with st.expander(title):

            st.write("**Original Requirement:**")
            st.info(rec["requirement"])

            st.write(
                f"**CIA Category:** `{rec['cia']}`"
            )

            st.write(
                f"**Security Concept:** `{rec['concept']}`"
            )

            st.write("**Issue Detected:**")
            st.warning(
                f"This requirement {rec['issue']}."
            )

            st.write("**Missing Details:**")

            for detail in rec["missing_details"]:
                st.write(
                    f"- {detail.capitalize()}"
                )

            st.write("**Recommended Improvement:**")
            st.success(rec["suggestion"])

            st.write("**Reason:**")
            st.caption(
                "The suggested revision adds important missing "
                "details so that the security requirement is "
                "clearer and easier to test. Values inside square "
                "brackets must be decided by project stakeholders."
            )


# ============================================================
# ADD RECOMMENDATIONS TO REPORT
# ============================================================

st.write("---")

if st.button(
    "Add Recommendations to Report",
    type="primary"
):

    st.session_state["recs"] = recs

    if recs:
        st.success(
            "Recommendations added. See the table below."
        )
    else:
        st.info(
            "There are no recommendations to add."
        )


# ============================================================
# RECOMMENDATIONS TABLE
# ============================================================

if (
    "recs" in st.session_state
    and st.session_state["recs"]
):

    st.write("### Recommendations Table")

    rec_df = pd.DataFrame([
        {
            "Req #": rec["req_number"],
            "Original Requirement": rec["requirement"],
            "CIA Category": rec["cia"],
            "Concept": rec["concept"],
            "Issue": rec["issue"],
            "Missing Details": ", ".join(
                rec["missing_details"]
            ),
            "Recommended Improvement": rec["suggestion"],
            "Reason": (
                "The recommendation adds missing details "
                "to make the requirement clearer and more testable."
            )
        }
        for rec in st.session_state["recs"]
    ])

    st.dataframe(
        rec_df,
        use_container_width=True
    )


    # CSV DOWNLOAD

    rec_csv = rec_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Download Recommendations as CSV",
        data=rec_csv,
        file_name="security_recommendations.csv",
        mime="text/csv"
    )

    # TXT DOWNLOAD

    rec_txt_lines = [
        "SECURITY REQUIREMENTS RECOMMENDATIONS",
        "=" * 70
    ]

    for rec in st.session_state["recs"]:

        missing_details_text = ", ".join(
            rec["missing_details"]
        )

        rec_txt_lines.extend([
            "",
            (
                f"Requirement #{rec['req_number']} "
                f"[{rec['cia']}]"
            ),
            f"Original        : {rec['requirement']}",
            f"Concept         : {rec['concept']}",
            f"Issue           : This requirement {rec['issue']}.",
            f"Missing Details : {missing_details_text}",
            f"Suggested       : {rec['suggestion']}",
            (
                "Reason          : The recommendation adds "
                "missing details so that the requirement is "
                "clearer and more testable."
            ),
            (
                "Note            : Values inside square brackets "
                "must be determined by project stakeholders."
            ),
            "-" * 70
        ])

    rec_txt = "\n".join(
        rec_txt_lines
    )

    st.download_button(
        label="Download Recommendations as TXT",
        data=rec_txt.encode("utf-8"),
        file_name="security_recommendations.txt",
        mime="text/plain"
    )
            
#Clear results button------------------------------------------------------------

if "results_df" in st.session_state:
    st.write("---")
    st.write("### Start a New Classification?")
    st.caption("Clear current results and classify a new set of requirements.")
    if st.button("Clear Results", type="primary", use_container_width=True):
        for key in ["results_df", "sentences", "cleaned_sentences", "combined_text", "input_was_csv", "recs"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
 
# Footer------------------------------------------------------------------------

st.write("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "AI-Assisted Security Requirements Identifier | RoBERTa-base | Two-Phase Classification"
    "</div>",
    unsafe_allow_html=True
)



  
