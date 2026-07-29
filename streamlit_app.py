import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from transformers import pipeline
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
#from collections import Counter

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
        # Remove content inside parentheses temporarily to find safe split points
        temp = re.sub(r'\([^)]*\)', lambda m: 'X' * len(m.group()), text)
        # Split on period+space+capital, but NOT after known abbreviations
        parts = re.split(r'(?<!\b(?:e\.g|i\.e|etc|vs|Dr|Mr|Ms|No))\.\s+(?=[A-Z])', temp)
        if len(parts) > 1:
            # Apply split positions back to original text
            positions = [0]
            pos = 0
            for part in parts[:-1]:
                pos += len(part) + 2  # +2 for '. '
                positions.append(pos)
            positions.append(len(text))
            lines = [text[positions[i]:positions[i+1]].strip() for i in range(len(parts))]
 
    return [r for r in lines if len(r.split()) >= 3]
 
def clean_for_display(sentence):
    """
    What is kept:
      - Numbers (e.g. '99.9%', 'AES-256', '5 minutes') — carry meaning
      - Important words like 'as', 'only', 'not', 'must' — affect security meaning
      - Special characters that are part of technical terms
    """
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

#Classification Functions---------------------------------------------------------------

def phase1_classify(sentence, classifier):
    raw = classifier(sentence)
    if isinstance(raw[0], list):
        results = raw[0]
    else:
        results = raw
    best = max(results, key=lambda x: x["score"])
    raw_label = best["label"]
    if raw_label in PHASE1_ID2LABEL.values():
        label = raw_label
    else:
        label = PHASE1_ID2LABEL[int(raw_label.split("_")[-1])]
    return label, best["score"]
 
def phase2_classify(sentence, classifier):
    raw = classifier(sentence)
    if isinstance(raw[0], list):
        results = raw[0]
    else:
        results = raw
    best = max(results, key=lambda x: x["score"])
    raw_label = best["label"]
    if raw_label in PHASE2_ID2LABEL.values():
        label = raw_label
    else:
        label = PHASE2_ID2LABEL[int(raw_label.split("_")[-1])]
    return label, best["score"]
 
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
 
        # Preprocessing 
        st.write("---")
        st.write("## Preprocessing")
 
        sentences, cleaned_sentences = preprocess_pipeline(combined_text)
 
        with st.expander("View raw text"):
            st.text_area("Raw", combined_text[:1000] + ("..." if len(combined_text) > 1000 else ""), height=150)
 
        with st.expander("View before vs after cleaning"):
            st.dataframe(pd.DataFrame({
                "Original Sentence": sentences[:len(cleaned_sentences)],
                "Cleaned Sentence":  cleaned_sentences
            }), use_container_width=True)
 
        st.success(f"{len(sentences)} sentences extracted and preprocessed")
 
        #Classification 
        st.write("---")
        st.write("## Classification Results")
 
        with st.spinner("Loading models and classifying..."):
            results_df = classify_requirements(sentences)
 
        st.session_state["results_df"] = results_df
 
        total          = len(results_df)
        security_count = len(results_df[results_df["Phase I (Type)"] == "Security"])
        functional_count = total - security_count
 
        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Sentences",        total)
        c2.metric("Security Requirements",  security_count)
        c3.metric("Functional Requirements",functional_count)
 
        #Charts 
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
 
         # ── Full results table ────────────────────────────────────────────
        st.write("### Full Classification Table")
        display_df = results_df.copy()
        display_df["Phase I Confidence"]  = display_df["Phase I Confidence"].apply(lambda x: f"{x:.0%}")
        display_df["Phase II Confidence"] = display_df["Phase II Confidence"].apply(
            lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
 
        # Add short label columns — F/SE and C/I/A
        display_df["Type Label"] = display_df["Phase I (Type)"].map(
            {"Security": "SE", "Functional": "F"}
        )
        display_df["CIA Label"] = display_df["Phase II (CIA)"].map(
            {"Confidentiality": "C", "Integrity": "I", "Availability": "A"}
        ).fillna("-")
 
        # Reorder for clarity
        display_df = display_df[[
            "Sentence",
            "Type Label", "Phase I (Type)", "Phase I Confidence",
            "CIA Label",  "Phase II (CIA)", "Phase II Confidence"
        ]]
        st.dataframe(display_df, use_container_width=True)
 
        # ── Format-aware download ─────────────────────────────────────
        st.write("#### Download Results")
 
        input_was_csv = any(
            uf.name.lower().endswith(".csv") for uf in uploaded_files
        ) if uploaded_files else False
 
        # Always offer CSV
        csv_out = display_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download as CSV",
            csv_out, "classification_results.csv", "text/csv"
        )
 
        # Offer TXT if input was txt or text area
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
                "⬇️ Download as TXT",
                txt_out, "classification_results.txt", "text/plain"
            )
 
        # ── Summary Report ────────────────────────────────────────────────
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
 
# Footer------------------------------------------------------------------------

st.write("---")
st.caption("AI-Assisted Security Requirements Identifier | RoBERTa-base | Two-Phase Classification")




  
