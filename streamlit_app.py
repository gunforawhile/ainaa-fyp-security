import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
from collections import Counter

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
 
#Word-Net Synonym Augmentation (Class Imbalance Handling)--------------------------------
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
 
def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    tokens = [w for w in sentence.split() if w not in STOP_WORDS]
    return " ".join(tokens)
 
def preprocess_pipeline(combined_text):
    expanded = expand_abbreviations(combined_text)
    sentences = sent_tokenize(expanded)
    cleaned = [clean_sentence(s) for s in sentences]
    pairs = [(o, c) for o, c in zip(sentences, cleaned) if c]
    return [p[0] for p in pairs], [p[1] for p in pairs]
 
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

#Classification Functions------------------------------------------------------------

def phase1_classify(sentence, classifier):
    results = classifier(sentence)[0]
    best = max(results, key=lambda x: x["score"])
    raw = best["label"]
    if raw in PHASE1_ID2LABEL.values():
        label = raw
    else:
        label = PHASE1_ID2LABEL[int(raw.split("_")[-1])]
    return label, best["score"]
 
def phase2_classify(sentence, classifier):
    results = classifier(sentence)[0]
    best = max(results, key=lambda x: x["score"])
    raw = best["label"]
    if raw in PHASE2_ID2LABEL.values():
        label = raw
    else:
        label = PHASE2_ID2LABEL[int(raw.split("_")[-1])]
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
 
    # ── Phase I metrics ───────────────────────────────────────────────────
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
 
    # ── Phase II metrics ──────────────────────────────────────────────────
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
 
    # ── Summary bar chart ─────────────────────────────────────────────────
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
    st.caption("Fine-tuned RoBERTa-base · C / I / A")
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
            st.error(f"❌ Failed to load {uf.name}: {e}")

st.write("")
start = st.button("Start Identifying Requirements", type="primary", use_container_width=True)
 
if start:
    if not all_text:
        st.warning("Please enter text or upload a file first.")
    else:
        combined_text = "\n\n".join(all_text)
 
        # ── Preprocessing ─────────────────────────────────────────────────
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
 
        # ── Classification ────────────────────────────────────────────────
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
 
        # ── Charts ────────────────────────────────────────────────────────
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
        st.dataframe(display_df, use_container_width=True)
 
        csv = display_df.to_csv(index=False)
        st.download_button("⬇️ Download Results as CSV", csv, "classification_results.csv", "text/csv")
 
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
 
#Evaluation -------------------------------------------------------------------
st.write("---")
st.write("## Model Evaluation")
st.caption(f"Evaluates both phases against the PROMISE NFR labelled dataset. Target F1: {TARGET_F1:.0%}")
 
promise_dataset = load_promise_nfr_dataset()
 
if not promise_dataset:
    st.error(f"PROMISE NFR dataset not found at `{PROMISE_NFR_PATH}`. Add `data/nfr.txt` alongside the app.")
else:
    f_count  = len([d for d in promise_dataset if d[1] == "Functional"])
    se_count = len([d for d in promise_dataset if d[1] == "Security"])
    st.info(f"{len(promise_dataset)} labelled sentences loaded — {f_count} Functional, {se_count} Security.")
 
    use_aug   = st.checkbox("Use augmented dataset (class-balanced)", value=False)
    sample_n  = st.slider("Sentences to evaluate", 20, len(promise_dataset),
                          min(50, len(promise_dataset)), step=10)
 
    if st.button("Run Evaluation"):
        eval_data = promise_dataset
        eval_data = random.sample(eval_data, sample_n)
        if use_aug:
            eval_data = augment_minority_class(eval_data)
        with st.spinner("Running evaluation..."):
            evaluate_model(eval_data)
 
#Demo----------------------------------------------------------------------
st.write("---")
st.write("## WordNet Synonym Augmentation Demo")
st.caption("Shows how the minority Security class is augmented using synonym replacement to address class imbalance.")
 
aug_data = load_promise_nfr_dataset()
if aug_data:
    sec_only  = [d for d in aug_data if d[1] == "Security"]
    func_only = [d for d in aug_data if d[1] == "Functional"]
 
    c1, c2 = st.columns(2)
    c1.metric("Original Security Count",    len(sec_only))
    c2.metric("Original Functional Count",  len(func_only))
 
    if st.button("🔄 Generate Augmented Samples"):
        aug_full = augment_minority_class(aug_data)
        new_sec  = len([d for d in aug_full if d[1] == "Security"])
        st.success(f"Security class: {len(sec_only)} → {new_sec} after augmentation")
 
        preview = [{
            "Original":  orig[0],
            "Augmented": aug[0],
            "Label":     aug[1]
        } for orig, aug in zip(sec_only, aug_full[len(aug_data):])]
        st.dataframe(pd.DataFrame(preview), use_container_width=True)
 
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



  
