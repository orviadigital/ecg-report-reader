import re
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import fitz
except ImportError:
    fitz = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ECG Report Reader",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero {
        padding: 35px;
        border-radius: 24px;
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1px solid #dbeafe;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 18px;
        color: #475569;
        line-height: 1.6;
    }

    .result-card {
        padding: 28px;
        border-radius: 20px;
        margin: 20px 0;
        border: 1px solid #e2e8f0;
    }

    .normal-card {
        background: #f0fdf4;
        border-color: #86efac;
    }

    .abnormal-card {
        background: #fef2f2;
        border-color: #fca5a5;
    }

    .review-card {
        background: #fffbeb;
        border-color: #fcd34d;
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .result-description {
        font-size: 16px;
        line-height: 1.7;
        color: #334155;
    }

    .metric-card {
        padding: 20px;
        border-radius: 16px;
        background: white;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
    }

    .metric-name {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 800;
        margin: 6px 0;
    }

    .metric-range {
        font-size: 14px;
        color: #64748b;
    }

    .section-title {
        font-size: 28px;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 15px;
    }

    .doctor-box {
        padding: 22px;
        border-radius: 16px;
        background: #fff7ed;
        border: 1px solid #fdba74;
        margin-top: 20px;
    }

    .doctor-box strong {
        color: #9a3412;
    }

    .disclaimer {
        padding: 25px;
        border-radius: 18px;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        margin-top: 35px;
        color: #475569;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REFERENCE RANGES
# ============================================================

def ranges(sex):
    """
    Educational adult reference ranges.

    These are not universal diagnostic cutoffs.
    """

    if sex == "Male":
        qtclow, qtchi = 350, 450
    elif sex == "Female":
        qtclow, qtchi = 360, 460
    else:
        qtclow, qtchi = 350, 460

    return {
        "Heart Rate": (60, 100, "bpm"),
        "PR Interval": (120, 200, "ms"),
        "QRS Duration": (70, 110, "ms"),
        "QT": (350, 450, "ms"),
        "QTc": (qtclow, qtchi, "ms"),
        "QRS Axis": (-30, 90, "°"),
        "P Axis": (0, 75, "°"),
        "T Axis": (15, 75, "°"),
    }


# ============================================================
# PDF TO IMAGE
# ============================================================

def pdf_images(data):

    if fitz is None:
        return []

    try:
        doc = fitz.open(
            stream=data,
            filetype="pdf"
        )
    except Exception:
        return []

    result = []

    for page in doc:

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2.5, 2.5),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        result.append(image)

    return result


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(img):

    gray = ImageOps.autocontrast(
        img.convert("L")
    )

    gray = ImageEnhance.Contrast(
        gray
    ).enhance(1.5)

    gray = ImageEnhance.Sharpness(
        gray
    ).enhance(2.0)

    return gray


# ============================================================
# OCR
# ============================================================

def ocr_passes(img):

    if pytesseract is None:
        return []

    try:

        base = preprocess(img)

        base = base.resize(
            (
                img.width * 4,
                img.height * 4
            )
        )

        outputs = []

        for angle in (-2, -1, 0, 1, 2):

            rotated = base.rotate(
                angle,
                expand=True,
                fillcolor="white"
            )

            threshold = rotated.point(
                lambda p: 255 if p > 165 else 0
            )

            for variant in (
                rotated,
                threshold
            ):

                for psm in (6, 11):

                    try:

                        text = pytesseract.image_to_string(
                            variant,
                            config=f"--psm {psm}"
                        )

                        if text:
                            outputs.append(text)

                    except Exception:
                        continue

        return outputs

    except Exception:
        return []


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_line(s):

    return re.sub(
        r"\s+",
        " ",
        s
    ).strip()


# ============================================================
# NUMERIC MEASUREMENT EXTRACTION
# ============================================================

def labeled_candidates(texts):

    patterns = {

        "Heart Rate": [
            r"\brate\b",
            r"\bv[- ]?rate\b",
            r"\bvent(?:ricular)?\s*rate\b",
        ],

        "PR Interval": [
            r"\bpr\b",
            r"\bp[- ]?r\b",
        ],

        "QRS Duration": [
            r"\bqrsd\b",
            r"\bqrs[- ]?d\b",
            r"\bqrs\b",
        ],

        "QTc": [
            r"\bqtc\b",
            r"\bqt[- ]?c\b",
        ],

        "QT": [
            r"\bqt\b(?!c)",
        ],
    }

    bounds = {

        "Heart Rate": (25, 220),

        "PR Interval": (60, 500),

        "QRS Duration": (40, 250),

        "QT": (150, 700),

        "QTc": (250, 700),
    }

    out = {
        key: []
        for key in patterns
    }

    for text in texts:

        for raw in text.splitlines():

            line = clean_line(raw)

            if len(line) > 180:
                line = line[:180]

            for field, pats in patterns.items():

                for pat in pats:

                    pattern = (
                        pat
                        + r"\s*[:=]?\s*"
                        + r"([+-]?\d{1,4}(?:\.\d+)?)\b"
                    )

                    match = re.search(
                        pattern,
                        line,
                        re.I
                    )

                    if not match:
                        continue

                    try:

                        value = int(
                            round(
                                float(
                                    match.group(1)
                                )
                            )
                        )

                    except ValueError:
                        continue

                    lo, hi = bounds[field]

                    if lo <= value <= hi:
                        out[field].append(value)

    return out


# ============================================================
# AXIS EXTRACTION
# ============================================================

def axis_candidates(texts):

    out = {
        "P Axis": [],
        "QRS Axis": [],
        "T Axis": [],
    }

    for text in texts:

        t = re.sub(
            r"\s+",
            " ",
            text
        )

        blocks = re.findall(
            r"AXIS.*?(?=12\s*lead|standard placement|abnormal ecg|unconfirmed diagnosis|$)",
            t,
            re.I | re.S
        )

        for block in blocks:

            fields = [
                ("P Axis", "P"),
                ("QRS Axis", "QRS"),
                ("T Axis", "T"),
            ]

            for field, label in fields:

                pattern = (
                    r"\b"
                    + label
                    + r"\s*[:=]?\s*"
                    + r"([+-]?\d{1,3})\b"
                )

                match = re.search(
                    pattern,
                    block,
                    re.I
                )

                if match:

                    try:
                        value = int(
                            match.group(1)
                        )
                    except ValueError:
                        continue

                    if -180 <= value <= 180:
                        out[field].append(value)

    return out


# ============================================================
# CONSENSUS / OCR CONFIDENCE
# ============================================================

def consensus(values):

    if not values:
        return None, 0.0, []

    counts = {}

    for value in values:

        counts[value] = (
            counts.get(value, 0)
            + 1
        )

    best = max(
        counts,
        key=counts.get
    )

    total = len(values)

    support = counts[best]

    conf = min(
        0.98,
        0.45 + 0.10 * support
    )

    if len(counts) > 1:
        conf = min(
            conf,
            0.72
        )

    alternatives = sorted(
        counts,
        key=counts.get,
        reverse=True
    )[:5]

    return (
        best,
        conf,
        alternatives
    )


# ============================================================
# MACHINE INTERPRETATION
# ============================================================

def machine_interpretation(texts):

    keywords = [

        "sinus rhythm",

        "normal sinus rhythm",

        "abnormal r-wave progression",

        "early transition",

        "st elevation",

        "st depression",

        "consider anterior injury",

        "infarct",

        "ischemia",

        "ischemic",

        "atrial fibrillation",

        "atrial flutter",

        "tachycardia",

        "bradycardia",

        "bundle branch block",

        "heart block",

        "abnormal ecg",

        "abnormal ekg",

        "unconfirmed diagnosis",

        "normal p axis",

        "normal qrs axis",

        "qrs area",
    ]

    found = []

    for text in texts:

        for raw in text.splitlines():

            line = clean_line(raw)

            low = line.lower()

            if (
                len(line) >= 5
                and any(
                    keyword in low
                    for keyword in keywords
                )
            ):

                if line not in found:
                    found.append(line)

    return found


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(
    value,
    lo,
    hi
):

    if value is None:

        return (
            "missing",
            "Not confidently detected"
        )

    if lo <= value <= hi:

        return (
            "normal",
            f"Within reference range ({lo:g}–{hi:g})"
        )

    if value < lo:

        return (
            "low",
            f"Below reference range ({lo:g}–{hi:g})"
        )

    return (
        "high",
        f"Above reference range ({lo:g}–{hi:g})"
    )


# ============================================================
# SIMPLE EXPLANATIONS
# ============================================================

def simple_explanation(
    field,
    status
):

    if status == "normal":

        explanations = {

            "Heart Rate":
                "The heart rate is within the usual adult resting reference range.",

            "PR Interval":
                "The electrical signal is taking a typical amount of time to travel from the atria toward the ventricles.",

            "QRS Duration":
                "The ventricular electrical signal is not prolonged by this measurement.",

            "QT":
                "The measured QT interval is within the broad educational reference shown here. QT depends on heart rate.",

            "QTc":
                "The corrected QT interval is within the selected adult reference range.",

            "QRS Axis":
                "The main ventricular electrical direction is within the usual adult axis reference.",

            "P Axis":
                "The P-wave axis is within the educational reference used by this prototype.",

            "T Axis":
                "The T-wave axis is within the educational reference used by this prototype.",
        }

        return explanations.get(
            field,
            "Within the reference range."
        )

    if status == "low":

        return (
            "This value is below the reference range. "
            "It should be checked against the original ECG "
            "and clinical context."
        )

    if status == "high":

        return (
            "This value is above the reference range. "
            "It should be checked against the original ECG "
            "and clinical context."
        )

    return (
        "The tool could not confidently read this value."
    )


# ============================================================
# MACHINE INTERPRETATION → PLAIN LANGUAGE
# ============================================================

def interpretation_plain_language(line):

    l = line.lower()

    if "normal sinus rhythm" in l:

        return (
            "The ECG machine reports a normal sinus rhythm, "
            "meaning the heartbeat is being initiated from the "
            "heart's usual pacemaker."
        )

    if "sinus rhythm" in l:

        return (
            "The ECG machine reports a sinus rhythm. "
            "This describes where the heartbeat appears to originate, "
            "but it does not by itself prove that the entire ECG is normal."
        )

    if (
        "st elevation" in l
        or "consider anterior injury" in l
    ):

        return (
            "The ECG machine is reporting ST-segment elevation "
            "and/or a possible anterior injury pattern. "
            "This cannot be confirmed from OCR alone and requires "
            "clinical review."
        )

    if "st depression" in l:

        return (
            "The ECG machine is reporting ST-segment depression. "
            "This finding can have different causes and should be "
            "reviewed by a healthcare professional."
        )

    if "abnormal r-wave progression" in l:

        return (
            "The machine reports an unusual R-wave progression "
            "across the chest leads. This can have several causes "
            "and should be interpreted from the actual ECG tracing."
        )

    if "early transition" in l:

        return (
            "The machine reports an early change in the R-wave "
            "pattern across the chest leads. The actual tracing "
            "and clinical context are needed for interpretation."
        )

    if "abnormal ecg" in l or "abnormal ekg" in l:

        return (
            "The ECG machine has labelled the tracing as abnormal. "
            "An automated ECG label is not a diagnosis and should "
            "be reviewed by a clinician."
        )

    if "unconfirmed diagnosis" in l:

        return (
            "The printed diagnosis is marked as unconfirmed. "
            "It should not be treated as a confirmed diagnosis."
        )

    if "tachycardia" in l:

        return (
            "The machine is reporting a fast heart rhythm."
        )

    if "bradycardia" in l:

        return (
            "The machine is reporting a slow heart rhythm."
        )

    if "atrial fibrillation" in l:

        return (
            "The machine is reporting a possible irregular atrial "
            "rhythm called atrial fibrillation. This requires clinical confirmation."
        )

    if "atrial flutter" in l:

        return (
            "The machine is reporting a possible atrial flutter rhythm. "
            "This requires confirmation by a healthcare professional."
        )

    if "block" in l:

        return (
            "The machine is reporting a possible conduction block. "
            "The actual ECG should be reviewed by a healthcare professional."
        )

    if (
        "ischemia" in l
        or "ischemic" in l
        or "infarct" in l
    ):

        return (
            "The machine is reporting a possible ischemic or "
            "injury-related finding. This requires professional review."
        )

    return (
        "The machine-generated interpretation contains this finding. "
        "Review the original ECG with a healthcare professional."
    )


# ============================================================
# REPORT STATUS
# ============================================================

def determine_report_status(
    interpretation,
    abnormal_measurements,
    confident_count
):

    text = " ".join(
        interpretation
    ).lower()

    # Findings that should trigger an abnormal/review result
    abnormal_keywords = [

        "abnormal ecg",
        "abnormal ekg",

        "st elevation",
        "st depression",

        "consider anterior injury",

        "infarct",

        "ischemia",
        "ischemic",

        "atrial fibrillation",
        "atrial flutter",

        "tachycardia",
        "bradycardia",

        "heart block",
        "bundle branch block",

        "abnormal r-wave progression",

        "early transition",

        "prolonged qt",
        "prolonged qtc",

        "abnormal t wave",
        "abnormal p wave",

        "left ventricular hypertrophy",
        "right ventricular hypertrophy",

        "low voltage",

        "axis deviation",
    ]

    found_abnormal = []

    for keyword in abnormal_keywords:

        if keyword in text:

            if keyword not in found_abnormal:
                found_abnormal.append(
                    keyword
                )

    # Any measurement outside the reference range
    # also requires review.
    if found_abnormal or abnormal_measurements:

        return (
            "abnormal",
            found_abnormal
        )

    # Explicit normal machine wording
    normal_keywords = [
        "normal ecg",
        "normal ekg",
        "normal sinus rhythm",
    ]

    if any(
        keyword in text
        for keyword in normal_keywords
    ):

        return (
            "normal",
            []
        )

    # If measurements were successfully extracted
    # but no interpretation was found, we should NOT
    # claim the entire ECG is normal.
    if confident_count > 0:

        return (
            "review",
            []
        )

    return (
        "uncertain",
        []
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>❤️ ECG Report Reader</h1>

        <p>
            Upload your ECG report and understand the printed
            measurements and findings in simple, easy-to-read language.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMPORTANT WARNING
# ============================================================

st.warning(
    "⚠️ Important: This is an educational screening tool, "
    "not a diagnostic medical device. It reads information "
    "printed on an ECG report. A qualified healthcare professional "
    "must confirm the ECG and any abnormal findings."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Patient information"
    )

    sex = st.selectbox(
        "Sex for QTc reference",
        [
            "Not specified",
            "Male",
            "Female"
        ]
    )

    st.caption(
        "Adult reference ranges are used. "
        "Pediatric ECGs require different reference ranges."
    )

    st.divider()

    st.caption(
        "Reference ranges are educational adult ranges "
        "and can vary with age, sex, method, device and clinical context."
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload ECG report photo or PDF",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
        "pdf"
    ]
)


# ============================================================
# PROCESS UPLOAD
# ============================================================

if uploaded:

    data = uploaded.read()

    if uploaded.name.lower().endswith(".pdf"):

        images = pdf_images(
            data
        )

    else:

        try:

            image = Image.open(
                uploaded
            )

            images = [image]

        except Exception:

            images = []

    if not images:

        st.error(
            "❌ Could not read this file. "
            "Please upload a clear ECG image or PDF."
        )

        st.stop()


    # ========================================================
    # REPORT PREVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">1. Your ECG Report</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(
        min(3, len(images))
    )

    for i, image in enumerate(images):

        with cols[
            i % len(cols)
        ]:

            st.image(
                image,
                caption=f"Page {i + 1}",
                use_container_width=True
            )


    # ========================================================
    # ANALYZE
    # ========================================================

    if st.button(
        "🔎 Analyze ECG Report",
        type="primary",
        use_container_width=True
    ):

        if pytesseract is None:

            st.error(
                "OCR is not available. "
                "Please make sure pytesseract is installed."
            )

            st.stop()


        with st.spinner(
            "Reading the ECG report..."
        ):

            texts = []

            for image in images:

                page_text = ocr_passes(
                    image
                )

                texts.extend(
                    page_text
                )


        if not texts:

            st.error(
                "❌ The tool could not read the report. "
                "Please upload a clearer image or PDF."
            )

            st.stop()


        # ====================================================
        # EXTRACT MEASUREMENTS
        # ====================================================

        raw = labeled_candidates(
            texts
        )

        raw.update(
            axis_candidates(texts)
        )

        extracted = {
            field: consensus(values)
            for field, values in raw.items()
        }

        interp = machine_interpretation(
            texts
        )

        ref = ranges(
            sex
            if sex != "Not specified"
            else "Unknown"
        )


        # ====================================================
        # COUNT CONFIDENT VALUES
        # ====================================================

        confident = 0

        abnormal_measurements = []

        uncertain = []

        for field, (
            value,
            conf,
            alternatives
        ) in extracted.items():

            if (
                value is not None
                and conf >= 0.75
            ):

                confident += 1

                status, _ = classify(
                    value,
                    *ref[field][:2]
                )

                if status != "normal":

                    abnormal_measurements.append(
                        field
                    )

            elif value is not None:

                uncertain.append(
                    field
                )


        # ====================================================
        # OVERALL REPORT STATUS
        # ====================================================

        report_status, status_reasons = (
            determine_report_status(
                interp,
                abnormal_measurements,
                confident
            )
        )


        # ====================================================
        # SIMPLE RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">2. Simple Result</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        if report_status == "normal":

            st.markdown(
                """
                <div class="result-card normal-card">

                    <div class="result-title">
                        🟢 ECG REPORTED AS NORMAL
                    </div>

                    <div class="result-description">

                        The printed ECG report appears to be
                        <strong>reported as normal</strong> based on
                        the information that this tool could read.

                        <br><br>

                        No supported abnormal finding was detected
                        in the printed interpretation.

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "This does not guarantee that the ECG is medically "
                "normal. The original ECG should be reviewed by a "
                "qualified healthcare professional when appropriate."
            )


        # ----------------------------------------------------
        # ABNORMAL
        # ----------------------------------------------------

        elif report_status == "abnormal":

            st.markdown(
                """
                <div class="result-card abnormal-card">

                    <div class="result-title">
                        🔴 ECG REPORTED AS ABNORMAL
                    </div>

                    <div class="result-description">

                        The ECG report contains one or more findings
                        that were flagged as abnormal or measurements
                        that fall outside the reference ranges shown
                        below.

                        <br><br>

                        <strong>
                        Please contact your doctor or an appropriate
                        healthcare specialist for professional review.
                        </strong>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # Reported findings
            if status_reasons:

                st.markdown(
                    "### ⚠️ Reported Finding(s)"
                )

                readable_reasons = []

                for reason in status_reasons:

                    readable_reasons.append(
                        reason.replace(
                            "_",
                            " "
                        ).title()
                    )

                for reason in dict.fromkeys(
                    readable_reasons
                ):

                    st.write(
                        f"• **{reason}**"
                    )


            # Abnormal measurements
            if abnormal_measurements:

                st.markdown(
                    "### 📊 Measurements Outside Reference Range"
                )

                for field in abnormal_measurements:

                    value, conf, alternatives = (
                        extracted[field]
                    )

                    lo, hi, unit = ref[field]

                    status, _ = classify(
                        value,
                        lo,
                        hi
                    )

                    if status == "low":

                        st.error(
                            f"🔴 **{field}: {value} {unit}** — "
                            f"below the typical adult reference "
                            f"range of **{lo}–{hi} {unit}**."
                        )

                    elif status == "high":

                        st.error(
                            f"🔴 **{field}: {value} {unit}** — "
                            f"above the typical adult reference "
                            f"range of **{lo}–{hi} {unit}**."
                        )


            st.markdown(
                """
                <div class="doctor-box">

                    🩺 <strong>
                    Please contact your doctor or appropriate
                    healthcare specialist.
                    </strong>

                    <br><br>

                    An abnormal result from this tool does not
                    establish a diagnosis. It means the printed
                    ECG report contains a finding that deserves
                    professional review.

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # REVIEW REQUIRED
        # ----------------------------------------------------

        elif report_status == "review":

            st.markdown(
                """
                <div class="result-card review-card">

                    <div class="result-title">
                        🟡 ECG REQUIRES PROFESSIONAL REVIEW
                    </div>

                    <div class="result-description">

                        Some ECG measurements could be read, but
                        the tool could not confidently classify the
                        entire printed ECG report as normal.

                        <br><br>

                        This does <strong>not</strong> mean the ECG
                        is abnormal. It means the available information
                        is not enough for this tool to safely label
                        the entire report as normal.

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "Please review the original ECG with your doctor "
                "or an appropriate healthcare professional."
            )


        # ----------------------------------------------------
        # UNCERTAIN
        # ----------------------------------------------------

        else:

            st.markdown(
                """
                <div class="result-card review-card">

                    <div class="result-title">
                        🟡 COULD NOT CONFIDENTLY DETERMINE RESULT
                    </div>

                    <div class="result-description">

                        The tool could not confidently read enough
                        information from this ECG report to determine
                        whether it was reported as normal or abnormal.

                        <br><br>

                        Try uploading a clearer, higher-resolution
                        image or PDF.

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "If you are concerned about the ECG, have the original "
                "report reviewed by a healthcare professional."
            )


        # ====================================================
        # MEASUREMENT TABLE
        # ====================================================

        st.markdown(
            '<div class="section-title">3. Your ECG Numbers</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Here is what the tool could read from your report "
            "compared with the educational adult reference ranges."
        )


        rows = []

        units = {

            "Heart Rate": "bpm",

            "PR Interval": "ms",

            "QRS Duration": "ms",

            "QT": "ms",

            "QTc": "ms",

            "P Axis": "°",

            "QRS Axis": "°",

            "T Axis": "°",
        }


        order = [

            "Heart Rate",

            "PR Interval",

            "QRS Duration",

            "QT",

            "QTc",

            "P Axis",

            "QRS Axis",

            "T Axis",
        ]


        for field in order:

            value, conf, alternatives = (
                extracted.get(
                    field,
                    (
                        None,
                        0,
                        []
                    )
                )
            )

            lo, hi, unit = ref[field]


            # Missing
            if value is None:

                rows.append(
                    {
                        "Test": field,

                        "Your Report":
                            "Not confidently detected",

                        "Typical Adult Range":
                            f"{lo:g}–{hi:g} {unit}",

                        "Result":
                            "🟡 Need manual check",

                        "Explanation":
                            "The tool could not reliably read this value.",
                    }
                )

                continue


            status, status_text = classify(
                value,
                lo,
                hi
            )


            if conf < 0.75:

                result = (
                    "🟡 Low OCR confidence"
                )

                meaning = (
                    f"Possible readings: "
                    f"{', '.join(map(str, alternatives))}. "
                    f"Verify the original ECG."
                )

            elif status == "normal":

                result = (
                    "🟢 Within range"
                )

                meaning = simple_explanation(
                    field,
                    status
                )

            elif status == "low":

                result = (
                    "🔴 Below range"
                )

                meaning = simple_explanation(
                    field,
                    status
                )

            else:

                result = (
                    "🔴 Above range"
                )

                meaning = simple_explanation(
                    field,
                    status
                )


            rows.append(
                {
                    "Test": field,

                    "Your Report":
                        f"{value:g} {units[field]}",

                    "Typical Adult Range":
                        f"{lo:g}–{hi:g} {unit}",

                    "Result":
                        result,

                    "Explanation":
                        meaning,
                }
            )


        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # QUICK MEASUREMENT SUMMARY
        # ====================================================

        if confident > 0:

            st.markdown(
                "### 📌 Measurement Summary"
            )

            normal_count = 0

            abnormal_count = 0

            for field in order:

                value, conf, alternatives = (
                    extracted.get(
                        field,
                        (
                            None,
                            0,
                            []
                        )
                    )
                )

                if (
                    value is not None
                    and conf >= 0.75
                ):

                    lo, hi, unit = ref[field]

                    status, _ = classify(
                        value,
                        lo,
                        hi
                    )

                    if status == "normal":
                        normal_count += 1
                    else:
                        abnormal_count += 1


            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Measurements Read",
                    confident
                )

            with col2:

                st.metric(
                    "Within Range",
                    normal_count
                )

            with col3:

                st.metric(
                    "Outside Range",
                    abnormal_count
                )


        # ====================================================
        # REFERENCE NOTE
        # ====================================================

        st.info(
            "Reference note: QT is heart-rate dependent, so QTc is "
            "generally more useful than raw QT. Reference ranges are "
            "not universal and should not be used alone to diagnose "
            "a medical condition."
        )


        # ====================================================
        # MACHINE INTERPRETATION
        # ====================================================

        st.markdown(
            '<div class="section-title">4. What the ECG Machine Printed</div>',
            unsafe_allow_html=True
        )


        if interp:

            seen_plain = set()

            for line in interp:

                plain = interpretation_plain_language(
                    line
                )

                key = plain.lower()

                if key in seen_plain:
                    continue

                seen_plain.add(
                    key
                )


                is_abnormal = any(
                    keyword in line.lower()
                    for keyword in [

                        "st elevation",

                        "st depression",

                        "consider anterior injury",

                        "infarct",

                        "ischemia",

                        "ischemic",

                        "atrial fibrillation",

                        "atrial flutter",

                        "abnormal ecg",

                        "abnormal ekg",

                        "tachycardia",

                        "bradycardia",

                        "block",

                    ]
                )


                if is_abnormal:

                    st.error(
                        "⚠️ " + plain
                    )

                else:

                    st.info(
                        "• " + plain
                    )


                with st.expander(
                    "Show original machine/OCR wording"
                ):

                    st.write(
                        line
                    )


        else:

            st.info(
                "No supported machine interpretation was confidently detected."
            )


        # ====================================================
        # IF ABNORMAL — RECOMMEND PROFESSIONAL REVIEW
        # ====================================================

        if report_status == "abnormal":

            st.markdown(
                """
                <div class="doctor-box">

                    <h3>🩺 What should you do next?</h3>

                    <p>
                    Because the report contains a flagged finding or
                    a measurement outside the displayed reference range,
                    the original ECG should be reviewed by a healthcare
                    professional.
                    </p>

                    <p>
                    <strong>
                    Contact your doctor or an appropriate specialist
                    for proper interpretation and follow-up.
                    </strong>
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # WHAT TOOL CAN / CANNOT DO
        # ====================================================

        st.markdown(
            '<div class="section-title">5. What This Tool Can and Cannot Tell You</div>',
            unsafe_allow_html=True
        )


        col1, col2 = st.columns(2)


        with col1:

            st.success(
                "### ✅ This tool can"
            )

            st.write(
                "• Read printed ECG measurements"
            )

            st.write(
                "• Compare values with educational adult reference ranges"
            )

            st.write(
                "• Explain measurements in simple language"
            )

            st.write(
                "• Highlight machine-generated findings that deserve review"
            )

            st.write(
                "• Show whether the printed report appears normal, "
                "abnormal, or requires review"
            )


        with col2:

            st.error(
                "### ❌ This tool cannot"
            )

            st.write(
                "• Diagnose heart disease"
            )

            st.write(
                "• Confirm a heart attack"
            )

            st.write(
                "• Reliably diagnose an ECG from a photograph alone"
            )

            st.write(
                "• Replace a clinician's review of the 12-lead waveform"
            )

            st.write(
                "• Guarantee that a green result means the entire ECG is normal"
            )


        # ====================================================
        # RAW OCR
        # ====================================================

        with st.expander(
            "🔍 Show raw OCR text"
        ):

            st.text(
                "\n\n--- OCR PASS ---\n\n".join(
                    texts
                )
            )


        # ====================================================
        # EMERGENCY WARNING
        # ====================================================

        st.error(
            "🚨 If the person has severe or new chest pain, "
            "trouble breathing, fainting, severe weakness, or other "
            "serious symptoms, do not wait for this tool's result. "
            "Seek urgent medical care."
        )


# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        <strong>⚕️ Medical Disclaimer</strong>

        <br><br>

        ECG Report Reader is an educational screening prototype.
        It is not a medical device and does not diagnose or rule out
        heart disease, heart attack, arrhythmia, or any other medical
        condition.

        <br><br>

        The tool uses OCR to read information printed on an ECG report.
        OCR can make mistakes, and automated ECG interpretations can
        also be incorrect.

        <br><br>

        A result marked "Reported as Normal" means that no supported
        abnormal finding was detected in the information the tool could
        read. It does not mean that a healthcare professional has
        confirmed the ECG is normal.

        <br><br>

        Always consider the original ECG tracing, symptoms, medical
        history and professional clinical assessment.

    </div>
    """,
    unsafe_allow_html=True
)


st.caption(
    "ECG Report Reader v5 • Educational screening prototype • Not a medical device"
)