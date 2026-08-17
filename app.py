import re
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import fitz
except ImportError:
    fitz = None

st.set_page_config(page_title="ECG Report Reader", page_icon="❤️", layout="wide")

# ---------- Reference ranges ----------
# Adult screening references. These are intentionally presented as reference ranges,
# not diagnostic cutoffs for every patient.
def ranges(sex):
    qtclow, qtchi = (350, 450) if sex == "Male" else (360, 460) if sex == "Female" else (350, 460)
    return {
        "Heart Rate": (60, 100, "bpm"),
        "PR Interval": (120, 200, "ms"),
        "QRS Duration": (70, 110, "ms"),
        "QT": (350, 450, "ms"),  # rough educational reference; QT is rate-dependent
        "QTc": (qtclow, qtchi, "ms"),
        "QRS Axis": (-30, 90, "°"),
        "P Axis": (0, 75, "°"),
        "T Axis": (15, 75, "°"),
    }

def pdf_images(data):
    if fitz is None:
        return []
    doc = fitz.open(stream=data, filetype="pdf")
    result = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        result.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return result

def preprocess(img):
    gray = ImageOps.autocontrast(img.convert("L"))
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)
    return gray

def ocr_passes(img):
    if pytesseract is None:
        return []
    # OCR is deliberately run on several small rotations to handle photographed ECG paper.
    base = preprocess(img).resize((img.width * 4, img.height * 4))
    outputs = []
    for angle in (-2, -1, 0, 1, 2):
        rotated = base.rotate(angle, expand=True, fillcolor="white")
        threshold = rotated.point(lambda p: 255 if p > 165 else 0)
        for variant in (rotated, threshold):
            for psm in (6, 11):
                outputs.append(pytesseract.image_to_string(variant, config=f"--psm {psm}"))
    return outputs

def clean_line(s):
    return re.sub(r"\s+", " ", s).strip()

def labeled_candidates(texts):
    patterns = {
        "Heart Rate": [r"\brate\b", r"\bv[- ]?rate\b", r"\bvent(?:ricular)?\s*rate\b"],
        "PR Interval": [r"\bpr\b", r"\bp[- ]?r\b"],
        "QRS Duration": [r"\bqrsd\b", r"\bqrs[- ]?d\b", r"\bqrs\b"],
        "QTc": [r"\bqtc\b", r"\bqt[- ]?c\b"],
        "QT": [r"\bqt\b(?!c)"],
    }
    bounds = {
        "Heart Rate": (25, 220),
        "PR Interval": (60, 500),
        "QRS Duration": (40, 250),
        "QT": (150, 700),
        "QTc": (250, 700),
    }
    out = {k: [] for k in patterns}

    for text in texts:
        for raw in text.splitlines():
            line = clean_line(raw)
            if len(line) > 180:
                line = line[:180]
            for field, pats in patterns.items():
                for pat in pats:
                    m = re.search(pat + r"\s*[:=]?\s*([+-]?\d{1,4}(?:\.\d+)?)\b", line, re.I)
                    if not m:
                        continue
                    try:
                        value = int(round(float(m.group(1))))
                    except ValueError:
                        continue
                    lo, hi = bounds[field]
                    if lo <= value <= hi:
                        out[field].append(value)
    return out

def axis_candidates(texts):
    out = {"P Axis": [], "QRS Axis": [], "T Axis": []}
    for text in texts:
        t = re.sub(r"\s+", " ", text)
        blocks = re.findall(
            r"AXIS.*?(?=12\s*lead|standard placement|abnormal ecg|unconfirmed diagnosis|$)",
            t, re.I | re.S
        )
        for block in blocks:
            for field, label in [("P Axis", "P"), ("QRS Axis", "QRS"), ("T Axis", "T")]:
                m = re.search(r"\b" + label + r"\s*[:=]?\s*([+-]?\d{1,3})\b", block, re.I)
                if m:
                    v = int(m.group(1))
                    if -180 <= v <= 180:
                        out[field].append(v)
    return out

def consensus(values):
    if not values:
        return None, 0.0, []
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=counts.get)
    total = len(values)
    support = counts[best]
    # Confidence is intentionally conservative.
    conf = min(0.98, 0.45 + 0.10 * support)
    if len(counts) > 1:
        conf = min(conf, 0.72)
    alternatives = sorted(counts, key=counts.get, reverse=True)[:5]
    return best, conf, alternatives

def machine_interpretation(texts):
    keywords = [
        "sinus rhythm", "abnormal r-wave progression", "early transition",
        "st elevation", "st depression", "consider anterior injury",
        "infarct", "ischemia", "ischemic", "atrial fibrillation",
        "atrial flutter", "tachycardia", "bradycardia", "block",
        "abnormal ecg", "unconfirmed diagnosis", "normal p axis",
        "normal qrs axis", "qrs area"
    ]
    found = []
    for text in texts:
        for raw in text.splitlines():
            line = clean_line(raw)
            low = line.lower()
            if len(line) >= 5 and any(k in low for k in keywords):
                if line not in found:
                    found.append(line)
    return found

def classify(value, lo, hi):
    if value is None:
        return "missing", "Not confidently detected"
    if lo <= value <= hi:
        return "normal", f"Within reference range ({lo:g}–{hi:g})"
    if value < lo:
        return "low", f"Below reference range ({lo:g}–{hi:g})"
    return "high", f"Above reference range ({lo:g}–{hi:g})"

def simple_explanation(field, status):
    if status == "normal":
        return {
            "Heart Rate": "The heart rate is in the usual adult resting range.",
            "PR Interval": "The electrical signal is taking a typical amount of time to travel from the atria toward the ventricles.",
            "QRS Duration": "The ventricular electrical signal is not prolonged by this measurement.",
            "QT": "The measured QT interval is within the broad educational reference shown here. QT depends on heart rate.",
            "QTc": "The corrected QT interval is within the selected adult reference range.",
            "QRS Axis": "The main ventricular electrical direction is within the usual adult axis range.",
            "P Axis": "The P-wave axis is within the educational reference used by this prototype.",
            "T Axis": "The T-wave axis is within the educational reference used by this prototype.",
        }.get(field, "Within the reference range.")
    if status == "low":
        return "This value is below the reference range. It should be checked against the original ECG and clinical context."
    if status == "high":
        return "This value is above the reference range. It should be checked against the original ECG and clinical context."
    return "The tool could not confidently read this value."

def interpretation_plain_language(line):
    l = line.lower()
    if "sinus rhythm" in l:
        return "The ECG machine reports a sinus rhythm, meaning it says the heartbeat is being initiated from the heart's usual pacemaker."
    if "st elevation" in l or "consider anterior injury" in l:
        return "The ECG machine is reporting ST-segment elevation and/or a possible anterior injury pattern. This cannot be confirmed from OCR alone and needs clinical review."
    if "abnormal r-wave progression" in l:
        return "The machine reports an unusual R-wave progression across the chest leads. This can have several causes and should be interpreted from the actual ECG tracing."
    if "early transition" in l:
        return "The machine reports an early change in the R-wave pattern across the chest leads. This finding needs the actual tracing and clinical context for interpretation."
    if "abnormal ecg" in l:
        return "The ECG machine has labelled the tracing as abnormal. Automated ECG labels are not a diagnosis and should be reviewed by a clinician."
    if "unconfirmed diagnosis" in l:
        return "The printed diagnosis is marked as unconfirmed. It should not be treated as a confirmed diagnosis."
    if "tachycardia" in l:
        return "The machine is reporting a fast rhythm."
    if "bradycardia" in l:
        return "The machine is reporting a slow rhythm."
    if "atrial fibrillation" in l:
        return "The machine is reporting a possible irregular atrial rhythm called atrial fibrillation; this requires clinical confirmation."
    if "block" in l:
        return "The machine is reporting a possible conduction block; the actual ECG should be reviewed."
    if "ischemia" in l or "ischemic" in l or "infarct" in l:
        return "The machine is reporting a possible ischemic/injury-related finding. This requires clinician review."
    return "The machine-generated interpretation contains this finding; review the original ECG."

st.title("❤️ ECG Report Reader")
st.write("Upload a photo/PDF and get a simple, human-readable comparison of the printed measurements.")

st.warning(
    "Important: This is an educational screening tool, not a diagnostic device. "
    "A result can look 'normal' on measurements while the ECG tracing still has an important abnormality. "
    "A qualified clinician must confirm the ECG."
)

with st.sidebar:
    st.header("Patient information")
    sex = st.selectbox("Sex for QTc reference", ["Not specified", "Male", "Female"])
    st.caption("Adult reference ranges are used. Pediatric ECGs require different ranges.")
    st.divider()
    st.caption("Reference ranges are educational adult ranges and can vary with age, sex, method, device and clinical context.")

uploaded = st.file_uploader("Upload ECG report photo or PDF", type=["png","jpg","jpeg","webp","pdf"])

if uploaded:
    data = uploaded.read()
    images = pdf_images(data) if uploaded.name.lower().endswith(".pdf") else [Image.open(uploaded)]
    if not images:
        st.error("Could not read the file.")
        st.stop()

    st.subheader("1. Your ECG report")
    cols = st.columns(min(3, len(images)))
    for i, im in enumerate(images):
        with cols[i % len(cols)]:
            st.image(im, caption=f"Page {i+1}", use_container_width=True)

    if st.button("🔎 Analyze ECG", type="primary"):
        with st.spinner("Reading the printed ECG measurements and interpretation..."):
            texts = []
            for im in images:
                texts.extend(ocr_passes(im))

        raw = labeled_candidates(texts)
        raw.update(axis_candidates(texts))
        extracted = {field: consensus(vals) for field, vals in raw.items()}
        interp = machine_interpretation(texts)
        ref = ranges(sex if sex != "Not specified" else "Unknown")

        # ----- Summary -----
        st.subheader("2. Simple result")
        confident = 0
        abnormal_measurements = []
        uncertain = []
        for field, (value, conf, alts) in extracted.items():
            if value is not None and conf >= 0.75:
                confident += 1
                status, _ = classify(value, *ref[field][:2])
                if status != "normal":
                    abnormal_measurements.append(field)
            elif value is not None:
                uncertain.append(field)

        serious_printed = any(
            any(k in x.lower() for k in [
                "st elevation", "consider anterior injury", "infarct",
                "ischemia", "ischemic", "atrial fibrillation", "abnormal ecg"
            ]) for x in interp
        )

        if serious_printed:
            st.error("⚠️ REVIEW NEEDED: The printed ECG interpretation contains a potentially important finding.")
            st.write("This does **not** mean the person has a specific disease. It means the machine-generated finding should be reviewed by a qualified clinician.")
        elif abnormal_measurements:
            st.warning("⚠️ Some measured values are outside the reference ranges shown below.")
        elif uncertain:
            st.warning("🟡 Some measurements could not be read with enough confidence.")
        elif confident:
            st.success("🟢 All confidently extracted measurements are within the adult reference ranges used by this prototype.")
        else:
            st.info("No measurements were confidently extracted.")

        st.caption(
            "This summary only describes the measurements the tool could read. "
            "It is not a statement that the entire ECG is normal."
        )

        # ----- Comparison table -----
        st.subheader("3. What does each number mean?")
        rows = []
        units = {"Heart Rate":"bpm","PR Interval":"ms","QRS Duration":"ms","QT":"ms","QTc":"ms",
                 "P Axis":"°","QRS Axis":"°","T Axis":"°"}

        order = ["Heart Rate","PR Interval","QRS Duration","QT","QTc","P Axis","QRS Axis","T Axis"]
        for field in order:
            value, conf, alts = extracted.get(field, (None,0,[]))
            lo, hi, unit = ref[field]
            if value is None:
                rows.append({
                    "Test": field,
                    "Your report": "Not confidently detected",
                    "Typical adult range": f"{lo:g}–{hi:g} {unit}",
                    "Result": "🟡 Need manual check",
                    "What it means": "The photo/OCR did not provide a reliable value."
                })
                continue

            status, status_text = classify(value, lo, hi)
            if conf < 0.75:
                result = "🟡 Low OCR confidence"
            elif status == "normal":
                result = "🟢 Within range"
            elif status == "low":
                result = "🔴 Below range"
            else:
                result = "🔴 Above range"

            if conf < 0.75:
                meaning = f"Possible readings: {', '.join(map(str, alts))}. Verify the original ECG."
            else:
                meaning = simple_explanation(field, status)

            rows.append({
                "Test": field,
                "Your report": f"{value:g} {units[field]}",
                "Typical adult range": f"{lo:g}–{hi:g} {unit}",
                "Result": result,
                "What it means": meaning
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.info(
            "Reference note: QT is heart-rate dependent, so QTc is generally more useful than raw QT. "
            "Reference ranges are not universal and should not be used alone to diagnose a condition."
        )

        # ----- Printed interpretation -----
        st.subheader("4. What the ECG machine printed")
        if interp:
            seen_plain = set()
            for line in interp:
                plain = interpretation_plain_language(line)
                key = plain.lower()
                if key in seen_plain:
                    continue
                seen_plain.add(key)

                if any(k in line.lower() for k in [
                    "st elevation","consider anterior injury","infarct","ischemia",
                    "atrial fibrillation","abnormal ecg"
                ]):
                    st.error("⚠️ " + plain)
                else:
                    st.info("• " + plain)

                with st.expander("Show machine/OCR wording"):
                    st.write(line)
        else:
            st.success("No supported abnormal phrase was confidently detected in the printed interpretation.")

        # ----- Important limitation -----
        st.subheader("5. What this tool can and cannot tell you")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ✅ It can")
            st.write("- Read printed ECG measurements")
            st.write("- Compare them with common adult reference ranges")
            st.write("- Explain the numbers in simple language")
            st.write("- Highlight machine-generated findings that deserve review")
        with c2:
            st.markdown("### ❌ It cannot")
            st.write("- Confirm a heart attack or other diagnosis")
            st.write("- Reliably diagnose an ECG from a photograph alone")
            st.write("- Replace a clinician's review of the 12-lead waveform")
            st.write("- Guarantee that a 'green' measurement table means the ECG is normal")

        with st.expander("Show raw OCR text"):
            st.text("\n\n--- OCR PASS ---\n\n".join(texts))

        st.error(
            "If the person has severe or new chest pain, trouble breathing, fainting, or other serious symptoms, "
            "do not wait for this tool's result—seek urgent medical care."
        )

st.divider()
st.caption("ECG Report Reader v4 • Educational screening prototype • Not a medical device")
