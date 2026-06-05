from flask import Flask, render_template, request, jsonify
from langdetect import detect
import os
import cv2
import torch
import re
import unicodedata
import time
import numpy as np
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from sacrebleu import corpus_bleu
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# =============================
# ENV FIX
# =============================
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# =============================
# FLASK SETUP
# =============================
app = Flask(__name__, template_folder="templates")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================
# OCR SETUP
# =============================
ocr = PaddleOCR(
    lang="latin",
    use_angle_cls=True,
    use_gpu=False,
    det_model_dir="C:/Users/acer/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer",
    rec_model_dir="C:/Users/acer/.paddleocr/whl/rec/latin/latin_PP-OCRv3_rec_infer",
    cls_model_dir="C:/Users/acer/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer"
)

# =============================
# NLLB MODEL SETUP
# =============================
MODEL_PATH = "./models/nllb"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =============================
# LANGUAGES
# =============================
LANG_NAME_TO_CODE = {
    "English": "eng_Latn", "French": "fra_Latn", "German": "deu_Latn",
    "Spanish": "spa_Latn", "Italian": "ita_Latn", "Dutch": "nld_Latn",
    "Romanian": "ron_Latn",  "Chinese Traditional": "zho_Hant",
    "Japanese": "jpn_Jpan", "Korean": "kor_Hang", "Thai": "tha_Thai",
    "Vietnamese": "vie_Latn", "Indonesian": "ind_Latn", "Malay": "zsm_Latn",
    "Tamil": "tam_Taml", "Hindi": "hin_Deva", "Telugu": "tel_Telu",
    "Malayalam": "mal_Mlym", "Kannada": "kan_Knda", "Bengali": "ben_Beng",
    "Marathi": "mar_Deva", "Gujarati": "guj_Gujr", "Punjabi": "pan_Guru",
    "Odia": "ory_Orya",  
    "Persian": "pes_Arab", "Turkish": "tur_Latn", "Polish": "pol_Latn",
    "Russian": "rus_Cyrl", "Ukrainian": "ukr_Cyrl", "Czech": "ces_Latn",
     "Hungarian": "hun_Latn", "Greek": "ell_Grek",
    "Swahili": "swh_Latn",  "Filipino": "fil_Latn",
    "Latin": "lat_Latn"
}

INDIC_LANGS = {
    "tam_Taml": sanscript.TAMIL, "mal_Mlym": sanscript.MALAYALAM,
    "hin_Deva": sanscript.DEVANAGARI, "kan_Knda": sanscript.KANNADA,
    "tel_Telu": sanscript.TELUGU, "ben_Beng": sanscript.BENGALI,
    "mar_Deva": sanscript.DEVANAGARI, "guj_Gujr": sanscript.GUJARATI,
    "pan_Guru": sanscript.GURMUKHI, "ory_Orya": sanscript.ORIYA
}
def detect_language_code(text):
    try:
        lang = detect(text)
    except:
        return "eng_Latn"  # fallback

    mapping = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "ta": "tam_Taml",
        "hi": "hin_Deva"
    }

    return mapping.get(lang, "eng_Latn")
# =============================
# UTILITIES
# =============================
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Light denoise (safe)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray

def extract_text_from_image(path):
    img = cv2.imread(path)
    if img is None:
        return "", 0.0

    img = preprocess_image(img)

    # UPDATED LINE
    result = ocr.ocr(img)

    words, confidences = [], []

    if result:
        for line in result:
            if line:
                for word in line:
                    txt = word[1][0]
                    conf = word[1][1]
                    if conf > 0.3:
                        words.append(txt)
                        confidences.append(conf)
    print("RAW OCR:", words)
    text = clean_text(" ".join(words))
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return text, avg_conf

def detect_real_names(text):
    words = re.findall(r"[A-Za-z]+", text)
    return list(set([w for w in words if len(w) > 2 and w[0].isupper()]))

def nllb_translate(text, tgt_code):
    src_lang = detect_language_code(text)
    tokenizer.src_lang = src_lang
    sentences = split_sentences(text)
    translated_sentences = []
    for sent in sentences:
        sent = f"{src_lang} {sent}"
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=256).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
            num_beams=8,
            
            max_length=256,
           
        )
        translated_sentences.append(clean_text(tokenizer.decode(outputs[0], skip_special_tokens=True)))
    return " ".join(translated_sentences)

def smart_translate(text, target_langs):
    results = {}
    names = detect_real_names(text)
    for tgt in target_langs:
        tgt_code = LANG_NAME_TO_CODE[tgt]
        translated = nllb_translate(text, tgt_code)
        if tgt_code in INDIC_LANGS:
            for name in names:
                translit = transliterate(name, sanscript.ITRANS, INDIC_LANGS[tgt_code])
                translated = translated.replace(name, translit)
        results[tgt] = translated
    return results

def compute_translation_bleu(pred, reference):
    if not reference:
        return None
    return corpus_bleu([pred], [[reference]]).score
def text_input_accuracy(pred):
    words = pred.split()
    if not words:
        return 0
    unique_words = len(set(words))
    return round((unique_words / len(words)) * 100, 2)
# =============================
# ROUTES
# =============================
@app.route("/")
def home():
    return render_template("index.html", languages=sorted(LANG_NAME_TO_CODE.keys()))

@app.route("/translate", methods=["POST"])
def translate_api():
    start_time = time.time()
    text = request.form.get("text", "").strip()
    ocr_confidence = None
    is_image_input = False

    # IMAGE INPUT
    if "image" in request.files:
        image = request.files["image"]
        if image.filename:
            is_image_input = True  
            path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(path)
            text, ocr_confidence = extract_text_from_image(path)

    if not text:
        return jsonify({"error": "No text found"})
    print("Detected Language:", detect_language_code(text))
    targets = request.form.getlist("languages[]")
    reference_texts = request.form.getlist("references[]")  # optional references

    # TRANSLATION
    translation_start = time.time()
    translations = smart_translate(text, targets)
    translation_time = time.time() - translation_start
    total_time = time.time() - start_time

    # BLEU SCORE
    translation_bleu = {}
    text_accuracy = {}
    for i, tgt in enumerate(targets):
        pred = translations[tgt]
        ref = reference_texts[i] if i < len(reference_texts) else None
        if ref:
            bleu_score = round(compute_translation_bleu(pred, ref), 2)
            translation_bleu[tgt] = bleu_score
            print(f"{tgt} BLEU Score: {bleu_score}")
        else:
            translation_bleu[tgt] = None

        # TEXT INPUT ACCURACY
        
           
    # RETURN JSON
    return jsonify({
        "translations": translations,
        "metrics": {
            "ocr_confidence": round(ocr_confidence, 2) if ocr_confidence else None,
            "translation_time_sec": round(translation_time, 2),
            "total_processing_time_sec": round(total_time, 2),
            "translation_accuracy": translation_bleu
            
        }
    })

if __name__ == "__main__":
    app.run(debug=True)

