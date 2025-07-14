from flask import Flask, render_template, request, redirect, url_for, send_file
from dotenv import load_dotenv
import os
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
import s3fs
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from io import BytesIO
import json
from ast import literal_eval
from thefuzz import fuzz, process as fuzz_process
import numpy as np
import tempfile
import uuid

# ----------------------------------------
# Flask app setup
# ----------------------------------------
app = Flask(__name__)
load_dotenv()

# ----------------------------------------
# S3 connection setup
# ----------------------------------------
S3_KEY = os.getenv('S3_KEY')
S3_SECRET = os.getenv('S3_SECRET')
S3_BUCKET = os.getenv('S3_BUCKET_NAME')

s3 = s3fs.S3FileSystem(
    anon=False,
    key=S3_KEY,
    secret=S3_SECRET
)

BASE_PATH = f's3://{S3_BUCKET}'
DATA_PATH = f"{BASE_PATH}/Data"
DF_PATH = f"{BASE_PATH}/doc_df.csv"

# ----------------------------------------
# Routes
# ----------------------------------------

@app.route('/')
def home():
    return redirect(url_for('upload'))

# ---------- Upload ----------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        pdf_file = request.files['pdf']
        file_type = request.form['file_type']
        notes = request.form['notes']

        pdf_reader = PdfReader(pdf_file)
        for i, page in enumerate(pdf_reader.pages):
            output = PdfWriter()
            output.add_page(page)
            save_page_path = f"{DATA_PATH}/{pdf_file.filename}_{i}.pdf"

            if not s3.exists(DATA_PATH):
                s3.mkdirs(DATA_PATH)

            with s3.open(save_page_path, 'wb') as f:
                output.write(f)

            temp_df = pd.DataFrame({
                'file_name': [pdf_file.filename],
                'page_number': [i],
                'file_path': [save_page_path],
                'file_type': [file_type],
                'notes': [notes],
                'upload_time': [datetime.now()],
                'words': [[]],
                'OCR_attempted': [False],
            })

            if s3.exists(DF_PATH):
                with s3.open(DF_PATH, 'rb') as f:
                    df = pd.read_csv(f)
                df = pd.concat([df, temp_df], ignore_index=True)
            else:
                df = temp_df

            with s3.open(DF_PATH, 'wb') as f:
                df.to_csv(f, index=False)

        return redirect(url_for('upload_success'))
    return render_template('upload.html')

@app.route('/upload_success')
def upload_success():
    return "✅ Upload successful! <a href='/upload'>Upload another</a> | <a href='/process'>Go to Process</a>"

# ---------- Process ----------
@app.route('/process', methods=['GET', 'POST'])
def process():
    if not s3.exists(DF_PATH):
        return "⚠️ No uploaded files found. Upload something first! <a href='/upload'>Upload</a>"

    with s3.open(DF_PATH, 'rb') as f:
        df = pd.read_csv(f)

    to_process = df[df['OCR_attempted'] == False]

    if request.method == 'POST' and not to_process.empty:
        model = ocr_predictor(pretrained=True, detect_orientation=True)

        for idx, row in to_process.iterrows():
            file_path = row['file_path']
            with s3.open(file_path, 'rb') as f:
                pdf_data = BytesIO(f.read())

            pdf_doc = DocumentFile.from_pdf(pdf_data)
            result = model(pdf_doc)
            json_output = result.export()

            bag_of_words = []
            for page in json_output['pages']:
                for block in page['blocks']:
                    for line in block['lines']:
                        bag_of_words.extend([w['value'] for w in line['words']])

            df.at[idx, 'words'] = json.dumps(bag_of_words)
            df.at[idx, 'OCR_attempted'] = True

        with s3.open(DF_PATH, 'wb') as f:
            df.to_csv(f, index=False)

        return redirect(url_for('process'))

    return render_template('process.html', to_process=to_process)

# ---------- Search ----------
@app.route('/search', methods=['GET', 'POST'])
def search():
    results = []
    selected_file = None
    temp_local_name = None

    if not s3.exists(DF_PATH):
        return "⚠️ No documents found. Upload & process something first! <a href='/upload'>Upload</a>"

    with s3.open(DF_PATH, 'rb') as f:
        df = pd.read_csv(f)

    if request.method == 'POST':
        if 'search_term' in request.form:
            term = request.form['search_term']
            if term.strip():
                search_scores = []
                for _, row in df.iterrows():
                    words = literal_eval(row['words']) if isinstance(row['words'], str) else []
                    match_value = fuzz.partial_token_set_ratio(' '.join(words), term)
                    search_scores.append(match_value)

                top_indices = np.argsort(search_scores)[-5:][::-1]
                for idx in top_indices:
                    word_list = literal_eval(df.iloc[idx]['words'])
                    best_matches = fuzz_process.extract(term, word_list, limit=5)
                    results.append({
                        'file_name': os.path.basename(df.iloc[idx]['file_path']),
                        'file_type': df.iloc[idx]['file_type'],
                        'notes': df.iloc[idx]['notes'],
                        'best_matches': best_matches
                    })

        elif 'filename' in request.form:
            selected_file = request.form['filename']
            pdf_path = f"{DATA_PATH}/{selected_file}"
            if s3.exists(pdf_path):
                with s3.open(pdf_path, 'rb') as f:
                    pdf_data = f.read()

                temp_filename = f"ocr_{uuid.uuid4().hex}.pdf"
                temp_file_path = os.path.join(tempfile.gettempdir(), temp_filename)
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(pdf_data)

                temp_local_name = temp_filename
            else:
                selected_file = None

    return render_template('search.html',
                           results=results,
                           selected_file=selected_file,
                           temp_local_name=temp_local_name)

# ---------- Preview ----------
@app.route('/preview/<filename>')
def preview(filename):
    temp_file_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(temp_file_path):
        return send_file(temp_file_path, mimetype='application/pdf')
    else:
        return "❌ File not found."

# ----------------------------------------
# Run the app
# ----------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
