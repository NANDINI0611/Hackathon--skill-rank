from flask import Flask, request
import pdfplumber
import os
from transformers import pipeline

app = Flask(__name__)

# Hugging Face classifier (small model for faster download)
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

# Define required fields
REQUIRED_FIELDS = {
    "contract": ["party_1", "party_2", "signature", "date", "payment_terms"],
    "invoice": ["invoice_number", "amount", "due_date", "tax", "bill_to", "bill_from"],
    "report": []  # no strict fields for report
}

# Home Page
@app.route("/")
def index():
    return '''
    <h2>Upload PDF</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" />
        <input type="submit" />
    </form>
    '''

# Handle upload
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # Extract text
    with pdfplumber.open(filepath) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    # Hugging Face classification
    candidate_labels = ["contract", "invoice", "report"]
    result = classifier(text[:1000], candidate_labels)
    doc_type = result["labels"][0]
    confidence = float(result["scores"][0])

    # Check missing fields
    required = REQUIRED_FIELDS.get(doc_type.lower(), [])
    missing = [field for field in required if field.lower() not in text.lower()]

    return f"""
    <h3>Document Type:</h3>
    <pre>{doc_type} (confidence: {confidence:.2f})</pre>
    <br>
    <h3>Missing Fields:</h3>
    <pre>{missing if missing else "None 🎉"}</pre>
    <br>
    <h3>Extracted Text:</h3>
    <pre>{text}</pre>
    """

if __name__ == "__main__":
    app.run(debug=True)
























































