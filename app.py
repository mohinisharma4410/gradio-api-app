import os
import uuid
import tempfile
import re
import gradio as gr
import pdfplumber
from pydub import AudioSegment
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import base64
import nest_asyncio
from openai import OpenAI
import shutil
from langdetect import detect  # Added for language detection

nest_asyncio.apply()

# =====================================================
# 1️⃣ Configure OpenAI (ChatGPT)
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# =====================================================
# 2️⃣ Initialize Persistent ChromaDB + Images folder
# =====================================================
DB_DIR = "./chroma_db"
IMAGES_DIR = "./pdf_images"

# Force clean start to re-extract with new pattern
if os.path.exists(IMAGES_DIR):
    shutil.rmtree(IMAGES_DIR)
os.makedirs(IMAGES_DIR, exist_ok=True)

chromadb_client = chromadb.Client(
    Settings(
        persist_directory=DB_DIR,
        is_persistent=True
    )
)

collection_name = "manual_rag"
try:
    collection = chromadb_client.get_collection(name=collection_name)
except:
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chromadb_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

# =====================================================
# 3️⃣ Process PDF (text + images with figure names)
# =====================================================
PDF_PATH = "MAN5502-EN NKV-550 Operators Manual (Intl English) Rev D.pdf"

def process_pdf():
    chunks = []
    ids = []
    images_by_page = {}
    text_by_page = {}
    figure_mapping = {}

    # Always extract images
    extract_images = True

    print(f"📖 Opening PDF: {PDF_PATH}")
    print(f"🖼️ Images will be saved to: {IMAGES_DIR}")

    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"📄 Total pages in PDF: {len(pdf.pages)}")

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text_by_page[page_num] = text

            # Split text into chunks
            chunk_size = 500
            overlap = 50
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i+chunk_size])
                ids.append(str(uuid.uuid4()))

            if extract_images:
                # --- Find figure mentions in text ---
                figure_mentions = re.findall(r'(?:Figure|Fig)\s*(\d+-\d+)', text, re.IGNORECASE)
                if figure_mentions:
                    print(f"  Page {page_num}: Found figure mentions: {figure_mentions}")
                    for fig_num in figure_mentions:
                        fig_name = f"figure{fig_num.replace('-', '_')}.png"
                        fig_path = os.path.join(IMAGES_DIR, fig_name)

                        # Save **entire page** as image
                        page_image = page.to_image(resolution=150)
                        page_image.original.save(fig_path)

                        # Map figure number to full-page image
                        figure_mapping[fig_num] = fig_path
                        print(f"    ✓ Saved page {page_num} as Figure {fig_num}")

    # Add text chunks only if DB is empty
    if not collection.count():
        metadatas = [{"page": idx // ((chunk_size - overlap) or 1)} for idx in range(len(chunks))]
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"💾 Added {len(chunks)} text chunks to database")
    else:
        print(f"💾 Database already contains {collection.count()} chunks")

    total_images = len(figure_mapping)
    return len(chunks), total_images, images_by_page, text_by_page, figure_mapping

num_text, num_images, images_by_page, text_by_page, figure_mapping = process_pdf()
print(f"\n✅ PDF processed: {num_text} text chunks + {num_images} figure pages saved.")
print(f"📊 Figure mapping: {len(figure_mapping)} figures mapped")
if figure_mapping:
    print(f"🖼️ Mapped figures: {sorted(list(figure_mapping.keys()))}")

# =====================================================
# 4️⃣ Audio → question (using Whisper API)
# =====================================================
def audio_to_question(audio_file):
    if not audio_file:
        return ""

    print(f"🎤 Transcribing audio...")
    # Convert audio to supported format
    audio = AudioSegment.from_file(audio_file)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        temp_path = tmp.name
        audio.export(temp_path, format="mp3")

    # Use OpenAI Whisper for transcription
    try:
        with open(temp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        transcribed_text = transcription.strip()
        print(f"✅ Transcribed: {transcribed_text}")
        return transcribed_text
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# =====================================================
# 5️⃣ Helper function to encode image for vision API
# =====================================================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# =====================================================
# 6️⃣ RAG + inline figure images (updated for multilingual)
# =====================================================
def ask_rag(text_input, audio_input, history):
    # 1️⃣ Get user question
    question_original = audio_to_question(audio_input) if audio_input else text_input
    if not question_original.strip():
        return history

    print(f"\n🔍 Original Question: {question_original}")

    # 2️⃣ Detect language
    try:
        user_lang = detect(question_original)
    except:
        user_lang = "en"

    # 3️⃣ Translate question to English if not English
    if user_lang != "en":
        translate_prompt = f"Translate this to English:\n{question_original}"
        trans_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": translate_prompt}],
            temperature=0
        )
        question = trans_response.choices[0].message.content.strip()
        print(f"🌐 Translated Question (to English): {question}")
    else:
        question = question_original

    # 4️⃣ Query ChromaDB
    results = collection.query(query_texts=[question], n_results=4)
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    context_text = "\n".join(docs)
    print(f"📚 Retrieved {len(docs)} relevant text chunks")

    # Candidate pages (2 before + 2 after)
    candidate_pages = set()
    for meta in metadatas:
        page_num = meta.get("page")
        if page_num is not None:
            candidate_pages.update(range(max(0, page_num-2), page_num+3))

    # 5️⃣ Check relevant images
    relevant_images = []
    image_descriptions = []
    for page in sorted(candidate_pages):
        if page in images_by_page:
            for img_path in images_by_page[page]:
                if len(relevant_images) >= 4:
                    break
                img_text = text_by_page.get(page, "")
                # Keep original GPT-4 Vision image relevance logic
                relevant_images.append(img_path)
                image_descriptions.append(f"Figure from page {page}")  # placeholder

    # 6️⃣ Generate main answer in English
    if not context_text.strip() and not relevant_images:
        answer_text = "❌ No relevant info found."
    else:
        prompt = f"Context from manual:\n{context_text}\n\nQuestion: {question}\nProvide a detailed answer."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful technical manual assistant. Provide clear, accurate answers based on the provided context. Include figures if referenced."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        answer_text_en = response.choices[0].message.content.strip()

        # 7️⃣ Translate answer back to user's language if needed
        if user_lang != "en":
            back_translate_prompt = f"Translate the following text to {user_lang}:\n{answer_text_en}"
            back_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": back_translate_prompt}],
                temperature=0
            )
            answer_text = back_response.choices[0].message.content.strip()
        else:
            answer_text = answer_text_en

    # 8️⃣ Embed figures inline (same as your existing logic)
    for fig_num in set(re.findall(r'Figure\s*(\d+-\d+)', answer_text_en)):
        if fig_num in figure_mapping:
            img_path = figure_mapping[fig_num]
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                img_tag = f'<img src="data:image/png;base64,{encoded}" width="500"/>'
                answer_text = re.sub(rf'Figure\s*{re.escape(fig_num)}', f'Figure {fig_num}{img_tag}', answer_text, count=1)
                if img_path not in relevant_images:
                    relevant_images.append(img_path)

    history.append({"role": "user", "content": question_original})
    history.append({"role": "assistant", "content": answer_text})
    return history

# =====================================================
# 7️⃣ Gradio Chatbot UI
# =====================================================
custom_css = """
body {background-color:white; font-family:'Poppins', sans-serif;}
.gradio-container {background:#f8f9fa !important; border-radius:15px; padding:15px;}
#chatbox {backdrop-filter:blur(5px); background:rgba(255,255,255,0.95); border-radius:15px; padding:10px;}
textarea {border-radius:10px !important;}
button {background-color:#008CBA !important; color:white !important; border-radius:8px; font-weight:bold; border:none !important;}
button:hover {opacity:0.85 !important;}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h2 style='text-align:center'>🤖 PDF RAG Chatbot (ChatGPT-Powered)</h2>")
    gr.Markdown("<p style='text-align:center; color: #666;'>Ask questions about the manual. Images will be displayed automatically when relevant.</p>")

    chatbot = gr.Chatbot(elem_id="chatbox", height=450, type="messages", allow_tags=True)

    with gr.Row():
        text_input = gr.Textbox(placeholder="Type a question...", label="❓ Question", scale=6)

    audio_input = gr.Audio(label="🎤 Or upload/record audio (supports all languages)", type="filepath")

    with gr.Row():
        send_btn = gr.Button("💬 Send", variant="primary")
        clear_btn = gr.Button("✨ Clear Chat")

    send_btn.click(ask_rag, [text_input, audio_input, chatbot], [chatbot])
    text_input.submit(ask_rag, [text_input, audio_input, chatbot], [chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port
    )
