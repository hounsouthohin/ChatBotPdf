import json
import re

# Paramètres du chunking
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

def clean_text(text):
    """Nettoie le texte pour le rendre plus lisible et homogène."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([!?.,]){2,}", r"\1", text)
    text = re.sub(r"[^\x20-\x7E\u00A0-\u017F]+", " ", text)
    return text.strip()

def chunk_text(text, chunk_size=200, overlap=40):
    """Découpe un texte en chunks avec chevauchement."""
    words = text.split()
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start = end - overlap

def build_chunks(hierarchy, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Construit des chunks exploitables depuis la hiérarchie."""
    chunks = []
    chunk_id = 1
    
    for sec in hierarchy:
        section_text = clean_text(" ".join(sec.get("content", [])))
        
        if section_text:
            for piece in chunk_text(section_text, chunk_size, overlap):
                piece = clean_text(piece)  # nettoyage final
                chunks.append({
                    "chunk_id": chunk_id,
                    "title": sec["title"],
                    "subtitle": None,
                    "page": sec["page"],
                    "text": piece
                })
                chunk_id += 1
        
        for sub in sec.get("subsections", []):
            sub_text = clean_text(" ".join(sub.get("content", [])))
            if sub_text:
                for piece in chunk_text(sub_text, chunk_size, overlap):
                    piece = clean_text(piece)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "title": sec["title"],
                        "subtitle": sub["subtitle"],
                        "page": sub["page"],
                        "text": piece
                    })
                    chunk_id += 1
                    
    return chunks

# Exemple d’utilisation
with open("pdf_sections.json", "r", encoding="utf-8") as f:
    hierarchy = json.load(f)

chunks = build_chunks(hierarchy)

# Affichage formaté
for ch in chunks[:5]:
    print(f"Chunk {ch['chunk_id']} | {ch['title']} -> {ch['subtitle']} (p.{ch['page']})")
    print(f"Texte: {ch['text'][:120]}...")
    print("-"*80)

# Sauvegarde JSON
with open("pdf_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
