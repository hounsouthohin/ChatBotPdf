import pdfplumber
import os

# === Paramètres ===
pdf_path = "ton_document.pdf"
output_dir = "pdf_extraction_output"
os.makedirs(output_dir, exist_ok=True)

results = []  # stockage structuré

def detect_section(page):
    """Détection simple des sections/titres avec pdfplumber"""
    words = page.extract_words()
    if not words:
        return None
    
    # Heuristique : mots en haut de page, taille > 12
    candidates = [w for w in words if float(w.get('size', 0)) > 12 and w['top'] < 120]
    if candidates:
        return " ".join([w['text'] for w in candidates])
    
    # Fallback par mot-clé
    text = " ".join([w['text'] for w in words])
    for keyword in ["Introduction", "Chapitre", "Section", "Conclusion"]:
        if keyword.lower() in text.lower():
            return keyword
    
    return None


with pdfplumber.open(pdf_path) as pdf:
    total_pages = len(pdf.pages)
    print(f"📄 Nombre total de pages détectées : {total_pages}")

    for page_num, page in enumerate(pdf.pages, start=1):
        print(f"\n➡️ Traitement page {page_num}/{total_pages} ...")
        
        # Essayer d'extraire du texte
        text = page.extract_text()
        section_title = detect_section(page)

        if text and text.strip():
            # Sauvegarder aussi une image si la page contient des figures
            image_path = os.path.join(output_dir, f"page_{page_num}.png")
            page.to_image(resolution=150).save(image_path, format="PNG")

            results.append({
                "page": page_num,
                "type": "texte+image",
                "section": section_title,
                "content": text.strip(),
                "image": image_path
            })
            print(f"✅ Texte extrait ({len(text.split())} mots), image sauvegardée.")
        
        else:
            # Page vide → ignorer
            print("⚠️ Page vide ignorée.")
            continue

# === Résumé final ===
nb_text = sum(1 for r in results if r["type"].startswith("texte"))
print("\n=== Résumé Étape 1 ===")
print(f"Pages avec texte : {nb_text}")
print(f"Pages totalement vides ignorées : {total_pages - nb_text}")

print("\nExemple de données extraites :")
for r in results[:3]:  # afficher 3 exemples
    print(r)
