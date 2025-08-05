import pdfplumber
import statistics
import re
import json

pdf_path = "deepLearning.pdf"

def detect_titles_and_text(pdf_path, skip_first_pages=15):
    sections = []
    
    with pdfplumber.open(pdf_path) as pdf:
        font_sizes = []
        
        # Collecte des tailles de police en ignorant les premières pages
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num <= skip_first_pages:
                continue
            words = page.extract_words(extra_attrs=["size", "fontname", "top"])
            for w in words:
                try:
                    font_sizes.append(float(w["size"]))
                except:
                    continue
        
        if not font_sizes:
            return sections
        
        median_size = statistics.median(font_sizes)
        medium_threshold = median_size * 1.5
        large_threshold = medium_threshold * 1.5
        
        current_section = None
        current_subsection = None
        
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num <= skip_first_pages:
                continue
            
            words = page.extract_words(extra_attrs=["size", "fontname", "top"])
            lines = {}
            for w in words:
                y = round(w["top"], -1)
                if y not in lines:
                    lines[y] = []
                lines[y].append(w)
            
            for y, line_words in sorted(lines.items()):
                text = " ".join(w["text"] for w in line_words).strip()
                if not text or len(text) < 3:
                    continue
                
                avg_size = statistics.mean([float(w["size"]) for w in line_words])
                
                is_caps = text.isupper() or sum(c.isupper() for c in text)/len(text) > 0.3
                is_short = len(text.split()) <= 12
                is_clean = not text.endswith(".")
                has_no_punct = re.match(r"^[A-Za-z0-9\s,:;’'“”\-]+$", text) is not None
                
                if len(text.replace(" ", "")) < 3:
                    continue
                
                # Détection du type de ligne
                if avg_size >= large_threshold and (is_short or is_caps) and is_clean:
                    current_section = {
                        "title": text,
                        "page": page_num,
                        "subsections": [],
                        "content": []
                    }
                    sections.append(current_section)
                    current_subsection = None
                elif avg_size >= medium_threshold and (is_short or is_caps) and is_clean:
                    if current_section:
                        current_subsection = {
                            "subtitle": text,
                            "page": page_num,
                            "content": []
                        }
                        current_section["subsections"].append(current_subsection)
                else:
                    # Sinon, c'est du texte normal
                    if current_subsection:
                        current_subsection["content"].append(text)
                    elif current_section:
                        current_section["content"].append(text)
    
    return sections

# 🔎 Exécution
sections = detect_titles_and_text(pdf_path, skip_first_pages=15)

# Affichage formaté
for sec in sections:
    print(f"- {sec['title']} (p.{sec['page']})")
    for line in sec["content"]:
        print(f"    {line}")
    for sub in sec["subsections"]:
        print(f"    - {sub['subtitle']} (p.{sub['page']})")
        for line in sub["content"]:
            print(f"        {line}")

# Sauvegarde JSON
with open("pdf_sections.json", "w", encoding="utf-8") as f:
    json.dump(sections, f, indent=2, ensure_ascii=False)
