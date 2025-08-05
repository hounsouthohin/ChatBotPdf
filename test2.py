import pdfplumber
import statistics
import re
import json

pdf_path = "deepLearning.pdf"

def detect_titles_from_pdf(pdf_path):
    titles = []
    with pdfplumber.open(pdf_path) as pdf:
        font_sizes = []
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size", "fontname", "top"])
            for w in words:
                try:
                    font_sizes.append(float(w["size"]))
                except:
                    continue
        if not font_sizes:
            return titles
        
        median_size = statistics.median(font_sizes)
        medium_threshold = median_size * 1.5
        large_threshold = medium_threshold * 1.5

        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size", "fontname", "top"])
            lines = {}
            for w in words:
                y = round(w["top"], -1)
                if y not in lines:
                    lines[y] = []
                lines[y].append(w)
            
            for y, line_words in lines.items():
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

                level = None
                if avg_size >= large_threshold and (is_short or is_caps) and is_clean:
                    level = 1
                elif avg_size >= medium_threshold and (is_short or is_caps) and is_clean:
                    level = 2

                if level:
                    titles.append({
                        "title": text,
                        "page": page_num,
                        "level": level,
                        "y": y  # position pour relier au contenu suivant
                    })
    return titles

def build_hierarchy_with_content(pdf_path, titles):
    """Construit la hiérarchie et regroupe textes + images"""
    hierarchy = []
    current_section = None
    current_subsection = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            words = page.extract_words(extra_attrs=["top"])
            
            # Récupération des images de la page
            images = []
            for img in page.images:
                images.append({
                    "x0": img["x0"],
                    "top": img["top"],
                    "x1": img["x1"],
                    "bottom": img["bottom"],
                    "page": page_num
                })

            for line in page_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                # Vérifier si cette ligne correspond à un titre/sous-titre
                match_title = next((t for t in titles if t["page"] == page_num and line.startswith(t["title"])), None)
                
                if match_title:
                    if match_title["level"] == 1:
                        current_section = {
                            "title": match_title["title"],
                            "page": page_num,
                            "content": [],
                            "images": [],
                            "subsections": []
                        }
                        hierarchy.append(current_section)
                        current_subsection = None
                    elif match_title["level"] == 2 and current_section:
                        current_subsection = {
                            "subtitle": match_title["title"],
                            "page": page_num,
                            "content": [],
                            "images": []
                        }
                        current_section["subsections"].append(current_subsection)
                else:
                    # Ajouter la ligne dans la section courante
                    if current_subsection:
                        current_subsection["content"].append(line)
                    elif current_section:
                        current_section["content"].append(line)
            
            # Attacher les images
            if current_subsection:
                current_subsection["images"].extend(images)
            elif current_section:
                current_section["images"].extend(images)
    
    return hierarchy

# 🔎 Exécution
titles = detect_titles_from_pdf(pdf_path)
hierarchy = build_hierarchy_with_content(pdf_path, titles)

# Affichage formaté
for section in hierarchy:
    print(f"- {section['title']} (p.{section['page']})")
    for sub in section["subsections"]:
        print(f"    - {sub['subtitle']} (p.{sub['page']}) [{len(sub['content'])} lignes, {len(sub['images'])} images]")

# Sauvegarde JSON
with open("pdf_full_hierarchy.json", "w", encoding="utf-8") as f:
    json.dump(hierarchy, f, indent=2, ensure_ascii=False)
