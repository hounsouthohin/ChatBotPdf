import pdfplumber
import statistics
import re
import json

pdf_path = "deepLearning.pdf"

def fix_spacing(text):
    """
    Corrige les espacements artificiels entre lettres dans les titres.
    Exemple : 'D e e p L e a r n i n g' -> 'Deep Learning'
    """
    # Supprimer les espaces entre lettres individuelles (ex: D e e p)
    text = re.sub(r"(?<=\w)\s(?=\w)", "", text)
    # Remplacer plusieurs espaces par un seul
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def detect_titles_from_pdf(pdf_path):
    titles = []
    
    with pdfplumber.open(pdf_path) as pdf:
        font_sizes = []
        
        # Étape 1 : Collecte de toutes les tailles de police du document
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
        medium_threshold = median_size * 1.5  # sous-titres
        large_threshold = medium_threshold * 1.5  # grands titres
        
        # Étape 2 : Détection page par page
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size", "fontname", "top"])
            
            # Grouper les mots par ligne (selon leur position verticale arrondie)
            lines = {}
            for w in words:
                y = round(w["top"], -1)  # arrondi pour grouper par ligne
                if y not in lines:
                    lines[y] = []
                lines[y].append(w)
            
            for y, line_words in lines.items():
                text = " ".join(w["text"] for w in line_words).strip()
                if not text or len(text) < 3:
                    continue

                # 🔧 Correction des espacements
                text = fix_spacing(text)

                # Calcul de la taille moyenne des mots de la ligne
                avg_size = statistics.mean([float(w["size"]) for w in line_words])
                
                # Heuristiques pour reconnaître un titre
                is_caps = text.isupper() or sum(c.isupper() for c in text)/len(text) > 0.6
                is_short = len(text.split()) <= 12
                is_clean = not text.endswith(".")  # éviter les phrases complètes
                has_no_punct = re.match(r"^[A-Za-z0-9\s,:;’'“”\-]+$", text) is not None
                
                # Détermination du niveau
                level = None
                if avg_size >= large_threshold and (is_short or is_caps) and is_clean:
                    level = 1  # grand titre
                elif avg_size >= medium_threshold and (is_short or is_caps) and is_clean:
                    level = 2  # sous-titre
                
                if level:
                    titles.append({
                        "title": text,
                        "page": page_num,
                        "level": level
                    })
    
    return titles


# 🔎 Test du pipeline
titles = detect_titles_from_pdf(pdf_path)

# Affichage formaté
for t in titles:
    print(f"{'  '*(t['level']-1)}- {t['title']} (p.{t['page']})")

# Sauvegarde JSON
with open("pdf_titles.json", "w", encoding="utf-8") as f:
    json.dump(titles, f, indent=2, ensure_ascii=False)
