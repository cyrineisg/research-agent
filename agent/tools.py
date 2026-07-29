import arxiv
import os
import re
import fitz
import urllib.request
import tempfile
from langchain.tools import tool
from dotenv import load_dotenv
import requests
import tempfile
import fitz
from agent.memory import save_paper, search_memory, get_memory_count
load_dotenv()
from datetime import datetime

@tool
def search_arxiv(query: str) -> str:
    """
    Cherche des papers scientifiques sur arXiv.
    Input : un sujet ou une question en anglais.
    Output : liste des 5 papers les plus pertinents.
    """
    client = arxiv.Client(
        num_retries=2,
        delay_seconds=2      # ← limite les retries
    )
    search = arxiv.Search(
        query=query,
        max_results=3,       # ← réduit de 5 à 3 pour aller plus vite
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    for paper in client.results(search):
        results.append({
            "titre": paper.title,
            "auteurs": [a.name for a in paper.authors[:2]],
            "resume": paper.summary[:300],  # ← réduit pour aller plus vite
            "pdf_url": paper.pdf_url,
            "date": str(paper.published.date())
        })

    if not results:
        return "Aucun paper trouvé."

    output = f"📚 {len(results)} papers trouvés :\n\n"
    for i, p in enumerate(results, 1):
        output += f"{i}. {p['titre']} ({p['date']})\n"
        output += f"   Auteurs : {', '.join(p['auteurs'])}\n"
        output += f"   Résumé : {p['resume']}...\n"
        output += f"   PDF : {p['pdf_url']}\n\n"

    return output

@tool
def read_paper_pdf(pdf_url: str) -> str:
    """
    Télécharge et extrait le texte d'un paper PDF depuis une URL arXiv.
    Input : URL du PDF
    Output : texte extrait du paper
    """
    try:
        # Timeout strict de 10 secondes
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        total_pages = len(doc)

        pages_to_read = [0, 1]
        if total_pages > 3:
            pages_to_read.append(total_pages - 1)

        extracted = ""
        for page_num in pages_to_read:
            extracted += doc[page_num].get_text()

        doc.close()

        # Sauvegarde en mémoire
        try:
            from agent.memory import save_paper
            title = extracted[:100]
            save_paper(title=title, content=extracted, pdf_url=pdf_url)
        except:
            pass  # la mémoire est optionnelle, pas bloquante
        
        return f"📄 Contenu extrait :\n\n{extracted[:2000]}"

    except requests.Timeout:
        return "❌ Timeout : PDF trop long à télécharger, essaie un autre paper."
    except Exception as e:
        return f"❌ Erreur : {str(e)}"
    

@tool
def check_memory(query: str) -> str:
    """
    Vérifie si des papers liés à ce sujet sont déjà en mémoire.
    À appeler EN PREMIER avant search_arxiv.
    Input : sujet ou titre du paper
    Output : papers déjà connus ou message vide
    """
    count = get_memory_count()
    if count == 0:
        return "Aucun paper en mémoire pour l'instant."

    result = search_memory(query)
    if result:
        return f"✅ Trouvé en mémoire ({count} papers stockés) :\n\n{result}"
    return f"Rien en mémoire sur ce sujet ({count} papers stockés sur d'autres sujets)."

@tool
def export_summary(content: str, filename: str = "") -> str:
    """
    Exporte un résumé en fichier Markdown.
    À appeler quand l'utilisateur demande de sauvegarder ou exporter un résumé.
    Input : contenu du résumé, nom du fichier (optionnel)
    Output : chemin du fichier créé
    """
 
  

    if not filename:
        filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Nettoie le nom de fichier — supprime tous les caractères spéciaux
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = filename.replace(" ", "_")[:50]  # limite la longueur
    filepath = f"exports/{filename}.md"

    os.makedirs("exports", exist_ok=True)

    # Nettoie le contenu avant d'écrire
    clean_content = content.encode('utf-8', errors='ignore').decode('utf-8')

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Research Summary\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write(clean_content)

    return f"Résumé exporté avec succès dans le fichier {filepath}"


@tool
def search_arxiv(query: str) -> str:
    """
    Cherche des papers scientifiques sur arXiv.
    Input : un sujet ou une question en anglais.
    Output : liste des papers les plus pertinents.
    """
    import time

    try:
        time.sleep(3)  # pause pour éviter le rate limit arXiv

        client = arxiv.Client(
            num_retries=2,
            delay_seconds=5
        )
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in client.results(search):
            results.append({
                "titre": paper.title,
                "auteurs": [a.name for a in paper.authors[:2]],
                "resume": paper.summary[:300],
                "pdf_url": paper.pdf_url,
                "date": str(paper.published.date())
            })

        if not results:
            return "Aucun paper trouvé."

        output = f"📚 {len(results)} papers trouvés :\n\n"
        for i, p in enumerate(results, 1):
            output += f"{i}. {p['titre']} ({p['date']})\n"
            output += f"   Auteurs : {', '.join(p['auteurs'])}\n"
            output += f"   Résumé : {p['resume']}...\n"
            output += f"   PDF : {p['pdf_url']}\n\n"

        return output

    except Exception as e:
        return f"❌ Erreur arXiv : {str(e)}. Réessaie dans quelques secondes."