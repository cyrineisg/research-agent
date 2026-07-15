import arxiv
import os
import fitz
import urllib.request
import tempfile
from langchain.tools import tool
from dotenv import load_dotenv
import requests
import tempfile
import fitz

load_dotenv()

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
        return f"📄 Contenu extrait :\n\n{extracted[:2000]}"

    except requests.Timeout:
        return "❌ Timeout : PDF trop long à télécharger, essaie un autre paper."
    except Exception as e:
        return f"❌ Erreur : {str(e)}"