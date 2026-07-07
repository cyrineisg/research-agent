import arxiv
import os
import fitz
import urllib.request
import tempfile
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def search_arxiv(query: str) -> str:
    """Cherche des papiers scientifiques sir arXiv.
    Input : un sujet ou une question en anglais.
    Output : liste les 5 papiers les plus pertinents.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for paper in client.results(search):
        results.append({
            "titre": paper.title,
            "auteurs": [a.name for a in paper.authors[:3]],
            "resume": paper.summary[:400],
            "pdf_url": paper.pdf_url,
            "date": str(paper.published.date())
        })
    if not results: 
        return "Aucun papier trouvé pour la requête : "
    
    #formate la réponse proprement
    output = f"{len(results)} papiers trouvés pour la requête : '{query}'\n\n"
    for i, p in enumerate(results, 1):
        output += f"**{i}. {p['titre']}**\n"
        output += f"    Auteurs : {', '.join(p['auteurs'])}\n"
        output += f"    Date de publication : {p['date']}\n"
        output += f"    Résumé : {p['resume']}...\n"
        output += f"    PDF : {p['pdf_url']}\n\n"
    return output

@tool
def read_paper_pdf(pdf_url: str) -> str:
    """
    télécharge et extrait le texte d'un fichier PDF depuis une URL arXiv.
    Retourne l'introduction et la conclusion du papier.
    Input: URL du PDF (ex: https://arxiv.org/pdf/2307.09288)
    Output: texte extrait du papier
    """
    try:
        # Télécharge le PDF dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            urllib.request.urlretrieve(pdf_url, tmp.name)
            tmp_path = tmp.name
        
        # Ouvre le PDF avec PyMuPDF
        doc = fitz.open(tmp_path)
        total_pages = len(doc)

        # Stratégie : intro (pages 1-2) + conclusion (dernière page)
        pages_to_read = [0, 1]
        if total_pages > 3:
            pages_to_read.append(total_pages - 1)
        
        extracted = ""
        for page_num in pages_to_read:
            page = doc[page_num]
            extracted += f"\n--- Page {page_num + 1} ---\n"
            extracted += page.get_text()

        doc.close()

        # Limite à 3000 caractères pour ne pas dépasser le contexte LLM
        extracted = extracted[:3000]

        return f" Contenu extrait du papier :\n\n{extracted}"
    
    except Exception as e:
        return f"Erreur lors de la lecture du PDF : {str(e)}"
