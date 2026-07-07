from agent.tools import search_arxiv, read_paper_pdf

#Test simple
result = search_arxiv.invoke("retrieval augmented generation 2024")
print(result)

#1. Cherche un papier sur arXiv
results = search_arxiv.invoke("retrieval augmented generation")
print("== Recherche ==")
print(results)

# 2. Lis le premier PDF trouvé
# Copie une URL PDF depuis les résultats ci-dessus et colle-la ici
pdf_url = "https://arxiv.org/pdf/2005.11401" #papier RAG original de Meta
content = read_paper_pdf.invoke(pdf_url)
print("\n=== LECTURE PDF ===")
print(content)
