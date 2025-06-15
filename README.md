# 📚 Projet RAG — Retrieval-Augmented Generation

Ce projet met en œuvre une architecture **RAG (Retrieval-Augmented Generation)** pour améliorer la qualité des réponses générées par des modèles de langage (LLM). L’approche repose sur l’extraction automatique de contenu documentaire pertinent pour réduire les hallucinations.

---

## 🎯 Objectifs pédagogiques

- ⚙️ Implémenter une architecture RAG fonctionnelle (embeddings + moteur vectoriel + LLM)
- 💡 Comparer deux frameworks : **LangChain** et **LlamaIndex**
- 🖥️ Créer une interface utilisateur interactive avec **Streamlit**
- 🔍 Tester des optimisations : moteur hybride, multilingue, feedback utilisateur, paramétrage dynamique

---

## ✅ Fonctionnalités principales

- 📄 Upload de fichiers PDF et vectorisation automatique
- ❓ Système de question-réponse avec contexte injecté
- 🧠 Choix du framework : `LangChain` ou `LlamaIndex`
- 🌐 Sélecteur de langue : Français, Anglais, Espagnol, Japonais
- 🛠️ Contrôle du nombre de documents (`top_k`) à récupérer
- 📝 Enregistrement du feedback utilisateur en base SQLite

---

## 🚀 Lancer l'application localement

```bash
git clone https://github.com/Larak01/projet-RAG.git
cd projet-RAG
pip install -r requirements.txt
streamlit run app.py

🌍 Application déployée
🔗 Accéder à l’application sur Streamlit Cloud
