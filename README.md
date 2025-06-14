# 📚 Projet RAG — Retrieval-Augmented Generation

Ce projet met en œuvre une architecture **RAG (Retrieval-Augmented Generation)**, un cadre qui améliore la qualité des réponses générées par les modèles de langage (LLM) en y intégrant automatiquement du **contenu extrait de documents** pertinents. Cette approche permet de **réduire les hallucinations** en s’appuyant sur une base de connaissances documentaire comme support contextuel.

---

## 🎯 Objectif pédagogique

Ce projet a pour objectifs :

- ⚙️ Implémenter une architecture **RAG** fonctionnelle, combinant embeddings, moteur de recherche vectorielle et LLM
- 💡 Explorer deux frameworks : **LangChain** et **LlamaIndex**
- 🖥️ Déployer une **interface utilisateur interactive** via **Streamlit**
- 🔍 Expérimenter des **améliorations ciblées** du RAG : multilingue, feedback utilisateur, moteur hybride, paramétrage dynamique


## ✅ Fonctionnalités clés

- Upload et vectorisation de fichiers PDF
- Question-answering contextuel
- Sélecteur de framework (LangChain / LlamaIndex)
- Choix de la langue de réponse
- Feedback utilisateur enregistré en base SQLite
- Paramétrage du nombre de documents à utiliser (`top_k`)

## 🚀 Lancer l'application localement

```bash
git clone https://github.com/Larak01/-projet-RAG.git
cd -projet-RAG
pip install -r requirements.txt
streamlit run app.py

## 🌍 Application déployée

👉 Accédez à l’application ici : [https://projet-rag-larak01.streamlit.app](https://projet-rag-larak01.streamlit.app)


