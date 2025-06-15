# 📚 Projet RAG — Retrieval-Augmented Generation

Ce projet met en œuvre une architecture **RAG (Retrieval-Augmented Generation)** pour améliorer la qualité des réponses générées par des modèles de langage (LLM). L’approche repose sur l’extraction automatique de contenu documentaire pertinent pour réduire les hallucinations.

---

## 🎯 Objectifs pédagogiques

- ⚙️ Implémenter une architecture RAG fonctionnelle (embeddings + moteur vectoriel + LLM)
- 💡 Comparer deux frameworks : **LangChain** et **LlamaIndex**
- 🖥️ Créer une interface utilisateur interactive avec **Streamlit**
- 🌐 Intégrer la **multilingue**, **personnalisation dynamique** et **feedback utilisateur**

---

## ✅ Fonctionnalités principales

- 📄 Upload de fichiers PDF avec vectorisation automatique
- ❓ Système de question-réponse avec injection de contexte
- 🧠 Choix du framework : `LangChain` ou `LlamaIndex`
- 🌍 Sélecteur de langue : Français, Anglais, Espagnol, Japonais
- 🛠️ Paramétrage dynamique du nombre de documents récupérés (`top_k`)
- 📝 Feedback utilisateur avec `st.radio` et enregistrement en base SQLite

---

## 🚀 Lancer l'application localement

```bash
git clone https://github.com/Larak01/projet-RAG.git
cd projet-RAG
pip install -r requirements.txt
streamlit run app.py
```

---

## 🗄️ Structure du projet

```
projet-RAG/
│
├── app.py                      # Application principale Streamlit
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
├── secrets/                   # Informations d'API (non incluses)
├── rag/
│   ├── langchain.py           # Pipeline utilisant LangChain
│   └── llamaindex.py          # Pipeline utilisant LlamaIndex
└── feedback.db                # (généré automatiquement) base SQLite pour le feedback
```

---

## 💡 Déploiement sur Streamlit Cloud

1. Crée un compte sur https://streamlit.io/cloud
2. Connecte ton dépôt GitHub
3. Déploie l'application avec `streamlit run app.py`
4. Crée un fichier `.streamlit/secrets.toml` pour stocker tes clés API Azure OpenAI

---

## 📬 Contact

Pour toute question, contactez [Larak01](https://github.com/Larak01).

Bon RAG ! 🎉
