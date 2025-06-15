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
- 🔐 Connexion sécurisée à **Azure OpenAI** pour le LLM et les embeddings

---

## 🚀 Lancer l'application localement

```bash
git clone https://github.com/Larak01/projet-RAG.git
cd projet-RAG
pip install -r requirements.txt
streamlit run app.py
```

> 🧠 Ce projet utilise **Azure OpenAI** pour accéder aux modèles `gpt-35-turbo` et `text-embedding-ada-002`. Les paramètres API sont stockés localement dans un fichier `config.toml` (non inclus dans le dépôt).

---

## 🗄️ Structure du projet

```
projet-RAG/
│
├── app.py                      # Application principale Streamlit
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
├── samples/                   # Fichiers PDF de démonstration
├── rag/
│   ├── langchain.py           # Pipeline utilisant LangChain
│   └── llamaindex.py          # Pipeline utilisant LlamaIndex
├── pages/feedback.py          # Page dédiée à l'analyse des retours utilisateur
└── feedback.db                # Base SQLite pour stocker les feedbacks
```

---

## 🔐 Configuration locale

Créer un fichier `config.toml` (non versionné) contenant vos identifiants Azure OpenAI :

```toml
[chat]
azure_deployment = "gpt-chat"
azure_api_key = "sk-..."
azure_endpoint = "https://projet-rag-openai.azure.com/"
azure_api_version = "2023-12-01-preview"

[embedding]
azure_deployment = "embed-ada"
azure_api_key = "sk-..."
azure_endpoint ="https://projet-rag-openai.azure.com/"
azure_api_version = "2023-12-01-preview"
```


---

## 💡 Déploiement sur Streamlit Cloud

1. Crée un compte sur https://streamlit.io/cloud
2. Connecte ton dépôt GitHub
3. Ajoute tes secrets dans `Settings > Secrets` de l'app

---

## 📬 Contact

Pour toute question, contactez [Larak01](https://github.com/Larak01).

Bon RAG avec Azure OpenAI ! 🎉
