# 📚 Projet RAG — Retrieval-Augmented Generation

Ce projet met en œuvre une architecture **RAG (Retrieval-Augmented Generation)** pour améliorer la qualité des réponses générées par des modèles de langage (LLM). L’approche repose sur l’extraction automatique de contenu documentaire pertinent pour réduire les hallucinations.

---

## 🎯 Ce que j'ai réalisé dans ce projet

- 🔧 J'ai intégré **deux frameworks** RAG au choix : **LangChain** et **LlamaIndex**
- 📁 J'ai ajouté une interface d'upload de fichiers PDF avec vectorisation automatique
- 🔤 J'ai ajouté un **sélecteur de langue de réponse** multilingue : Français, Anglais, Espagnol, Japonais
- 🔎 J'ai intégré un **slider** permettant de choisir dynamiquement combien de documents (`k`) sont récupérés par la recherche vectorielle
- 🧪 J'ai implémenté un système de **feedback utilisateur** via `st.radio` puis sauvegardé les réponses dans une base **SQLite**
- 🧠 J'ai utilisé **Azure OpenAI** comme fournisseur de LLM et d'embeddings
- ☁️ J'ai configuré deux **déploiements personnalisés** sur Azure :
  - `gpt-35-turbo` (déployé sous le nom `gpt-chat`) pour les réponses du chatbot
  - `text-embedding-ada-002` (déployé sous le nom `embed-ada`) pour la vectorisation des documents
- 🔐 J’ai appris à sécuriser mes clés API via `st.secrets` sur Streamlit Cloud

---

## ✅ Fonctionnalités principales

- 📄 Upload de fichiers PDF avec vectorisation automatique
- ❓ Système de question-réponse avec injection de contexte
- 🧠 Choix du framework : `LangChain` ou `LlamaIndex`
- 🌍 Sélecteur de langue : Français, Anglais, Espagnol, Japonais
- 🛠️ Paramétrage dynamique du nombre de documents récupérés (`top_k`)
- 📝 Feedback utilisateur avec `st.radio` et enregistrement en base SQLite
- 🔐 Connexion sécurisée à **Azure OpenAI** pour le LLM et les embeddings
- 📦 Utilisation de deux modèles Azure déployés manuellement :
  - `gpt-35-turbo` pour le chat (`gpt-chat`)
  - `text-embedding-ada-002` pour les embeddings (`embed-ada`)

---

## 🚀 Lancer l'application localement

```bash
git clone "https://github.com/Larak01/projet-RAG.git"
cd projet-RAG
pip install -r requirements.txt
streamlit run app.py
```

> 🧠 Ce projet utilise **Azure OpenAI** pour accéder aux modèles `gpt-35-turbo` et `text-embedding-ada-002`. Les paramètres API sont stockés localement dans un fichier `config.toml` (non inclus dans le dépôt) ou gérés dans `st.secrets` si déployé sur Streamlit Cloud.

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

## 💡 Déploiement sur Streamlit Cloud

1. Crée un compte: "https://share.streamlit.io/user/larak01"
2. Connecte mon dépôt GitHub
3. Ajoute mes secrets dans `Settings > Secrets` de l'app

---

## 📬 Contact

Pour toute question, contactez [Larak01](https://github.com/Larak01).

Bon RAG avec Azure OpenAI ! 🎉
