# projet RAG
# 📚 RAG UI — Retrieval-Augmented Generation avec LangChain & LlamaIndex

Ce projet implémente une interface RAG (Retrieval-Augmented Generation) permettant de questionner le contenu de documents PDF. Il s’appuie sur des embeddings et modèles de chat OpenAI déployés sur Azure, et propose deux versions du moteur RAG : via **LangChain** et via **LlamaIndex**.

---

## 🔍 Objectif pédagogique

Ce projet est réalisé dans le cadre du module "Hands-on Project" (MAG 3). Il a pour but de :
- Implémenter une architecture RAG fonctionnelle
- Comparer deux frameworks (LangChain vs LlamaIndex)
- Déployer une interface interactive via Streamlit
- Explorer des pistes d’amélioration (langue, feedback, top-k…)

---

## 📁 Structure du projet


---

## ⚙️ Prérequis techniques

- Python 3.10+
- Accès à Azure OpenAI avec :
  - Un modèle d'embedding (`text-embedding-ada-002` ou autre)
  - Un modèle de chat (`gpt-35-turbo`, `gpt-4`, etc.)
- Un environnement virtuel Python (recommandé)

---

## 🔐 Configuration Azure (secrets/config.yaml)

Créer un fichier `secrets/config.yaml` **(non versionné)** :

```yaml
embedding:
  azure_endpoint: "https://mon-endpoint.openai.azure.com/"
  azure_deployment: "embedding-model"
  azure_api_version: "2023-05-15"
  azure_api_key: "ma-cle-api"

chat:
  azure_endpoint: "https://mon-endpoint.openai.azure.com/"
  azure_deployment: "chat-model"
  azure_api_version: "2023-05-15"
  azure_api_key: "ma-cle-api"
