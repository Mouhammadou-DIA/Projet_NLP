# Architecture Technique - Reddit RAG Chatbot

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture système](#2-architecture-système)
3. [Pipeline RAG](#3-pipeline-rag)
4. [Composants techniques](#4-composants-techniques)
5. [Flux de données](#5-flux-de-données)
6. [Stack technologique](#6-stack-technologique)
7. [Structure du projet](#7-structure-du-projet)
8. [API REST](#8-api-rest)
9. [Sécurité et performance](#9-sécurité-et-performance)
10. [Déploiement](#10-déploiement)

---

## 1. Vue d'ensemble

### 1.1 Description du projet

Le **Reddit RAG Chatbot** est un système de question-réponse intelligent basé sur l'architecture RAG (Retrieval-Augmented Generation). Il utilise une base de connaissances de **56 297 conversations Reddit** pour fournir des réponses contextuelles et pertinentes en **français et anglais**.

### 1.2 Objectifs

| Objectif | Description |
|----------|-------------|
| **Pertinence** | Fournir des réponses basées sur des conversations réelles |
| **Multilingue** | Support français/anglais avec réponse dans la langue de la question |
| **Performance** | Temps de réponse < 3 secondes |
| **Scalabilité** | Architecture modulaire et extensible |

### 1.3 Caractéristiques principales

- Architecture RAG (Retrieval-Augmented Generation)
- Support multilingue (60+ langues)
- API REST documentée (OpenAPI/Swagger)
- Interface web moderne (HTML/CSS/JS)
- LLM cloud via Groq (gratuit et rapide)
- Base vectorielle persistante (ChromaDB)

---

## 2. Architecture système

### 2.1 Diagramme d'architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COUCHE PRÉSENTATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐         ┌─────────────────────────────────────┐   │
│  │   Frontend Web      │         │         API REST (FastAPI)          │   │
│  │   (HTML/CSS/JS)     │ ──────► │    http://localhost:8000/api/v1     │   │
│  │   Port: 3000        │  HTTP   │    - POST /chat/                    │   │
│  └─────────────────────┘         │    - GET /chat/stats                │   │
│                                  │    - GET /health/                   │   │
│                                  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COUCHE SERVICE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                      ┌─────────────────────────────┐                        │
│                      │     ChatbotService          │                        │
│                      │  (Orchestration RAG)        │                        │
│                      │                             │                        │
│                      │  • Validation des entrées   │                        │
│                      │  • Coordination des services│                        │
│                      │  • Gestion des erreurs      │                        │
│                      └─────────────────────────────┘                        │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                  │
│              ▼                     ▼                     ▼                  │
│  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐         │
│  │ EmbeddingService  │ │ VectorStoreService│ │    LLMService     │         │
│  │                   │ │                   │ │                   │         │
│  │ • Vectorisation   │ │ • Stockage        │ │ • Génération      │         │
│  │ • Batch processing│ │ • Recherche       │ │ • Multi-provider  │         │
│  │ • Multilingue     │ │ • Similarité      │ │ • Groq/Ollama     │         │
│  └───────────────────┘ └───────────────────┘ └───────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                │                     │                     │
                ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COUCHE DONNÉES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐         │
│  │  Sentence         │ │    ChromaDB       │ │    Groq API       │         │
│  │  Transformers     │ │                   │ │                   │         │
│  │                   │ │  • 56,297 docs    │ │  • Llama 3.1 8B   │         │
│  │  • MiniLM-L12-v2  │ │  • SQLite backend │ │  • Cloud hosted   │         │
│  │  • 384 dimensions │ │  • Persistant     │ │  • Gratuit        │         │
│  └───────────────────┘ └───────────────────┘ └───────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Architecture en couches

| Couche | Responsabilité | Technologies |
|--------|---------------|--------------|
| **Présentation** | Interface utilisateur, API HTTP | HTML/CSS/JS, FastAPI |
| **Service** | Logique métier, orchestration | Python, Pydantic |
| **Données** | Stockage, embeddings, LLM | ChromaDB, Sentence Transformers, Groq |

---

## 3. Pipeline RAG

### 3.1 Qu'est-ce que RAG ?

**RAG (Retrieval-Augmented Generation)** est une architecture qui combine :
- **Retrieval** : Recherche de documents pertinents dans une base de connaissances
- **Augmented** : Enrichissement du contexte avec les documents trouvés
- **Generation** : Génération de réponse par un LLM avec ce contexte

### 3.2 Pourquoi RAG ?

| Approche | Avantages | Inconvénients |
|----------|-----------|---------------|
| **LLM seul** | Simple | Hallucinations, pas de données spécifiques |
| **Fine-tuning** | Personnalisé | Coûteux, données figées |
| **RAG** | Données à jour, traçable, pas d'hallucination | Plus complexe |

### 3.3 Pipeline détaillé

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE RAG COMPLET                               │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: INDEXATION (Offline - une seule fois)
═══════════════════════════════════════════════

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Données     │     │  Nettoyage   │     │  Génération  │
    │  Reddit CSV  │ ──► │  & Validation│ ──► │  Embeddings  │
    │  (56,297)    │     │              │     │  (384 dim)   │
    └──────────────┘     └──────────────┘     └──────────────┘
                                                     │
                                                     ▼
                                              ┌──────────────┐
                                              │  Stockage    │
                                              │  ChromaDB    │
                                              └──────────────┘

PHASE 2: INFÉRENCE (Online - à chaque requête)
══════════════════════════════════════════════

    ┌─────────────┐
    │  Question   │
    │  Utilisateur│
    └─────────────┘
          │
          ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Validation │     │  Embedding  │     │  Recherche  │
    │  & Nettoyage│ ──► │  Question   │ ──► │  Similarité │
    │             │     │  (384 dim)  │     │  Top-K (5)  │
    └─────────────┘     └─────────────┘     └─────────────┘
                                                   │
                                                   ▼
                                            ┌─────────────┐
                                            │  Documents  │
                                            │  Pertinents │
                                            └─────────────┘
                                                   │
          ┌────────────────────────────────────────┘
          ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Construction│     │  Génération │     │  Réponse    │
    │  Prompt +   │ ──► │  LLM (Groq) │ ──► │  Finale     │
    │  Contexte   │     │  Llama 3.1  │     │             │
    └─────────────┘     └─────────────┘     └─────────────┘
```

### 3.4 Étapes détaillées

#### Étape 1 : Réception de la question
```python
# Exemple de requête
{
    "message": "Quel téléphone me recommandes-tu ?",
    "use_llm": true,
    "n_results": 5
}
```

#### Étape 2 : Validation et nettoyage
- Vérification de la longueur (max 1000 caractères)
- Suppression des caractères spéciaux
- Détection d'injections potentielles

#### Étape 3 : Vectorisation (Embedding)
```python
# Conversion texte → vecteur 384 dimensions
embedding = embedding_service.embed_text("Quel téléphone me recommandes-tu ?")
# Résultat: [0.023, -0.156, 0.089, ..., 0.045]  # 384 valeurs float
```

**Pourquoi les embeddings ?**
- Représentation sémantique du texte
- Permet la recherche par similarité
- Multilingue : "phone" ≈ "téléphone" dans l'espace vectoriel

#### Étape 4 : Recherche par similarité
```python
# Recherche des K conversations les plus similaires
results = vector_store.search(embedding, n_results=5)
# Utilise la similarité cosinus pour le ranking
```

**Similarité cosinus** :
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```
- Résultat entre 0 et 1
- 1 = identique, 0 = aucun rapport

#### Étape 5 : Construction du prompt
```python
prompt = f"""
Context from Reddit conversations:
1. User asked about phones, response: "I recommend the iPhone 14..."
2. Discussion about smartphones: "Samsung Galaxy S23 is great..."

User question: Quel téléphone me recommandes-tu ?

IMPORTANT: Respond in French (same language as the question).
"""
```

#### Étape 6 : Génération LLM
```python
response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
)
```

#### Étape 7 : Retour de la réponse
```python
{
    "message": "Je te recommande le Samsung Galaxy S23 ou l'iPhone 14...",
    "sources": [...],
    "metadata": {
        "duration_ms": 1523,
        "method": "llm",
        "model": "llama-3.1-8b-instant"
    }
}
```

---

## 4. Composants techniques

### 4.1 EmbeddingService

**Rôle** : Convertir le texte en vecteurs numériques pour la recherche sémantique.

**Fichier** : `src/core/embeddings.py`

```python
class EmbeddingService:
    """Service d'embeddings multilingues"""

    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384

    def embed_text(self, text: str) -> List[float]:
        """Convertit un texte en vecteur"""

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Traitement par lots pour l'indexation"""
```

**Caractéristiques** :

| Paramètre | Valeur |
|-----------|--------|
| Modèle | `paraphrase-multilingual-MiniLM-L12-v2` |
| Dimensions | 384 |
| Langues | 60+ |
| Performance | ~10ms par texte |
| Taille modèle | ~120MB |

### 4.2 VectorStoreService

**Rôle** : Stocker et rechercher des vecteurs efficacement.

**Fichier** : `src/core/vector_store.py`

```python
class VectorStoreService:
    """Service de stockage vectoriel avec ChromaDB"""

    def add_conversations(self, conversations: List[Conversation]):
        """Indexe des conversations"""

    def search(self, embedding: List[float], n_results: int) -> List[SearchResult]:
        """Recherche par similarité cosinus"""

    def count(self) -> int:
        """Nombre de documents indexés"""
```

**Caractéristiques** :

| Paramètre | Valeur |
|-----------|--------|
| Backend | ChromaDB |
| Stockage | SQLite + fichiers binaires |
| Collection | `reddit_conversations_pro` |
| Documents | 56,297 |
| Métrique | Similarité cosinus |

### 4.3 LLMService

**Rôle** : Générer des réponses naturelles à partir du contexte.

**Fichier** : `src/core/llm_handler.py`

```python
class LLMService:
    """Service LLM multi-provider"""

    providers = ["groq", "ollama", "openai", "anthropic"]

    def generate(self, query: str, context: str) -> str:
        """Génère une réponse avec le contexte RAG"""
```

**Providers supportés** :

| Provider | Modèle | Vitesse | Coût |
|----------|--------|---------|------|
| **Groq** ✅ | llama-3.1-8b-instant | ~1-2s | Gratuit |
| Ollama | llama3:8b | ~30-60s | Gratuit (local) |
| OpenAI | gpt-4o-mini | ~2-3s | Payant |
| Anthropic | claude-3-haiku | ~2-3s | Payant |

### 4.4 ChatbotService

**Rôle** : Orchestrer le pipeline RAG complet.

**Fichier** : `src/services/chatbot_service.py`

```python
class ChatbotService:
    """Orchestrateur principal du chatbot"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService()

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Pipeline complet:
        1. Validation
        2. Embedding
        3. Recherche
        4. Génération (optionnelle)
        5. Formatage réponse
        """
```

---

## 5. Flux de données

### 5.1 Diagramme de séquence

```
┌────────┐     ┌────────┐     ┌──────────┐     ┌─────────┐     ┌───────┐     ┌──────┐
│Frontend│     │  API   │     │ Chatbot  │     │Embedding│     │Vector │     │ LLM  │
│        │     │FastAPI │     │ Service  │     │ Service │     │ Store │     │ Groq │
└───┬────┘     └───┬────┘     └────┬─────┘     └────┬────┘     └───┬───┘     └──┬───┘
    │              │               │                │              │            │
    │ POST /chat/  │               │                │              │            │
    │─────────────►│               │                │              │            │
    │              │  chat(req)    │                │              │            │
    │              │──────────────►│                │              │            │
    │              │               │                │              │            │
    │              │               │ embed_text()   │              │            │
    │              │               │───────────────►│              │            │
    │              │               │                │              │            │
    │              │               │   embedding    │              │            │
    │              │               │◄───────────────│              │            │
    │              │               │                │              │            │
    │              │               │        search(embedding)      │            │
    │              │               │──────────────────────────────►│            │
    │              │               │                │              │            │
    │              │               │           results             │            │
    │              │               │◄──────────────────────────────│            │
    │              │               │                │              │            │
    │              │               │              generate(query, context)      │
    │              │               │───────────────────────────────────────────►│
    │              │               │                │              │            │
    │              │               │                        response            │
    │              │               │◄───────────────────────────────────────────│
    │              │               │                │              │            │
    │              │   response    │                │              │            │
    │              │◄──────────────│                │              │            │
    │              │               │                │              │            │
    │   JSON       │               │                │              │            │
    │◄─────────────│               │                │              │            │
```

### 5.2 Format des données

#### Requête (ChatRequest)
```json
{
    "message": "Quel téléphone me recommandes-tu ?",
    "use_llm": true,
    "n_results": 5
}
```

#### Réponse (ChatResponse)
```json
{
    "message": "Je te recommande le Samsung Galaxy S23...",
    "sources": [
        {
            "context": "Looking for a new phone recommendation",
            "response": "Samsung Galaxy S23 is great for the price",
            "score": 0.89
        }
    ],
    "metadata": {
        "duration_ms": 1523,
        "method": "llm",
        "model": "llama-3.1-8b-instant",
        "n_sources": 5
    }
}
```

---

## 6. Stack technologique

### 6.1 Backend

| Technologie | Version | Usage |
|-------------|---------|-------|
| **Python** | 3.10+ | Langage principal |
| **FastAPI** | 0.109+ | Framework API REST |
| **Pydantic** | 2.5+ | Validation des données |
| **Uvicorn** | 0.27+ | Serveur ASGI |

### 6.2 Machine Learning / NLP

| Technologie | Version | Usage |
|-------------|---------|-------|
| **Sentence Transformers** | 2.3+ | Embeddings multilingues |
| **ChromaDB** | 0.4+ | Base de données vectorielle |
| **Groq SDK** | 0.1+ | Client LLM cloud |
| **PyTorch** | 2.2+ | Backend ML |

### 6.3 Frontend

| Technologie | Usage |
|-------------|-------|
| **HTML5** | Structure sémantique |
| **CSS3** | Styles (variables CSS, flexbox, grid, animations) |
| **JavaScript ES6+** | Logique, appels API (fetch async/await) |

### 6.4 Outils de développement

| Outil | Usage |
|-------|-------|
| **Ruff** | Linting et formatage Python |
| **Pytest** | Tests unitaires |
| **Loguru** | Logging structuré |

---

## 7. Structure du projet

```
📁 Projet-NLP/
│
├── 📁 api/                          # API REST FastAPI
│   ├── main.py                      # Point d'entrée, middleware, routes
│   ├── routes/
│   │   ├── chat.py                  # POST /chat/, GET /stats, GET /examples
│   │   └── health.py                # GET /health/, /ready, /live
│   └── schemas/
│       ├── request.py               # ChatRequest, SearchRequest
│       └── response.py              # ChatResponse, ErrorResponse
│
├── 📁 src/                          # Code source principal
│   ├── config/
│   │   ├── settings.py              # Configuration centralisée (Pydantic)
│   │   └── logging_config.py        # Configuration Loguru
│   │
│   ├── core/                        # Services principaux
│   │   ├── embeddings.py            # EmbeddingService (Sentence Transformers)
│   │   ├── vector_store.py          # VectorStoreService (ChromaDB)
│   │   └── llm_handler.py           # LLMService (Groq/Ollama/OpenAI)
│   │
│   ├── services/
│   │   └── chatbot_service.py       # ChatbotService (orchestration RAG)
│   │
│   ├── models/
│   │   └── schemas.py               # Modèles Pydantic (Conversation, etc.)
│   │
│   └── utils/
│       ├── data_loader.py           # Chargement CSV/JSON
│       ├── text_processor.py        # Nettoyage, normalisation
│       └── validators.py            # Validation entrées, sécurité
│
├── 📁 frontend/                     # Interface web moderne
│   ├── index.html                   # Structure HTML (sidebar, chat, stats)
│   ├── styles.css                   # CSS moderne (variables, animations)
│   └── app.js                       # JavaScript (fetch API, DOM)
│
├── 📁 data/                         # Données
│   ├── raw/                         # Données brutes
│   │   └── casual_data_windows.csv  # 56,297 conversations Reddit
│   ├── processed/                   # Données traitées
│   │   └── conversations.json       # Format JSON nettoyé
│   └── vector_db/                   # Base vectorielle
│       └── chroma_db/               # Fichiers ChromaDB
│
├── 📁 scripts/                      # Scripts utilitaires
│   ├── prepare_data.py              # CSV → JSON (nettoyage)
│   └── index_conversations.py       # JSON → ChromaDB (embeddings)
│
├── 📁 docs/                         # Documentation
│   └── ARCHITECTURE.md              # Ce document
│
├── .env                             # Variables d'environnement
├── .env.example                     # Template de configuration
├── requirements.txt                 # Dépendances Python
├── run_api.py                       # python run_api.py → :8000
└── run_frontend.py                  # python run_frontend.py → :3000
```

---

## 8. API REST

### 8.1 Endpoints

#### POST /api/v1/chat/
Envoyer un message et recevoir une réponse.

```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Quel téléphone acheter ?",
    "use_llm": true,
    "n_results": 5
  }'
```

**Paramètres** :

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `message` | string | requis | Question de l'utilisateur |
| `use_llm` | boolean | false | Utiliser le LLM pour générer |
| `n_results` | integer | 5 | Nombre de sources RAG |

#### GET /api/v1/chat/stats
Obtenir les statistiques du chatbot.

```json
{
    "total_conversations": 56297,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "llm_provider": "groq",
    "llm_model": "llama-3.1-8b-instant",
    "llm_available": true
}
```

#### GET /api/v1/chat/examples
Obtenir des exemples de questions.

```json
{
    "french": ["Quel téléphone acheter ?", ...],
    "english": ["What phone should I buy?", ...]
}
```

#### GET /api/v1/health/
Vérifier l'état de santé du service.

```json
{
    "status": "healthy",
    "version": "2.0.0",
    "services": {
        "embedding": "ok",
        "vector_store": "ok",
        "llm": "ok"
    }
}
```

### 8.2 Documentation interactive

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Swagger UI (interactif) |
| http://localhost:8000/redoc | ReDoc (documentation) |
| http://localhost:8000/openapi.json | Schéma OpenAPI |

---

## 9. Sécurité et performance

### 9.1 Sécurité

| Mesure | Implémentation |
|--------|----------------|
| **Validation des entrées** | Max 1000 caractères, sanitization |
| **Détection d'injection** | Patterns SQL/XSS bloqués |
| **CORS** | Origines configurables |
| **Rate limiting** | 100 requêtes/minute |
| **Logging** | Traçabilité complète |

### 9.2 Performance

| Métrique | Valeur |
|----------|--------|
| Temps de recherche vectorielle | < 100ms |
| Temps d'embedding | ~10ms |
| Temps LLM (Groq) | ~1-3s |
| **Temps total** | **~2-4s** |
| Mémoire API | ~500MB |
| Mémoire embeddings | ~200MB |
| Taille base vectorielle | ~500MB |

### 9.3 Optimisations appliquées

1. **Singleton pattern** : Services instanciés une seule fois
2. **Lazy loading** : Modèles chargés à la demande
3. **Batch processing** : Indexation par lots de 500
4. **Connection pooling** : Réutilisation des connexions

---

## 10. Déploiement

### 10.1 Prérequis

- Python 3.10+
- 4GB RAM minimum (8GB recommandé)
- Clé API Groq (gratuite sur console.groq.com)

### 10.2 Installation

```bash
# 1. Cloner le projet
git clone <repository>
cd Projet-NLP

# 2. Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec votre clé GROQ_API_KEY

# 5. Indexer les données (si pas déjà fait)
python scripts/prepare_data.py
python scripts/index_conversations.py
```

### 10.3 Lancement

```bash
# Terminal 1 - API (obligatoire)
python run_api.py
# → http://localhost:8000

# Terminal 2 - Frontend
python run_frontend.py
# → http://localhost:3000
```

### 10.4 Variables d'environnement

```bash
# .env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API
API_HOST=127.0.0.1
API_PORT=8000

# LLM (Groq - gratuit et rapide)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=gsk_xxxxxxxxxxxxx

# Embeddings
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu
```

---

## Annexes

### A. Glossaire

| Terme | Définition |
|-------|------------|
| **RAG** | Retrieval-Augmented Generation - Architecture combinant recherche et génération |
| **Embedding** | Représentation vectorielle d'un texte |
| **LLM** | Large Language Model - Modèle de langage (ex: Llama, GPT) |
| **Similarité cosinus** | Mesure de similarité entre vecteurs |
| **ChromaDB** | Base de données vectorielle open-source |
| **Groq** | Plateforme cloud pour LLM (gratuit) |

### B. Métriques du projet

| Métrique | Valeur |
|----------|--------|
| Conversations indexées | 56,297 |
| Dimension des embeddings | 384 |
| Langues supportées | 60+ |
| Temps de réponse moyen | ~2-3s |
| Taille totale du projet | ~600MB |

### C. Références

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API](https://console.groq.com/)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

---

**Document rédigé pour le projet NLP - Reddit RAG Chatbot**
**Version** : 2.0.0
**Dernière mise à jour** : Février 2025
