# WLO Duplicate Detection API

FastAPI-basierter Dienst zur Erkennung von Dubletten (ähnlichen Inhalten) im WLO-Repository.

## Features

- **Hash-basierte Erkennung (MinHash)**: Schnelle Ähnlichkeitsberechnung basierend auf Textshingles
- **Embedding-basierte Erkennung (ONNX)**: Semantische Ähnlichkeit mit `paraphrase-multilingual-MiniLM-L12-v2`
- **Embedding-API**: Separater Endpunkt für Embedding-Generierung (ohne Rate Limit)
- **Vercel-kompatibel**: Nutzt ONNX Runtime statt PyTorch (~143MB quantisiert)
- **Flexible Eingabe**: Per Node-ID oder direkte Metadateneingabe
- **Paginierung**: Automatische Paginierung für große Kandidatenmengen (>100)
- **Rate Limiting**: Schutz vor Überlastung (100 Requests/Minute für Detection-Endpoints)
- **Konfigurierbare Schwellenwerte**: Ähnlichkeitsschwellen individuell einstellbar
- **Konfigurierbare Suchfelder**: Wählen Sie, welche Metadaten für die Kandidatensuche verwendet werden

## Installation

```bash
cd duplicate-detection
pip install -r requirements.txt
```

## Starten

```bash
# Direkt mit Python
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Oder mit dem Run-Script
python run.py
```

Die API ist dann unter `http://localhost:8000` erreichbar.

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpunkte

### Hash-basierte Erkennung

#### `POST /detect/hash/by-node`
Dublettenerkennung für einen bestehenden WLO-Inhalt per Node-ID.

```bash
curl -X POST "http://localhost:8000/detect/hash/by-node" \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "12345678-1234-1234-1234-123456789abc",
    "environment": "production",
    "similarity_threshold": 0.8,
    "search_fields": ["title", "description", "keywords", "url"],
    "max_candidates": 100
  }'
```

#### `POST /detect/hash/by-metadata`
Dublettenerkennung für neue Inhalte per direkter Metadateneingabe.

```bash
curl -X POST "http://localhost:8000/detect/hash/by-metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {
      "title": "Mathematik für Grundschüler",
      "description": "Lernen Sie die Grundlagen der Mathematik",
      "keywords": ["Mathematik", "Grundschule", "Rechnen"]
    },
    "environment": "production",
    "similarity_threshold": 0.8
  }'
```

### Embedding-basierte Erkennung

#### `POST /detect/embedding/by-node`
Semantische Dublettenerkennung per Node-ID.

```bash
curl -X POST "http://localhost:8000/detect/embedding/by-node" \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "12345678-1234-1234-1234-123456789abc",
    "environment": "production",
    "similarity_threshold": 0.85
  }'
```

#### `POST /detect/embedding/by-metadata`
Semantische Dublettenerkennung per direkter Metadateneingabe.

```bash
curl -X POST "http://localhost:8000/detect/embedding/by-metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {
      "title": "Mathematik für Grundschüler",
      "description": "Lernen Sie die Grundlagen der Mathematik"
    },
    "environment": "staging",
    "similarity_threshold": 0.85
  }'
```

### Embedding-Generierung (ohne Rate Limit)

#### `POST /embed`
Erzeugt einen 384-dimensionalen Embedding-Vektor für einen Text.

```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "Dies ist ein Beispieltext"}'
```

**Response:**
```json
{
  "success": true,
  "text": "Dies ist ein Beispieltext",
  "embedding": [0.0234, -0.0567, ...],
  "dimensions": 384,
  "model": "paraphrase-multilingual-MiniLM-L12-v2"
}
```

#### `POST /embed/batch`
Erzeugt Embeddings für mehrere Texte gleichzeitig (effizienter als Einzelaufrufe).

```bash
curl -X POST "http://localhost:8000/embed/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

**Response:**
```json
{
  "success": true,
  "embeddings": [[...], [...], [...]],
  "dimensions": 384,
  "count": 3,
  "model": "paraphrase-multilingual-MiniLM-L12-v2"
}
```

## Request-Parameter

### Gemeinsame Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `environment` | string | `production` | WLO-Umgebung: `production` oder `staging` |
| `search_fields` | array | `["title", "description", "keywords", "url"]` | Felder für Kandidatensuche |
| `max_candidates` | int | `100` | Max. Kandidaten pro Suchfeld (1-1000, Paginierung ab >100) |

### Hash-spezifisch

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `similarity_threshold` | float | `0.8` | Mindestähnlichkeit (0-1) |

### Embedding-spezifisch

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `similarity_threshold` | float | `0.85` | Mindest-Kosinus-Ähnlichkeit (0-1) |

### Metadata-Objekt

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `title` | string | Titel des Inhalts |
| `description` | string | Beschreibungstext |
| `keywords` | array[string] | Liste von Schlagwörtern |
| `url` | string | URL des Inhalts |

## Response-Format

```json
{
  "success": true,
  "source_node_id": "12345678-...",
  "source_metadata": {
    "title": "...",
    "description": "...",
    "keywords": ["..."],
    "url": "..."
  },
  "method": "hash",
  "threshold": 0.8,
  "total_candidates_checked": 42,
  "duplicates": [
    {
      "node_id": "abcdef12-...",
      "title": "Ähnlicher Inhalt",
      "similarity_score": 0.92,
      "match_source": "title",
      "url": "https://..."
    }
  ],
  "error": null
}
```

## Ablauf der Erkennung

1. **Metadaten laden**: Bei Node-ID-Anfragen werden die vollständigen Metadaten von WLO geladen
2. **Kandidatensuche**: Suche nach potenziellen Duplikaten über ngsearch:
   - `title`: Suche im Volltextindex
   - `description`: Suche in den ersten 100 Zeichen
   - `keywords`: Suche mit kombinierten Keywords
   - `url`: Exakte URL-Suche
3. **Ähnlichkeitsberechnung**:
   - **Hash**: MinHash-Signaturen + Kosinus-Ähnlichkeit
   - **Embedding**: Sentence-Transformer + Kosinus-Ähnlichkeit
4. **Filterung**: Nur Ergebnisse über dem Schwellenwert werden zurückgegeben

## Unterschied Hash vs. Embedding

| Aspekt | Hash (MinHash) | Embedding |
|--------|----------------|-----------|
| **Geschwindigkeit** | Sehr schnell | Langsamer (GPU empfohlen) |
| **Erkennung** | Wörtliche Ähnlichkeit | Semantische Ähnlichkeit |
| **Modell** | Shingle-basiert | paraphrase-MiniLM-L12 |
| **Ideal für** | Exakte/nahe Duplikate | Umformulierte Texte |

## Entwicklung

```bash
# Mit Auto-Reload
uvicorn app.main:app --reload --port 8000
```

## Modell exportieren (für schnelleren Start)

Das ONNX-Modell kann lokal gespeichert werden für schnellere Ladezeiten:

```bash
# Quantisiertes Modell exportieren (~143 MB, Vercel-kompatibel)
python scripts/export_model_quantized.py

# Oder: Volles Modell (~480 MB, nur für lokale Nutzung)
python scripts/export_model.py
```

Der Health-Endpoint zeigt an, ob das lokale Modell verwendet wird:
```json
{
  "embedding_model_local": true
}
```

## Deployment auf Vercel

```bash
# 1. Modell exportieren (quantisiert für Vercel)
python scripts/export_model_quantized.py

# 2. Vercel CLI installieren
npm i -g vercel

# 3. Deployen (inkl. models/ Ordner)
cd duplicate-detection
vercel
```

Die `vercel.json` ist bereits konfiguriert. Beide Erkennungsmethoden funktionieren auf Vercel dank ONNX Runtime und quantisiertem Modell.

## Rate Limits

| Endpunkt | Rate Limit |
|----------|------------|
| `/detect/*` | 100/Minute |
| `/embed` | Kein Limit |
| `/embed/batch` | Kein Limit |
| `/health` | Kein Limit |

## Technologien

- **FastAPI**: Web-Framework
- **ONNX Runtime**: Embedding-Modell (Vercel-kompatibel)
- **Optimum**: Hugging Face ONNX-Integration
- **scikit-learn**: Ähnlichkeitsberechnung
- **Pydantic**: Datenvalidierung
- **Loguru**: Logging
- **SlowAPI**: Rate Limiting
