# Sistema de Recuperación de Información Multimodal con RAG

Proyecto final — Recuperación de Información.

Sistema de búsqueda multimodal (texto + imagen) sobre un corpus de productos de
Amazon, con generación de respuestas mediante RAG y una interfaz conversacional.

**Integrantes:** Goyes Antony, Quilumba Joel, Sangango Jose Armando

## Arquitectura

- **Corpus:** ~1.800 productos de Amazon (subconjunto de ESCI/SQID) con título,
  imagen y juicios de relevancia graduados (E/S/C/I) para 100 consultas reales.
- **Embeddings:** CLIP (`openai/clip-vit-base-patch32`) — texto e imágenes en un
  espacio compartido de 512 dimensiones, normalizados L2.
- **Base vectorial:** ChromaDB persistente, métrica coseno, dos colecciones
  (títulos e imágenes).
- **Recuperación:** búsqueda por similitud sobre la señal de imagen (α=0, la
  mejor configuración según el barrido experimental) con pool de 50 candidatos.
- **Re-ranking:** cross-encoder `ms-marco-MiniLM-L-6-v2` reordena el pool
  (NDCG@10: 0.624 → 0.722).
- **RAG:** el Top-k recuperado se inyecta como contexto a un LLM (Llama 3.1 vía
  Groq; adaptador intercambiable con Gemini) que genera la respuesta.
- **Interfaz:** chat Gradio con visualización de evidencias (imágenes + scores).

### Funcionalidades de excelencia

1. **Re-ranking** con cross-encoder (evaluado: +15.8% NDCG@10 relativo).
2. **Query Expansion:** reformulación de consultas conversacionales con LLM
   (evaluada contra versiones conversacionales de las 100 queries).
3. **Relevance Feedback:** Me gusta / No me gusta con algoritmo de Rocchio
   sobre el vector de consulta.
4. **Memoria conversacional:** reescritura de consultas de seguimiento usando
   el historial del chat, aplicada a recuperación y generación.

## Ejecución en Google Colab (recomendado)

1. Abrir `ProyectoFinal_RI.ipynb` en Google Colab.
2. Activar GPU: `Entorno de ejecución → Cambiar tipo → T4 GPU` (opcional pero
   recomendado; en CPU funciona más lento).
3. Configurar los **Secrets** de Colab (icono de llave, activar acceso al notebook):
   - `HF_TOKEN`: token de lectura de Hugging Face (para descargar SQID).
   - `GROQ_API_KEY`: API key gratuita de [console.groq.com](https://console.groq.com).
   - `GOOGLE_API_KEY` (opcional): API key de Gemini como motor alternativo.
4. **Primera ejecución** (construcción del corpus, una sola vez):
   ejecutar las secciones **1 (Setup)** y **2 (Construcción del corpus)** en orden.
   Esto descarga ESCI y SQID, muestrea el corpus, baja ~1.800 imágenes y genera
   los embeddings. Todo se guarda en Google Drive (`MyDrive/proyecto_ri/`).
   Duración aproximada: 20-30 minutos.
5. **Ejecuciones posteriores:** solo la sección **1 (Setup)** (~5 minutos) —
   los datos persisten en Drive.
6. **Interfaz:** ejecutar la celda de la sección 4. Se genera un enlace público
   de Gradio.
7. **Evaluación:** las tablas de resultados están pre-calculadas en
   `data/resultados_*.csv`; las celdas de la sección 3 permiten regenerarlas.

## Ejecución local (alternativa)

Requiere Python 3.10+ y las dependencias de `requirements.txt`:

```bash
pip install -r requirements.txt
```

Notas para ejecución local:
- Reemplazar el montaje de Google Drive por una ruta local (variable `DATA_DIR`).
- Reemplazar `google.colab.userdata` por variables de entorno
  (`HF_TOKEN`, `GROQ_API_KEY`).
- Con GPU NVIDIA, instalar la build CUDA de torch para acelerar los embeddings.

## Estructura de datos generados (en Drive o `data/`)

```
data/
├── corpus.parquet              # 1.823 productos (id, título, bullets, image_url)
├── queries.csv                 # 100 consultas de evaluación
├── qrels.csv                   # juicios de relevancia (gain: E=3 S=2 C=1 I=0)
├── queries_conversacionales.csv# versiones conversacionales (Exc. 2)
├── reformulaciones.csv         # reformulaciones cacheadas (Exc. 2)
├── resultados_alpha.csv        # barrido de α (evaluación base)
├── resultados_rerank.csv       # evaluación Exc. 1
├── resultados_expansion.csv    # evaluación Exc. 2
├── images/                     # 1.823 imágenes de producto
├── embeddings/                 # matrices CLIP (.npy) + orden de ids
└── chroma_db/                  # índice vectorial persistente
```

## Notas de diseño

- La recuperación usa **solo la señal de imagen** (α=0): el barrido experimental
  mostró que CLIP-imagen supera ampliamente a CLIP-texto para ranking
  query-producto (NDCG@10 0.624 vs 0.297).
- El sistema **desacopla recuperación y generación**: si el LLM no está
  disponible, la interfaz muestra igualmente las evidencias recuperadas.
- Las llamadas al LLM están cacheadas donde es posible (conversacionales,
  reformulaciones) para tolerar límites de cuota de las APIs gratuitas.
- El adaptador `LLMCompat`/`LLMGroq` permite cambiar de proveedor de LLM
  (Groq/Gemini) modificando una línea.
