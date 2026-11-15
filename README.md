# RAG over your own docs

A small question-answering app over your own PDFs and notes. Every answer cites
the document and page it came from, it says "I don't know" when the answer is
not in your docs, and an eval script prints a pass rate.

```python
docs   = load("my\_docs/")
chunks = chunk(docs, size=400, overlap=50)
vecs   = embed(chunks)
index  = VectorIndex(vecs)
hits   = index.search(question, k=5)
answer = llm(question, hits)      # cites \[n] -> file, page
evaluate(answer)                  # grounded?
```

Those seven lines are the real structure of the code: see `rag/pipeline.py`.

## Quick start

```bash
python -m venv .venv \&\& source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env            # add your OpenRouter key (or any OpenAI-compatible endpoint)

python ingest.py my\_docs/       # load -> chunk -> embed -> index
python ask.py "At what temperature must frozen products be kept?"
```

```
Frozen products must be kept at -18 °C or colder \[1].

Sources:
  \[1] cold\_chain\_handbook.pdf, p. 1
```

Put your own `.pdf`, `.md` or `.txt` files in `my\_docs/` and run `ingest.py` again.
`my\_docs/` ships with two fictional sample documents so everything runs out of the box.

No API key yet? `python ask.py --offline "..."` and `python eval.py ... --offline`
swap the LLM for a simple sentence picker, and `EMBED\_MODEL=hash` swaps the
embedding model for a bag-of-words hash. Both are for smoke tests only.

## How it works

|Step|File|What it does|
|-|-|-|
|load|`rag/load.py`|Reads PDFs page by page (pypdf) and text/markdown files, keeping file name and page number|
|chunk|`rag/chunk.py`|\~400-token chunks with \~50 overlap. Chunks never cross a page, so each one maps to one citable page|
|embed|`rag/embed.py`|`all-MiniLM-L6-v2` by default (runs on a laptop CPU); `BAAI/bge-small-en-v1.5` is a stronger drop-in|
|index|`rag/index.py`|Chroma, persisted to `.chroma/`, cosine similarity|
|search|`rag/pipeline.py`|Top 5 chunks; anything under `MIN\_SCORE` is dropped|
|answer|`rag/llm.py`|The model sees numbered passages and must cite `\[n]` after each claim, or reply `I don't know`|
|evaluate|`eval.py`|Regex check per question, prints the pass rate|

"I don't know" has two layers: if no chunk clears `MIN\_SCORE` the app refuses
without calling the LLM, and the system prompt tells the model to refuse when
the passages do not contain the answer.

## Evaluation

```bash
python eval.py eval\_sets/sample.jsonl
# pass rate: 90% (9/10)   <- example output format
```

A test set is one JSON object per line:

```json
{"question": "How long are temperature records retained?", "expected": "(two|2)\\\\s+years"}
{"question": "Who won the 2022 FIFA World Cup?", "answerable": false}
```

An answerable question passes when the answer matches the `expected` regex and
carries a valid `\[n]` citation. An unanswerable one passes when the app says
"I don't know". `--min 0.8` makes the script exit 1 below 80%, handy for CI.

For a bigger benchmark, `scripts/fetch\_wiki\_eval.py` downloads
[`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)
(Wikipedia passages plus test questions with answers):

```bash
python scripts/fetch\_wiki\_eval.py --questions 100
python ingest.py data/wiki
python eval.py eval\_sets/wiki.jsonl
```

### Results

|Setup|Test set|Pass rate|
|-|-|-|
|`hash` embedder + `--offline`, `MIN\_SCORE=0.15` (no models at all)|`sample.jsonl`|90% (9/10)|
|`all-MiniLM-L6-v2` + *openai/gpt-4o*|`sample.jsonl`|90% (9/10)|
|`all-MiniLM-L6-v2` + *openai/gpt-4o*|`wiki.jsonl` (100 q)|90% (9/10)|

## Configuration

All settings live in `.env` (see `.env.example`): LLM endpoint and model,
embedding model, chunk size and overlap, `TOP\_K`, `MIN\_SCORE`. Re-run
`ingest.py` after changing the embedding model or chunking.

`MIN\_SCORE=0.25` is a starting point for MiniLM; use `python ask.py --show-chunks "..."`
to see real scores on your documents and tune it.

## Tests

```bash
pytest
```

## Next steps

* LLM judge on top of the regex eval (faithfulness and answer relevance)
* Reranker over the top 20 before sending the top 5
* Hybrid search (BM25 + vectors)
* pgvector instead of Chroma for a production-like setup

Based on the ideas in the DeepLearning.AI short course "Building and Evaluating Advanced RAG".

## License

MIT

