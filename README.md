# Evident — Reasoning Probes for Argumentation Mining in UN Resolutions

Team **POINTERS** submission for the [UZH ArgMining 2026 Shared Task](https://github.com/ZurichNLP/ArgMining-2026-UZH-Shared-Task) — recovering argumentative structure from UN/UNESCO resolutions.

The pipeline applies the **Evident Framework** to resolution analysis: preambular paragraphs are treated as evidence and operative paragraphs as the claims they support, with four named reasoning strategies bridging the two. Structured **reasoning probes** embedded in every prompt require the model to quote verbatim evidence, name the logical mechanism, and explicitly rule out an alternative — making each decision grounded and falsifiable. Generation runs fully locally using **Qwen3-8B-GGUF (Q8_0)** via `llama-cpp-python`; no cloud API is needed.

| Split | LLM-Judge Score |
|-------|----------------|
| Train (2,694 docs) | 81 / 100 |
| Test  (89 docs)    | 77 / 100 |

*Judge: Claude Sonnet 4.6 — used only for evaluation, not generation.*

---

## Files

| File | Purpose |
|------|---------|
| `config_qwen.py` | Paths, model settings, prompts, tag vocabulary loader |
| `claims_pipeline_qwen.py` | Main pipeline — classification + relation prediction |

---

## How It Works

**Step 1 — Paragraph Classification**
- Classifies each paragraph as `preambular` (evidence/rationale) or `operative` (directive/claim)
- Assigns education-dimension tags from a 126-code controlled vocabulary
- Uses a sliding context window of ~21 surrounding paragraphs for local context
- Requires a 4-part reasoning probe: opening keyword quote, tag justification phrase, dominant strategy, and structural role in the resolution's argument

**Step 2 — Relation Prediction**
- Retrieves top-15 semantically related candidate paragraphs via `sentence-transformers` (`all-MiniLM-L6-v2`)
- Predicts argumentative relations: `supporting`, `complemental`, `contradictive`, `modifying`
- Each relation is grounded in a named Evident Framework strategy: Causal / Corroboration / Contrastive / Triangulation
- Requires a 5-part reasoning probe: quoted phrase from source, quoted phrase from target, labelled relation with justification, strategy and its logical mechanism, and one explicitly rejected alternative

Non-argumentative paragraphs (headers, date stamps, institutional name lines) are flagged in Step 1 and skipped in Step 2.

Qwen3's internal chain-of-thought is suppressed via the `/no_think` prefix and `<think>` tag stripping, so outputs are clean JSON only.

---

## Requirements

### Hardware
- NVIDIA GPU with **≥ 16 GB VRAM** recommended (RTX 4080 tested)
- Q8_0 quantisation fits entirely in 16 GB with `n_gpu_layers=-1`

### Python packages

```bash
# llama-cpp-python with CUDA support (required)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# remaining dependencies
pip install huggingface_hub sentence-transformers scikit-learn numpy json-repair
```

---

## Setup

### 1. Dataset

Clone the shared task data into the **same directory as the scripts**:

```bash
git clone https://huggingface.co/datasets/ZurichNLP/ArgMining-2026-UZH-Shared-Task
```

The scripts expect:
```
ArgMining-2026-UZH-Shared-Task/
├── train-data/
├── test-data/
└── education_dimensions_updated.csv
```

### 2. Output path

Open `config_qwen.py` and update `OUTPUT_BASE_PATH` to a directory on your machine:

```python
OUTPUT_BASE_PATH = '/your/path/to/output'
```

### 3. HuggingFace token (optional)

The model (`Qwen/Qwen3-8B-GGUF`) is public. A token is only needed if you have gated model access. Set it via:

```bash
export HF_TOKEN="hf_..."
```

On first run the Q8_0 GGUF file (~8.6 GB) is downloaded automatically and cached at `~/.cache/huggingface/hub/`.

---

## Usage

### Run on test data (submission)
```bash
python claims_pipeline_qwen.py --test
```

### Run on train data
```bash
python claims_pipeline_qwen.py
```

### Process a limited number of documents (development / sanity check)
```bash
python claims_pipeline_qwen.py --test --limit 5
```

### Custom input folder
```bash
python claims_pipeline_qwen.py --input /path/to/json/folder
```

Already-processed files are skipped automatically — safe to resume interrupted runs.

---

## Output Schema

Each prediction file follows the official shared task schema:

```json
{
  "TEXT_ID": "ICPE-03-1934_RES1-FR_res_1",
  "METADATA": {
    "structure": {
      "preambular_para": [1, 2, 3],
      "operative_para":  [4, 5, 6, 7],
      "think": "Document-level reasoning..."
    }
  },
  "body": {
    "paragraphs": [
      {
        "para_number": 1,
        "para":        "Original French text...",
        "para_en":     "English translation...",
        "type":        "preambular",
        "tags":        ["ACT_IO", "LAW_INTER"],
        "matched_pars": {"4": ["supporting"], "6": ["complemental"]},
        "think":       "Step 1 reasoning...\n\n[→ para 4] Step 2 reasoning..."
      }
    ]
  }
}
```

---

## Evident Framework — Strategy to Relation Mapping

| Reasoning Strategy | Relation Type   | Description |
|--------------------|-----------------|-------------|
| Causal             | `supporting`    | Premise leads to or justifies a directive |
| Corroboration      | `complemental`  | Independent evidence converges on the same claim |
| Contrastive        | `contradictive` | Opposing or limiting positions |
| Triangulation      | `modifying`     | Conditions, exceptions, or scope refinement |
