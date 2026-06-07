import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import re
import argparse
from typing import List, Dict, Optional, Tuple

from huggingface_hub import login as hf_login
from llama_cpp import Llama
from json_repair import repair_json

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_TFIDF = True
except ImportError:
    HAS_TFIDF = False

from config_qwen import (
    INPUT_BASE_PATH, TEST_BASE_PATH, OUTPUT_BASE_PATH,
    HF_TOKEN, MODEL_REPO_ID, MODEL_FILENAME, N_GPU_LAYERS, N_CTX,
    STEP1_MODEL, STEP2_MODEL, STEP1_PROMPT, STEP2_PROMPT,
    SCOUT_TOP_K, SCOUT_MIN_SCORE,
    MAX_TOKENS_STEP1, MAX_TOKENS_STEP2,
    RELATION_TO_STRATEGY,
)


class Colors:
    RED    = '\033[31m'
    GREEN  = '\033[32m'
    YELLOW = '\033[33m'
    BLUE   = '\033[34m'
    CYAN   = '\033[36m'
    RESET  = '\033[0m'


# ---------------------------------------------------------------------------
# Model initialisation — downloaded from HuggingFace on first run,
# then cached at ~/.cache/huggingface/hub/
# ---------------------------------------------------------------------------

if HF_TOKEN:
    hf_login(token=HF_TOKEN, add_to_git_credential=False)

print(f'{Colors.CYAN}Loading {MODEL_REPO_ID} ({MODEL_FILENAME}) …{Colors.RESET}')
_llm = Llama.from_pretrained(
    repo_id=MODEL_REPO_ID,
    filename=MODEL_FILENAME,
    n_gpu_layers=N_GPU_LAYERS,
    n_ctx=N_CTX,
    verbose=False,
)
print(f'{Colors.GREEN}Model ready.{Colors.RESET}')


def parse_json_response(text: str) -> Optional[Dict]:
    """Extract and parse JSON from model response, repairing if malformed."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    raw = match.group()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(repair_json(raw))
        except Exception:
            return None


def _extract_text(response) -> str:
    """
    Extract the text content from a llama-cpp chat completion response.
    Strips any Qwen3 <think>…</think> blocks that may appear even when
    thinking is nominally disabled (defence-in-depth).
    """
    text = response["choices"][0]["message"]["content"] or ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text


def _chat(system: str, user: str, max_tokens: int) -> str:
    """
    Single helper that wraps llama-cpp's create_chat_completion.

    /no_think is prepended to the user message so Qwen3 skips its internal
    chain-of-thought and outputs only the JSON we need.
    """
    response = _llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"/no_think\n{user}"},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        repeat_penalty=1.1,
    )
    return _extract_text(response)


# ---------------------------------------------------------------------------
# Scouting Module — sentence-transformers with TF-IDF fallback
# ---------------------------------------------------------------------------

def build_index(paragraphs: List[str]):
    """
    Build a semantic index over paragraph texts.
    Priority: sentence-transformers > TF-IDF > None (positional fallback).
    Returns a (kind, matrix) tuple or None.
    """
    if len(paragraphs) < 2:
        return None

    if HAS_ST:
        embeddings = _ST_MODEL.encode(paragraphs, convert_to_numpy=True, show_progress_bar=False)
        return ('st', embeddings)

    if HAS_TFIDF and HAS_SKLEARN:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        return ('tfidf', vectorizer.fit_transform(paragraphs))

    return None


def scout_candidates(
    para_idx: int,
    paragraphs: List[str],
    index,
    top_k: int = SCOUT_TOP_K,
    min_score: float = SCOUT_MIN_SCORE,
) -> List[int]:
    """
    Return indices of paragraphs most semantically related to paragraphs[para_idx].
    Falls back to positional window when no index is available.
    """
    if index is None or not HAS_SKLEARN:
        others = [i for i in range(len(paragraphs)) if i != para_idx]
        return others[:top_k]

    kind, matrix = index
    if kind == 'st':
        query = matrix[para_idx:para_idx + 1]
    else:
        query = matrix[para_idx]

    scores = cosine_similarity(query, matrix).flatten()
    scores[para_idx] = -1.0
    ranked = np.argsort(scores)[::-1]
    return [int(i) for i in ranked if scores[i] >= min_score][:top_k]


# ---------------------------------------------------------------------------
# Step 1 — Subtask 1: Paragraph Classification (type + tags)
# ---------------------------------------------------------------------------

def classify_paragraph(
    para_idx: int,
    para_text: str,
    doc_paragraphs: List[str],
) -> Dict:
    """Classify one paragraph as preambular/operative and assign tags."""
    start = max(0, para_idx - 8)
    end   = min(len(doc_paragraphs), para_idx + 13)
    context_window = doc_paragraphs[start:end]
    context_preview = "\n\n".join(
        f"[{i}] {p[:300]}" for i, p in enumerate(context_window, start=start)
    )
    user_text = (
        f"RESOLUTION CONTEXT (paragraphs {start}–{end - 1}):\n"
        f"{context_preview}\n\n"
        f"TARGET PARAGRAPH [{para_idx}]:\n{para_text}"
    )

    try:
        text   = _chat(STEP1_PROMPT, user_text, MAX_TOKENS_STEP1)
        parsed = parse_json_response(text)
        if parsed:
            parsed["paragraph_idx"] = para_idx
            return parsed
        return {"paragraph_idx": para_idx, "paragraph_type": "operative", "tags": [], "think": ""}
    except Exception as e:
        print(f'{Colors.RED}Step 1 error [para {para_idx}]: {e}{Colors.RESET}')
        return {"paragraph_idx": para_idx, "paragraph_type": "operative", "tags": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Step 2 — Subtask 2: Argumentative Relation Prediction
# ---------------------------------------------------------------------------

def predict_relations(
    para_idx: int,
    para_text: str,
    candidate_indices: List[int],
    all_paragraphs: List[str],
    subtask1_results: Dict[int, Dict],
) -> Dict:
    """Predict relation types between source paragraph and candidate paragraphs."""
    if not candidate_indices:
        return {"source_idx": para_idx, "relations": []}

    src_meta = subtask1_results.get(para_idx, {})
    candidates_text = "\n\n".join(
        f"[{i}] type={subtask1_results.get(i, {}).get('paragraph_type', '?')} "
        f"tags={subtask1_results.get(i, {}).get('tags', [])[:5]}\n"
        f"{all_paragraphs[i][:400]}"
        for i in candidate_indices
    )
    user_text = (
        f"SOURCE PARAGRAPH [{para_idx}] "
        f"type={src_meta.get('paragraph_type','?')} "
        f"tags={src_meta.get('tags', [])[:5]}\n"
        f"{para_text}\n\n"
        f"CANDIDATE PARAGRAPHS:\n{candidates_text}"
    )

    try:
        text   = _chat(STEP2_PROMPT, user_text, MAX_TOKENS_STEP2)
        parsed = parse_json_response(text)
        if parsed:
            parsed["source_idx"] = para_idx
            for rel in parsed.get("relations", []):
                if not rel.get("reasoning_strategies"):
                    rel["reasoning_strategies"] = [
                        RELATION_TO_STRATEGY[rt]
                        for rt in rel.get("relation_types", [])
                        if rt in RELATION_TO_STRATEGY
                    ]
            return parsed
        return {"source_idx": para_idx, "relations": []}
    except Exception as e:
        print(f'{Colors.RED}Step 2 error [para {para_idx}]: {e}{Colors.RESET}')
        return {"source_idx": para_idx, "relations": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Think Field Generation — template-based fallback (System A style)
# ---------------------------------------------------------------------------

def generate_think_subtask1(para_idx: int, result: Dict) -> str:
    """Template think field for Subtask 1 when the LLM did not provide one."""
    ptype = result.get("paragraph_type", "?")
    tags  = result.get("tags", [])
    ptype_explanation = (
        "the opening keyword indicates a contextual/rationale framing typical of preambular text."
        if ptype == "preambular"
        else "the paragraph contains directive language typical of operative decisions."
    )
    return (
        f"Paragraph {para_idx} is classified as '{ptype}': {ptype_explanation} "
        f"Assigned tags: {', '.join(tags) if tags else 'none'}. "
        "These tags reflect the primary argumentative functions and thematic content."
    )


def generate_think_subtask2(
    src_idx: int,
    tgt_idx: int,
    relation_types: List[str],
    subtask1_results: Dict[int, Dict],
) -> str:
    """Template think field for Subtask 2 when the LLM did not provide one."""
    src = subtask1_results.get(src_idx, {})
    tgt = subtask1_results.get(tgt_idx, {})
    strategies = [RELATION_TO_STRATEGY[rt] for rt in relation_types if rt in RELATION_TO_STRATEGY]

    _strategy_explanation = {
        "supporting":    f"{src_idx}'s content causally grounds or justifies {tgt_idx}'s framing.",
        "complemental":  f"{src_idx} and {tgt_idx} corroborate the same point from different angles.",
        "contradictive": f"{src_idx} expresses a position that contrasts with or opposes {tgt_idx}.",
        "modifying":     f"{src_idx} triangulates and refines or qualifies the claim in {tgt_idx}.",
    }

    parts = [
        f"Paragraph {src_idx} is {src.get('paragraph_type','?')} "
        f"[tags: {', '.join(src.get('tags', [])[:5])}].",
        f"Paragraph {tgt_idx} is {tgt.get('paragraph_type','?')} "
        f"[tags: {', '.join(tgt.get('tags', [])[:5])}].",
        f"Predicted relation type(s): {', '.join(relation_types)}.",
        f"Reverse-mapped reasoning strategy/strategies: {', '.join(strategies)}.",
    ]
    for rt in relation_types:
        if rt in _strategy_explanation:
            strat = RELATION_TO_STRATEGY.get(rt, "")
            parts.append(f"{rt} ({strat}): {_strategy_explanation[rt]}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Data loading — train-data schema
# ---------------------------------------------------------------------------

_SKIP_TEXT = {"<!-- image -->"}
_MIN_PARA_LEN = 20


def load_paragraphs(json_path: str) -> Tuple[str, List[str]]:
    """
    Load a train-data resolution JSON and return (doc_id, paragraph_texts).

    Train-data schema: top-level list of items, each with:
      type       — "heading" | "paragraph" | "list"
      text_en    — English text  (paragraph / heading items)
      items_en   — list of str   (list items — operative clauses)

    Extraction rules:
    - Skip "heading" items entirely.
    - For "paragraph": use text_en, skip if it's an image placeholder or too short.
    - For "list": emit each items_en entry as a separate paragraph.
    """
    with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)

    doc_id = os.path.splitext(os.path.basename(json_path))[0]
    doc_id = doc_id.replace('-fr-parsed', '')

    texts: List[str] = []
    for item in data:
        item_type = item.get("type", "")
        if item_type == "heading":
            continue
        if item_type == "paragraph":
            text = (item.get("text_en") or "").strip()
            if text and text not in _SKIP_TEXT and len(text) >= _MIN_PARA_LEN:
                texts.append(text)
        elif item_type == "list":
            for entry in item.get("items_en") or []:
                entry = entry.strip()
                if entry and len(entry) >= _MIN_PARA_LEN:
                    texts.append(entry)

    return doc_id, texts


def load_test_doc(json_path: str) -> Tuple[str, Dict, List[str], List[int]]:
    """
    Load a test-data resolution JSON.

    Test-data schema: {TEXT_ID, RECOMMENDATION, TITLE, METADATA, body.paragraphs[]}
    Each paragraph has: {para_number, para, para_en, tags, type, think, matched_pars}

    Returns (text_id, doc_data, paragraph_texts, para_numbers).
    """
    with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
        doc_data = json.load(f)

    text_id = doc_data.get("TEXT_ID", os.path.splitext(os.path.basename(json_path))[0])
    raw_paras = doc_data.get("body", {}).get("paragraphs", [])

    texts: List[str] = []
    para_numbers: List[int] = []
    for p in raw_paras:
        text = (p.get("para_en") or p.get("para") or "").strip()
        if text and len(text) >= _MIN_PARA_LEN:
            texts.append(text)
            para_numbers.append(p["para_number"])

    return text_id, doc_data, texts, para_numbers


# ---------------------------------------------------------------------------
# Output formatting — official submission schema
# ---------------------------------------------------------------------------

def _build_output(
    doc_id: str,
    paragraphs: List[str],
    para_numbers: List[int],
    subtask1_results: Dict[int, Dict],
    subtask2_results: List[Dict],
    doc_data: Optional[Dict] = None,
) -> Dict:
    """Build the official submission JSON schema from pipeline results."""
    rel_lookup: Dict[int, List[Tuple]] = {}
    for entry in subtask2_results:
        src = entry["source_idx"]
        rel_lookup.setdefault(src, [])
        for rel in entry.get("relations", []):
            tgt = rel["target_idx"]
            rtypes = rel.get("relation_types", [])
            rthink = rel.get("think", "")
            if rtypes:
                rel_lookup[src].append((tgt, rtypes, rthink))

    preambular_para: List[int] = []
    operative_para:  List[int] = []
    body_paragraphs: List[Dict] = []

    for idx, para_text in enumerate(paragraphs):
        pnum   = para_numbers[idx]
        s1     = subtask1_results.get(idx, {})
        ptype  = s1.get("paragraph_type", "operative")
        tags   = s1.get("tags", [])
        s1_think = s1.get("think", "")

        if ptype == "preambular":
            preambular_para.append(pnum)
        else:
            operative_para.append(pnum)

        matched_pars: Dict[str, List[str]] = {}
        rel_think_parts: List[str] = []
        for tgt_idx, rtypes, rthink in rel_lookup.get(idx, []):
            if tgt_idx < len(para_numbers):
                tgt_pnum = para_numbers[tgt_idx]
                matched_pars[str(tgt_pnum)] = rtypes
                if rthink:
                    rel_think_parts.append(f"[→ para {tgt_pnum}] {rthink}")

        combined_think = s1_think
        if rel_think_parts:
            combined_think += "\n\n" + "\n\n".join(rel_think_parts)

        if doc_data:
            orig_paras = doc_data.get("body", {}).get("paragraphs", [])
            orig = next((p for p in orig_paras if p["para_number"] == pnum), {})
            para_orig = orig.get("para", para_text)
            para_en   = orig.get("para_en", para_text)
        else:
            para_orig = para_text
            para_en   = para_text

        body_paragraphs.append({
            "para_number": pnum,
            "para":        para_orig,
            "para_en":     para_en,
            "type":        ptype,
            "tags":        tags,
            "matched_pars": matched_pars,
            "think":       combined_think,
        })

    n_pre = len(preambular_para)
    n_op  = len(operative_para)
    n_rel = sum(len(v) for v in rel_lookup.values())
    meta_think = (
        f"Document structure analysis: {len(paragraphs)} total paragraphs — "
        f"{n_pre} preambular (normative/contextual framing) and "
        f"{n_op} operative (directive actions); "
        f"{n_rel} argumentative relations identified. "
        f"Preambular paragraphs establish the resolution's rationale through treaties, "
        f"prior resolutions, and documented concerns. "
        f"Operative paragraphs issue specific mandates, requests, and recommendations "
        f"grounded in the preambular evidence base."
    )

    if doc_data:
        text_id      = doc_data.get("TEXT_ID", doc_id)
        recommendation = doc_data.get("RECOMMENDATION")
        title        = doc_data.get("TITLE", "")
        doc_title    = doc_data.get("METADATA", {}).get("structure", {}).get("doc_title", doc_id)
    else:
        text_id      = doc_id
        recommendation = None
        title        = ""
        doc_title    = doc_id

    output: Dict = {
        "TEXT_ID": text_id,
        "METADATA": {
            "structure": {
                "doc_title":        doc_title,
                "nb_paras":         len(paragraphs),
                "preambular_para":  preambular_para,
                "operative_para":   operative_para,
                "think":            meta_think,
            }
        },
        "body": {
            "paragraphs": body_paragraphs,
        },
    }
    if recommendation is not None:
        output["RECOMMENDATION"] = recommendation
    if title:
        output["TITLE"] = title

    return output


# ---------------------------------------------------------------------------
# Resolution-level orchestration
# ---------------------------------------------------------------------------

def _output_exists(doc_id: str) -> bool:
    path = os.path.join(OUTPUT_BASE_PATH, f"{doc_id}_predictions.json")
    return os.path.exists(path)


def _process_doc(
    doc_id: str,
    paragraphs: List[str],
    para_numbers: Optional[List[int]] = None,
    doc_data: Optional[Dict] = None,
) -> bool:
    """Core pipeline: classify paragraphs + predict relations + save output."""
    print(f'\n{Colors.CYAN}{"="*60}{Colors.RESET}')
    print(f'{Colors.CYAN}Processing: {doc_id}{Colors.RESET}')
    print(f'{Colors.CYAN}{"="*60}{Colors.RESET}')

    if not paragraphs:
        print(f'{Colors.RED}No paragraphs — skipping.{Colors.RESET}')
        return False

    if para_numbers is None:
        para_numbers = list(range(1, len(paragraphs) + 1))

    print(f'{Colors.GREEN}Paragraphs: {len(paragraphs)}{Colors.RESET}')
    index = build_index(paragraphs)
    if HAS_ST:
        print(f'{Colors.GREEN}Scouting: sentence-transformers (all-MiniLM-L6-v2){Colors.RESET}')
    elif HAS_TFIDF:
        print(f'{Colors.YELLOW}Scouting: TF-IDF (install sentence-transformers for better results){Colors.RESET}')
    else:
        print(f'{Colors.YELLOW}Scouting: positional fallback (install scikit-learn){Colors.RESET}')

    # Step 1 — Subtask 1
    subtask1_results: Dict[int, Dict] = {}
    print(f'\n{Colors.YELLOW}Step 1: Classifying {len(paragraphs)} paragraphs...{Colors.RESET}')
    for idx, para_text in enumerate(paragraphs):
        result = classify_paragraph(idx, para_text, paragraphs)
        if not result.get("think"):
            result["think"] = generate_think_subtask1(idx, result)
        subtask1_results[idx] = result
        print(
            f'  {Colors.BLUE}[{idx:>3}]{Colors.RESET} '
            f'{result.get("paragraph_type","?"):12s} | '
            f'tags: {len(result.get("tags", []))}'
        )

    # Step 2 — Subtask 2
    subtask2_results: List[Dict] = []
    print(f'\n{Colors.YELLOW}Step 2: Predicting relations...{Colors.RESET}')
    for idx, para_text in enumerate(paragraphs):
        if not subtask1_results.get(idx, {}).get("tags"):
            print(f'  {Colors.YELLOW}[{idx:>3}] skipped (non-argumentative){Colors.RESET}')
            continue
        candidates = scout_candidates(idx, paragraphs, index)
        candidates = [c for c in candidates if subtask1_results.get(c, {}).get("tags")]
        if not candidates:
            continue
        rel_result = predict_relations(idx, para_text, candidates, paragraphs, subtask1_results)
        for rel in rel_result.get("relations", []):
            if not rel.get("think"):
                rel["think"] = generate_think_subtask2(
                    idx, rel["target_idx"], rel.get("relation_types", []), subtask1_results
                )
        if rel_result.get("relations"):
            subtask2_results.append(rel_result)
            print(f'  {Colors.BLUE}[{idx:>3}]{Colors.RESET} → {len(rel_result["relations"])} relation(s)')

    output = _build_output(
        doc_id, paragraphs, para_numbers,
        subtask1_results, subtask2_results, doc_data,
    )

    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    output_path = os.path.join(OUTPUT_BASE_PATH, f"{doc_id}_predictions.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n{Colors.GREEN}Saved → {output_path}{Colors.RESET}')
    return True


def _run_on(
    doc_id: str,
    paragraphs: List[str],
    stats: Dict,
    para_numbers: Optional[List[int]] = None,
    doc_data: Optional[Dict] = None,
):
    """Process one document and update stats in-place."""
    if _output_exists(doc_id):
        print(f'  {Colors.GREEN}[DONE]    {doc_id}{Colors.RESET}')
        stats["skipped"] += 1
        return
    try:
        if _process_doc(doc_id, paragraphs, para_numbers=para_numbers, doc_data=doc_data):
            stats["successful"] += 1
        else:
            stats["failed"] += 1
    except Exception as e:
        print(f'{Colors.RED}Error — {doc_id}: {e}{Colors.RESET}')
        stats["failed"] += 1


def main():
    parser = argparse.ArgumentParser(description="ArgMining Resolution Pipeline (Qwen3-8B-GGUF)")
    parser.add_argument(
        "--input", default=None,
        help="Folder containing input JSON files (overrides --test default path)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most N documents (0 = no limit, useful for quick tests)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Process test-data files instead of train-data",
    )
    args = parser.parse_args()

    if args.input:
        input_dir = args.input
    elif args.test:
        input_dir = TEST_BASE_PATH
    else:
        input_dir = INPUT_BASE_PATH

    print(f'{Colors.CYAN}ArgMining Resolution Pipeline — Qwen3-8B-GGUF Q8_0{Colors.RESET}')
    print(f'{Colors.CYAN}Mode:   {"test" if args.test else "train"}{Colors.RESET}')
    print(f'{Colors.CYAN}Input:  {input_dir}{Colors.RESET}')
    print(f'{Colors.CYAN}Output: {OUTPUT_BASE_PATH}{Colors.RESET}')
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f'{Colors.RED}Input folder not found: {input_dir}{Colors.RESET}')
        return

    json_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.json')
    ])
    if not json_files:
        print(f'{Colors.RED}No JSON files found in {input_dir}{Colors.RESET}')
        return

    if args.limit:
        json_files = json_files[:args.limit]

    print(f'{Colors.YELLOW}Found {len(json_files)} files to process{Colors.RESET}\n')
    stats = {"successful": 0, "failed": 0, "skipped": 0}

    for json_path in json_files:
        if args.test:
            text_id, doc_data, paragraphs, para_numbers = load_test_doc(json_path)
            _run_on(text_id, paragraphs, stats, para_numbers=para_numbers, doc_data=doc_data)
        else:
            doc_id, paragraphs = load_paragraphs(json_path)
            _run_on(doc_id, paragraphs, stats)

    print(f'\n{Colors.CYAN}{"="*60}{Colors.RESET}')
    print(
        f'{Colors.CYAN}DONE  |  '
        f'Success: {stats["successful"]}  |  '
        f'Failed: {stats["failed"]}  |  '
        f'Skipped: {stats["skipped"]}{Colors.RESET}'
    )
    print(f'{Colors.CYAN}{"="*60}{Colors.RESET}')


if __name__ == "__main__":
    main()
