import os
import csv

# Hugging Face token — set here or via HF_TOKEN environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Paths
DATASET_ROOT     = 'ArgMining-2026-UZH-Shared-Task'
INPUT_BASE_PATH  = os.path.join(DATASET_ROOT, 'train-data')
TEST_BASE_PATH   = os.path.join(DATASET_ROOT, 'test-data')
OUTPUT_BASE_PATH = '/home/sohom/Desktop/Neuroscience/ArgMining/Archive/output_qwen'
TAG_VOCAB_PATH   = os.path.join(DATASET_ROOT, 'education_dimensions_updated.csv')

# ---------------------------------------------------------------------------
# Model Configuration — Qwen3-8B-GGUF Q8_0 via llama-cpp-python
#
# Install with CUDA support:
#   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
#
# The model is downloaded automatically from HuggingFace on first run
# and cached in ~/.cache/huggingface/hub/
# ---------------------------------------------------------------------------
MODEL_REPO_ID  = "Qwen/Qwen3-8B-GGUF"
MODEL_FILENAME = "*Q8_0.gguf"   # glob pattern — matches the Q8_0 file in the repo
N_GPU_LAYERS   = -1             # -1 = offload all layers to GPU (RTX 4080 16 GB fits Q8_0)
N_CTX          = 8192           # context window in tokens

# Kept for compatibility with evaluate.py (which still uses the Anthropic judge)
STEP1_MODEL = "qwen3-8b-q8_0"
STEP2_MODEL = "qwen3-8b-q8_0"
JUDGE_MODEL = "claude-sonnet-4-6"

# Scouting
SCOUT_TOP_K     = 15
SCOUT_MIN_SCORE = 0.03

# Token budgets (max tokens the model may generate per call)
MAX_TOKENS_STEP1 = 2048
MAX_TOKENS_STEP2 = 4096

# ---------------------------------------------------------------------------
# Tag vocabulary — loaded from education_dimensions_updated.csv
# ---------------------------------------------------------------------------

def _load_tag_vocab(csv_path: str) -> dict:
    """Return {code: "Dimension — Category"} for all non-NA codes."""
    vocab = {}
    if not os.path.exists(csv_path):
        return vocab
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            code = row.get('CODE', '').strip()
            if not code or code == 'NA':
                continue
            dim = row.get('Dimensions', '').strip()
            cat = row.get('Categories', '').strip()
            vocab[code] = f"{dim} — {cat}"
    return vocab


def _build_tag_prompt_block(vocab: dict) -> str:
    """
    Format the tag vocabulary grouped by dimension for use in a prompt.
    Each dimension on one line: [Dimension] CODE: Category | CODE: Category | ...
    """
    if not vocab:
        return "(tag vocabulary unavailable — infer from content)"

    from collections import defaultdict
    dims: dict = defaultdict(list)
    for code, label in vocab.items():
        dim = label.split(' — ')[0]
        cat = label.split(' — ', 1)[1] if ' — ' in label else label
        dims[dim].append(f"{code}: {cat}")

    lines = []
    for dim, entries in dims.items():
        lines.append(f"[{dim}]  {' | '.join(entries)}")
    return '\n'.join(lines)


TAG_VOCAB   = _load_tag_vocab(TAG_VOCAB_PATH)
TAG_CODES   = sorted(TAG_VOCAB.keys())
_TAG_BLOCK  = _build_tag_prompt_block(TAG_VOCAB)

# ---------------------------------------------------------------------------
# Relation type <-> Reasoning strategy mappings (Section 2.3)
# ---------------------------------------------------------------------------
RELATION_TO_STRATEGY = {
    "supporting":    "Causal",
    "complemental":  "Corroboration",
    "contradictive": "Contrastive",
    "modifying":     "Triangulation",
}
STRATEGY_TO_RELATION  = {v: k for k, v in RELATION_TO_STRATEGY.items()}

RELATION_TYPES        = list(RELATION_TO_STRATEGY.keys())
REASONING_STRATEGIES  = list(STRATEGY_TO_RELATION.keys())

# ---------------------------------------------------------------------------
# Step 1 Prompt — Subtask 1: Paragraph Classification
# ---------------------------------------------------------------------------
STEP1_PROMPT = (
    "You are an expert in UN and UNESCO resolution analysis and argumentation mining.\n\n"
    "TASK\n"
    "Analyse the provided paragraph from a UN/UNESCO resolution document.\n"
    "Perform two classifications:\n\n"
    "SPECIAL CASE — ADMINISTRATIVE PARAGRAPHS\n"
    "Some paragraphs are non-argumentative metadata: document headers, agenda item titles,\n"
    "date/language stamps, institutional names alone (e.g. 'The Human Rights Council,'), or\n"
    "sentence fragments that continue a prior line. If the paragraph contains NO substantive\n"
    "claim, policy content, or normative statement, classify it as preambular, assign NO tags,\n"
    "and note in think that it is an administrative/structural marker.\n\n"
    "1. PARAGRAPH TYPE — classify as exactly one of:\n"
    "   'preambular': appears before operative section; opens with contextual/rationale keywords\n"
    "     such as Recalling, Recognizing, Noting, Reaffirming, Aware, Considering, Convinced,\n"
    "     Welcoming, Emphasizing, Bearing in mind, Having regard to, etc.\n"
    "   'operative': forms the main decisions/actions; opens with directive keywords\n"
    "     such as Requests, Urges, Calls upon, Decides, Recommends, Encourages, Invites,\n"
    "     Affirms, Declares, Instructs, Authorizes, etc.\n\n"
    "2. TAG ASSIGNMENT — assign ALL applicable codes from the vocabulary below.\n"
    "   Use ONLY the exact codes listed (e.g. ISC_1, POL_EQUIT, T_PDEV).\n"
    "   Assign every code that genuinely applies; omit codes that do not apply.\n"
    "   Do NOT invent codes or use free-form text.\n\n"
    f"TAG VOCABULARY\n{_TAG_BLOCK}\n\n"
    "REASONING STRATEGIES — apply the ONE most dominant strategy:\n"
    "- Causal: paragraph establishes an explicit cause → effect or problem → remedy chain\n"
    "- Corroboration: paragraph stacks multiple DISTINCT sources/treaties/actors all reinforcing\n"
    "  the SAME specific claim (not just listing topics)\n"
    "- Triangulation: THREE OR MORE genuinely different conceptual angles converge on ONE\n"
    "  precise central claim — not just 'multiple things appear'; the convergence must be tight\n"
    "- Contrastive: paragraph names a specific tension, opposition, or risk that it argues against\n\n"
    "THINK FIELD — write four numbered sentences:\n"
    "1. TYPE: Quote the exact opening keyword and explain what it signals about paragraph function.\n"
    "2. TAGS: For each tag, name the exact phrase in the paragraph that justifies it.\n"
    "3. STRATEGY: Name the ONE dominant strategy and show the specific logical mechanism at work.\n"
    "4. SIGNIFICANCE: Explain the role this paragraph plays in the resolution's overall argument\n"
    "   (e.g. establishes the factual basis for later operative demands, sets the normative scope,\n"
    "   narrows a prior commitment to a specific population, etc.).\n\n"
    "EXAMPLE THINK FIELD\n"
    "\"1. TYPE: 'Recalling' marks this as preambular — it invokes prior authority as normative\n"
    "grounding rather than issuing a directive. 2. TAGS: 'right to freedom of opinion' → CCUT__RIGH;\n"
    "'resolution 7/36' and 'all previous resolutions of the Commission on Human Rights' → LAW_INTER\n"
    "(named international instruments); 'Human Rights Council' as institutional actor → ACT_IO.\n"
    "3. STRATEGY: Corroboration — two distinct citation types (a specific dated resolution and the\n"
    "full historical chain of Commission resolutions) both reinforce the same normative claim,\n"
    "strengthening it through cumulative institutional authority. 4. SIGNIFICANCE: This paragraph\n"
    "establishes the continuity of the resolution with prior HRC and Commission mandates, providing\n"
    "the institutional legitimacy that operative paragraphs later depend on when requesting the\n"
    "Special Rapporteur to act.\"\n\n"
    "OUTPUT FORMAT (strictly valid JSON, no markdown fences)\n"
    "{\n"
    '  "paragraph_idx": <int>,\n'
    '  "paragraph_type": "preambular" | "operative",\n'
    '  "tags": ["CODE1", "CODE2"],\n'
    '  "think": "<four-sentence think field as described above>"\n'
    "}\n"
)

# ---------------------------------------------------------------------------
# Step 2 Prompt — Subtask 2: Argumentative Relation Prediction
# ---------------------------------------------------------------------------
STEP2_PROMPT = (
    "You are an expert in argumentative structure analysis of UN/UNESCO resolutions.\n\n"
    "TASK\n"
    "Given a SOURCE paragraph and CANDIDATE paragraphs from the same resolution, predict which\n"
    "candidates have a genuine argumentative relation with the source and label each relation.\n\n"
    "SKIP ADMINISTRATIVE PARAGRAPHS\n"
    "If the SOURCE is a metadata header, agenda title, date stamp, institutional name, or sentence\n"
    "fragment with no substantive argumentative content — output no relations at all.\n\n"
    "RELATION TYPE DEFINITIONS (lowercase exactly; assign all that apply per pair)\n"
    "- supporting:    one paragraph provides the factual, normative, or contextual PREMISE that\n"
    "                 directly necessitates or justifies the claim/action in the other.\n"
    "                 Test: remove SOURCE — does TARGET lose its justification? If yes: supporting.\n"
    "- complemental:  both paragraphs independently assert THE SAME specific claim but via\n"
    "                 different actors, evidence, or mechanisms. They are mutually reinforcing.\n"
    "                 Test: do they say the same thing in genuinely different ways? If yes: complemental.\n"
    "                 WARNING: shared topic alone is NOT enough. Two paragraphs about 'human rights'\n"
    "                 are not complemental unless they assert the same specific sub-claim.\n"
    "- contradictive: one paragraph asserts a position the other explicitly limits or opposes.\n"
    "                 Test: is there a real logical tension, not just a difference in emphasis?\n"
    "- modifying:     one paragraph adds exceptions, conditions, timelines, or beneficiary\n"
    "                 constraints that scope or refine the other.\n"
    "                 Test: does one narrow or condition the other's applicability?\n\n"
    "REASONING STRATEGY → RELATION TYPE\n"
    "  Causal        → supporting\n"
    "  Corroboration → complemental\n"
    "  Contrastive   → contradictive\n"
    "  Triangulation → modifying\n\n"
    "THINK FIELD — write five numbered sentences per relation:\n"
    "1. SOURCE PHRASE: Quote the specific phrase from SOURCE that drives this relation.\n"
    "2. TARGET PHRASE: Quote the specific phrase from TARGET that connects with it.\n"
    "3. RELATION: Name the relation type and show exactly how the two phrases instantiate it.\n"
    "4. STRATEGY: Name the reasoning strategy and the precise logical step it captures.\n"
    "5. SIGNIFICANCE: Explain what this argumentative link contributes to the resolution's\n"
    "   overall reasoning structure (e.g. 'this link shows the operative mandate is grounded in\n"
    "   a documented rights gap, preventing the request from appearing arbitrary').\n"
    "Also briefly note one relation type you considered and rejected, and why.\n\n"
    "EXAMPLE THINK FIELD\n"
    "\"1. SOURCE: 'Expressing deep concern that violations...continue to occur, including more\n"
    "frequent attacks and killings of journalists.' 2. TARGET: 'Requests the Special Rapporteur\n"
    "to continue reporting on violations of the right to freedom of expression.' 3. RELATION:\n"
    "supporting — the documented pattern of violations in SOURCE is the direct evidentiary premise\n"
    "that necessitates the monitoring mandate in TARGET; without the diagnosed problem, the request\n"
    "would lack justification. 4. STRATEGY: Causal — SOURCE identifies a persistent harm\n"
    "(cause), TARGET responds with an institutional remedy (effect). 5. SIGNIFICANCE: This link\n"
    "is structurally load-bearing — it grounds the operative mandate in empirical evidence,\n"
    "transforming a discretionary request into a normatively required institutional response.\n"
    "Complemental rejected: the paragraphs do not assert the same claim — one diagnoses, one\n"
    "prescribes.\"\n\n"
    "OUTPUT FORMAT (strictly valid JSON, no markdown fences)\n"
    "{\n"
    '  "source_idx": <int>,\n'
    '  "relations": [\n'
    "    {\n"
    '      "target_idx": <int>,\n'
    '      "relation_types": ["supporting"],\n'
    '      "reasoning_strategies": ["Causal"],\n'
    '      "think": "<five-sentence think field as described above>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Only include pairs where the argumentative connection is substantive and clear.\n"
    "Fewer high-confidence relations are better than many weak ones.\n"
)
