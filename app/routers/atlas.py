from __future__ import annotations

# stdlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional, Iterable
from uuid import uuid4
from collections import Counter, defaultdict
import math, re, json, logging

# third-party
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud import firestore
from rapidfuzz import process, fuzz
import httpx
from fastapi.responses import StreamingResponse
import asyncio, threading
from google.api_core.exceptions import FailedPrecondition

# local
from app.core.security import decode_token
from app.core.config import settings


# ==============================================================================
# Config / Globals
# ==============================================================================

router = APIRouter(prefix="/api/atlas", tags=["Atlas"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Firestore (async)
db = AsyncClient()

# Logging
logger = logging.getLogger("atlas")
logger.setLevel(logging.INFO)

# Collections / IDs
SYLLABI_COLL = "syllabi"
POINTS_INDEX_COLL = "points_index"
SYLLABUS_DOC_ID = "edexcel_gcse_physics_issue4"
COLL = "atlas_sessions"
FLASHCARDS_COLL = "atlas_flashcard_sets"  # <-- NEW

# Models
OPENAI_MODEL = "gpt-5-nano"         # answers & small rerank
OPENAI_RERANK_MODEL = OPENAI_MODEL  # can split later

# Embeddings are optional and OFF by default (corpus-only accuracy first)
EMBEDDINGS_ENABLED = False
EMBEDDING_MODEL = "text-embedding-3-small"

# ------------------------------------------------------------------------------
# OpenAI client helpers (fixes NameError and centralizes readiness)
# ------------------------------------------------------------------------------

def _sse(event: str, data: Any) -> bytes:
    """Pack a Server-Sent Event {event, data} as bytes."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")

def _chunk_text(s: str, n: int = 24):
    """Fallback: chunk plain text in ~n chars for pseudo-streaming."""
    for i in range(0, len(s), n):
        yield s[i:i+n]

try:
    from openai import OpenAI
    _oai_import_ok = True
except Exception:
    _oai_import_ok = False

def _oai_ready() -> bool:
    """Return True if OpenAI client is importable and an API key is set."""
    try:
        return _oai_import_ok and bool(settings.OPENAI_API_KEY)
    except Exception:
        return False

def _oai_client() -> "OpenAI":
    if not _oai_ready():
        raise RuntimeError("OpenAI not configured")
    return OpenAI(api_key=settings.OPENAI_API_KEY)

# A separate client we *might* use for embeddings; only created if needed
_emb_client: Optional["OpenAI"] = None
if EMBEDDINGS_ENABLED and _oai_ready():
    try:
        _emb_client = _oai_client()
    except Exception:
        _emb_client = None


def fix_latex(text: str) -> str:
    if not text:
        return text
    # Ensure bare 'log'/'ln' gain backslashes (but don't double-escape if already present)
    text = re.sub(r'(?<!\\)\blog\b', lambda m: r'\log', text)
    text = re.sub(r'(?<!\\)\bln\b',  lambda m: r'\ln',  text)

    # Fix subscripts/superscripts without braces: \log_b -> \log_{b}, \log^b -> \log^{b}
    text = re.sub(r'(\\log)_(\w)', lambda m: f'{m.group(1)}_{{{m.group(2)}}}', text)
    text = re.sub(r'(\\log)\^(\w)', lambda m: f'{m.group(1)}^{{{m.group(2)}}}', text)

    # Make change-of-base fraction nice (safe callable to keep backslashes literal)
    cob_pat = re.compile(
        r'\\log_{?(\w)}?\s*x\s*=\s*\\log_{?(\w)}?\s*x\s*/\s*\\log_{?(\w)}?\s*\2',
        re.IGNORECASE
    )
    text = cob_pat.sub(lambda m: r'\\log_{\1} x = \\dfrac{\\log_{\2} x}{\\log_{\2} \1}', text)

    return text


# ==============================================================================
# Schemas
# ==============================================================================

class UpdateSessionRequest(BaseModel):
    session_id: str
    new_title: str = Field(..., min_length=1, max_length=100)

class WeakArea(BaseModel):
    topic: str
    description: str
    severity: str = "medium"
    first_identified: str
    last_encountered: str
    session_ids: List[str]

class WeakAreasUpdate(BaseModel):
    weak_areas: List[WeakArea]

class WeakAreasResponse(BaseModel):
    weak_areas: List[WeakArea]


class Message(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str
    type: Optional[str] = None
    quiz: Optional[Dict[str, Any]] = None
    points: Optional[List[Dict[str, Any]]] = None
    flashcards: Optional[List[Flashcard]] = None  # was List[FlashcardItem]


class MessageIn(BaseModel):
    session_id: str
    role: str
    content: str
    type: Optional[str] = None
    quiz: Optional[Dict[str, Any]] = None
    points: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None
    meta_json: Optional[str] = None
    flashcards: Optional[List[Flashcard]] = None  # was List[FlashcardItem]

class CreateSessionResponse(BaseModel):
    session_id: str
    seed_messages: List[Message]
    remaining_messages: int = 20
    user_msg_limit: int = 20
    tutor_id: str
    tutor_title: Optional[str] = None


class CreateSessionIn(BaseModel):
    title: Optional[str] = None
    tutor_id: Optional[str] = None

class SessionOut(BaseModel):
    id: str
    messages: List[Message]
    title: Optional[str] = None
    updated_at: Optional[str] = None
    tutor_id: Optional[str] = None
    tutor_title: Optional[str] = None


class SessionsListItem(BaseModel):
    id: str
    title: Optional[str] = None
    last_message_preview: Optional[str] = None
    updated_at: Optional[str] = None
    can_edit: bool = False
    tutor_id: Optional[str] = None
    tutor_title: Optional[str] = None

class Tutor(BaseModel):
    id: str
    title: str

class TutorIn(BaseModel):
    id: str = Field(..., min_length=3, max_length=100)
    title: str = Field(..., min_length=3, max_length=200)

# ==========================
# Flashcards: Schemas
# ==========================

class FlashcardItem(BaseModel):
    q: str
    a: str
    # provenance (filled automatically server-side)
    sources: Dict[str, Any] = Field(default_factory=dict)   # {"message_ids":[...], "point_ids":[...], "matched": float}

class Flashcard(BaseModel):
    q: str = Field(..., min_length=1, max_length=400)
    a: str = Field(..., min_length=1, max_length=800)
    hint: Optional[str] = Field(default=None, max_length=300)
    cloze: Optional[bool] = False
    tags: Optional[List[str]] = []

class FlashcardSet(BaseModel):
    id: str
    user_email: str
    session_id: str
    tutor_id: Optional[str] = None
    tutor_title: Optional[str] = None
    title: str
    cards: List[Flashcard]
    card_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class FlashcardSetSummary(BaseModel):
    id: str
    session_id: str
    title: str
    card_count: int
    tutor_id: Optional[str] = None
    tutor_title: Optional[str] = None
    created_at: Optional[str] = None

class FlashcardGenerateIn(BaseModel):
    session_id: str
    num_cards: int = Field(default=10, ge=4, le=20)
    save_to_progress: bool = True

class FlashcardGenerateOut(BaseModel):
    set: FlashcardSet
    saved_to_progress: bool


DEFAULT_TUTOR_ID = "edexcel_gcse_physics_issue4"
DEFAULT_TUTOR_TITLE = "Edexcel GCSE Physics"


# ==============================================================================
# Auth helper
# ==============================================================================

def get_current_email(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = decode_token(token)  # {"sub": email}
        email = payload.get("sub")
        if not email:
            raise ValueError("no sub")
        return email
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ==============================================================================
# Search engine: BM25F + Char-3gram + Anchor lexicon + Topic router + SDM
#   - No human synonyms. Purely corpus-derived signals.
# ==============================================================================

STOP_WORDS = {
    "what","is","the","a","an","of","to","in","and","or","with","that",
    "describe","explain","use","carry","out","how","why","be","able","including",
    "for","on","into","from","by","as","it","their","any","this","these","those",
    "meant","mean","does","do","you","we","i","they","them","his","her"
}

DEFAULT_STYLE = "tight"

def _norm_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = (t.replace("Δ"," delta ").replace("θ"," theta ").replace("×"," x ")
           .replace("∙"," x ").replace("·"," x ").replace("⋅"," x ")
           .replace("÷"," / ").replace("–","-").replace("—","-")
           .replace("/", " per "))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _tokenize(text: str) -> List[str]:
    text = _norm_text(text)
    toks = re.findall(r"[a-z0-9=]+", text)
    return [t for t in toks if t not in STOP_WORDS]

def _char_ngrams(text: str, n: int = 3) -> set:
    s = re.sub(r"\s+", " ", _norm_text(text))
    s = f"  {s}  "
    return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))

def _norm_scores(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return {}
    mx = max(d.values()) or 1.0
    return {k: (v / mx) for k, v in d.items()}

def _l2norm(v: List[float]) -> List[float]:
    n2 = sum(x*x for x in v) ** 0.5
    if n2 == 0:
        return v
    return [x / n2 for x in v]

def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x*y for x, y in zip(a, b)))


class _BM25Fielded:
    def __init__(self, field_weights: Dict[str, float], k1: float = 1.2, b: float = 0.75):
        self.field_weights = field_weights
        self.k1, self.b = k1, b
        self.docs: List[Dict[str, List[str]]] = []
        self.doc_len: List[int] = []
        self.df: Dict[str, int] = {}
        self.N = 0
        self.avgdl = 1.0
        self._idfs: Dict[str, float] = {}

    def add(self, doc_fields: Dict[str, List[str]]):
        self.docs.append(doc_fields)

    def finalize(self):
        self.N = len(self.docs) or 1
        self.doc_len, self.df = [], defaultdict(int)
        for doc in self.docs:
            seen = set()
            tlen = 0
            for toks in doc.values():
                tlen += len(toks)
                for t in set(toks):
                    if t not in seen:
                        self.df[t] += 1
                        seen.add(t)
            self.doc_len.append(tlen)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 1.0
        self._idfs = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}

    def score(self, q_terms: Iterable[str], idx: int) -> float:
        if idx >= self.N:
            return 0.0
        q_counts = Counter(q_terms)
        dl = self.doc_len[idx]
        K = self.k1 * ((1 - self.b) + self.b * (dl / (self.avgdl or 1.0)))
        doc = self.docs[idx]
        field_tfs = {f: Counter(toks) for f, toks in doc.items()}
        score = 0.0
        for term in q_counts.keys():
            idf = self._idfs.get(term)
            if idf is None:
                continue
            field_sum = 0.0
            for f, w in self.field_weights.items():
                tf = field_tfs.get(f, {}).get(term, 0)
                if tf:
                    field_sum += w * ((tf * (self.k1 + 1)) / (tf + K))
            score += idf * field_sum
        return score


class GCSEPhysicsIndex:
    """
    Synonym-free hybrid ranker:
      BM25F + char-3gram Jaccard + RapidFuzz token_set + Anchor lexicon + Topic router + SDM proximity.
    """

    def __init__(self, syllabus_doc_id: str = SYLLABUS_DOC_ID):
        self.syllabus_doc_id = syllabus_doc_id
        self.loaded = False
        self.meta: Dict[str, Any] = {}
        self.points: List[Dict[str, Any]] = []
        self._bm25 = _BM25Fielded(field_weights={"statement": 0.62, "topic": 0.28, "tags": 0.10})
        self._choices: List[str] = []              # for RapidFuzz
        self._comb_for_ngrams: List[set] = []      # char-3grams per doc
        self._doc_texts: List[str] = []            # raw for SDM
        self._embs: Optional[List[List[float]]] = None

        # Anchor lexicon + routing
        self._anchor_lex: Dict[str, int] = {}          # phrase -> freq
        self._anchor_to_topics: Dict[str, Counter] = {}# phrase -> Counter(topic_id)

    # --- utilities ---
    def _ngrams(self, toks: List[str], nmin=1, nmax=4):
        for n in range(nmin, nmax+1):
            for i in range(len(toks)-n+1):
                yield " ".join(toks[i:i+n])

    # --- build index from Firestore ---
    async def load_from_firestore(self, db_client: AsyncClient):
        # syllabus meta (optional)
        syll_ref = db_client.collection(SYLLABI_COLL).document(self.syllabus_doc_id)
        syll_snap = await syll_ref.get()
        if syll_snap.exists:
            self.meta = syll_snap.to_dict() or {}

        # pull points
        q = db_client.collection(POINTS_INDEX_COLL).where("syllabus_id", "==", self.syllabus_doc_id)
        pts: List[Dict[str, Any]] = []
        async for doc in q.stream():
            d = doc.to_dict() or {}
            statement = (d.get("statement") or "").strip()
            if not statement:
                continue
            pt = {
                "paper_id": str(d.get("paper_id") or ""),
                "topic_id": str(d.get("topic_id") or ""),
                "point_id": str(d.get("point_id") or ""),
                "topic_name": (d.get("topic_name") or "").strip(),
                "statement": statement,
                "equations": d.get("equations") or [],
                "maths_skills": d.get("maths_skills") or [],
                "tags": d.get("tags") or [],
            }
            pts.append(pt)

        self.points = pts
        self._bm25 = _BM25Fielded(field_weights={"statement": 0.62, "topic": 0.28, "tags": 0.10})
        self._choices, self._comb_for_ngrams, self._doc_texts = [], [], []
        self._anchor_lex, self._anchor_to_topics = {}, {}

        # build per-doc fields
        for p in self.points:
            fields = {
                "statement": _tokenize(p["statement"]),
                "topic": _tokenize(p.get("topic_name","")),
                "tags": _tokenize(" ".join(p.get("tags") or [])),
            }
            p["_tokens"] = fields
            p["_termset"] = set(sum(fields.values(), []))
            self._bm25.add(fields)

            combo = f"{p['statement']} || {p.get('topic_name','')} || {' '.join(p.get('tags') or [])}"
            self._choices.append(combo)
            grams = _char_ngrams(combo)
            self._comb_for_ngrams.append(grams)

            raw_stmt = p["statement"].lower()
            raw_topic = (p.get("topic_name") or "").lower()
            raw_tags = " ".join(p.get("tags") or []).lower()
            doc_raw = " ".join([raw_stmt, raw_topic, raw_tags]).strip()
            self._doc_texts.append(doc_raw)

            # anchor harvesting (corpus-only)
            toks = _tokenize(doc_raw)
            seen_phrases = set()
            for phr in self._ngrams(toks, 1, 4):
                if len(phr) < 3: 
                    continue
                if phr in STOP_WORDS:
                    continue
                if any(ch.isdigit() for ch in phr):
                    continue
                seen_phrases.add(phr)
            for phr in seen_phrases:
                self._anchor_lex[phr] = self._anchor_lex.get(phr, 0) + 1
                self._anchor_to_topics.setdefault(phr, Counter())[p["topic_id"]] += 1

        self._bm25.finalize()

        # prune weak anchors
        kept = {}
        for phr, f in self._anchor_lex.items():
            if f >= 3 or len(self._anchor_to_topics.get(phr, {})) >= 2:
                kept[phr] = f
        self._anchor_lex = kept

        # optional embeddings
        self._embs = None
        if EMBEDDINGS_ENABLED and _emb_client and self.points:
            try:
                texts = [f"{p['statement']} [{p.get('topic_name','')}]" for p in self.points]
                embs: List[List[float]] = []
                B = 128
                for i in range(0, len(texts), B):
                    chunk = texts[i:i+B]
                    r = _emb_client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
                    for item in r.data:
                        embs.append(_l2norm(item.embedding))
                self._embs = embs
                logger.info("Embeddings cached for %d points.", len(self._embs))
            except Exception as e:
                logger.warning("Embeddings unavailable (%s); continuing without.", e)
                self._embs = None

        self.loaded = True
        logger.info("Indexed %d points for %s (avgdl=%.1f).", len(self.points), self.syllabus_doc_id, self._bm25.avgdl)

    # --- anchors from query (exact + tight fuzzy) ---
    def _query_anchors(self, q: str) -> List[str]:
        qn = _norm_text(q)
        toks = _tokenize(qn)
        cand = set()

        # exact n-grams
        for phr in self._ngrams(toks, 1, 4):
            if phr in self._anchor_lex:
                cand.add(phr)

        # tight fuzzy to frequent anchors
        if len(cand) < 3 and self._anchor_lex:
            top_lex = sorted(self._anchor_lex.items(), key=lambda kv: -kv[1])[:2000]
            keys = [k for k,_ in top_lex]
            hits = process.extract(" ".join(toks), keys, scorer=fuzz.partial_ratio, limit=10)
            for k, score, _ in hits:
                if score >= 70:
                    cand.add(k)

        out = sorted(cand, key=lambda s: (-len(s), -self._anchor_lex.get(s,0), s))[:5]
        return out

    # --- SDM proximity features ---
    def _bigram_score(self, q_toks: List[str], doc_raw: str) -> float:
        if len(q_toks) < 2:
            return 0.0
        score = 0
        for i in range(len(q_toks)-1):
            big = f"{q_toks[i]} {q_toks[i+1]}"
            if big in doc_raw:
                score += 1
        return float(score)

    def _window_cooccur(self, q_toks: List[str], doc_raw: str, W: int = 8) -> float:
        dtoks = [t for t in re.findall(r"[a-z0-9]+", doc_raw) if t not in STOP_WORDS]
        pos = defaultdict(list)
        for i,t in enumerate(dtoks):
            pos[t].append(i)
        q = [t for t in q_toks if t in pos]
        if len(q) < 2:
            return 0.0
        pairs = 0
        for i in range(len(q)-1):
            a, b = q[i], q[i+1]
            pa, pb = pos[a], pos[b]
            ia = ib = 0
            local = 0
            while ia < len(pa) and ib < len(pb):
                if abs(pa[ia] - pb[ib]) <= W:
                    local += 1
                    if pa[ia] < pb[ib]: ia += 1
                    else: ib += 1
                elif pa[ia] < pb[ib]:
                    ia += 1
                else:
                    ib += 1
            pairs += local
        return float(pairs)

    # --- main match ---
    def match(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        if not self.loaded or not self.points:
            return []

        q_norm = _norm_text(query)
        q_tokens0 = _tokenize(q_norm)
        q_ngrams = _char_ngrams(q_norm)

        # anchors (corpus-only signals)
        anchors = self._query_anchors(query)
        q_tokens = list(dict.fromkeys(q_tokens0 + sum([a.split() for a in anchors], [])))

        # candidate recall set
        cand_idxs: List[int] = []

        # a) BM25 wide pass
        prelim = [(i, self._bm25.score(q_tokens, i)) for i in range(len(self.points))]
        prelim.sort(key=lambda t: t[1], reverse=True)
        for i, _ in prelim[:max(80, top_k*8)]:
            cand_idxs.append(i)

        # b) char-3gram Jaccard pass
        jacc_scores = []
        for i, grams in enumerate(self._comb_for_ngrams):
            js = _jaccard(q_ngrams, grams)
            if js >= 0.12:
                jacc_scores.append((i, js))
        jacc_scores.sort(key=lambda t: t[1], reverse=True)
        for i, _ in jacc_scores[:60]:
            if i not in cand_idxs:
                cand_idxs.append(i)

        # enforce anchors if present; else route by topics
        if anchors:
            # keep docs that contain any anchor token
            anchor_terms = set(sum([a.split() for a in anchors], []))
            gated = [i for i in cand_idxs if (self.points[i]["_termset"] & anchor_terms)]
            if gated:
                cand_idxs = gated
            else:
                # route by anchor→topic votes
                topic_votes = Counter()
                for a in anchors:
                    topic_votes.update(self._anchor_to_topics.get(a, {}))
                top_topics = {tid for tid,_ in topic_votes.most_common(3)}
                routed = [i for i in range(len(self.points)) if self.points[i]["topic_id"] in top_topics]
                cand_idxs = routed if routed else cand_idxs

        if not cand_idxs:
            return []

        # score components
        bm25 = {i: self._bm25.score(q_tokens, i) for i in cand_idxs}
        rf   = {i: (fuzz.token_set_ratio(q_norm, self._choices[i]) / 100.0) for i in cand_idxs}
        jac  = {i: _jaccard(q_ngrams, self._comb_for_ngrams[i]) for i in cand_idxs}
        emb  = {}
        if EMBEDDINGS_ENABLED and self._embs and _emb_client:
            try:
                r = _emb_client.embeddings.create(model=EMBEDDING_MODEL, input=[q_norm])
                qv = _l2norm(r.data[0].embedding)
                for i in cand_idxs:
                    emb[i] = _dot(qv, self._embs[i])
            except Exception:
                emb = {}

        # SDM
        q_toks_sdm = _tokenize(q_norm)
        big = {i: self._bigram_score(q_toks_sdm, self._doc_texts[i]) for i in cand_idxs}
        win = {i: self._window_cooccur(q_toks_sdm, self._doc_texts[i], W=8) for i in cand_idxs}

        # normalise
        def _norm(d):
            if not d: return {}
            m = max(d.values()) or 1.0
            return {k:v/m for k,v in d.items()}

        bm25n, rfn, jacn, embn, bign, winn = map(_norm, [bm25, rf, jac, emb, big, win])

        # blend (BM25 dominates, SDM tightens)
        blended: List[Tuple[int, float]] = []
        def_query = q_norm.startswith(("what is","define","what is meant by","explain what is"))
        for i in cand_idxs:
            s = (0.70 * bm25n.get(i,0.0) +
                 0.15 * bign.get(i,0.0)  +
                 0.10 * winn.get(i,0.0)  +
                 0.05 * rfn.get(i,0.0))
            # phrase-first boost
            for a in anchors:
                if a in self._doc_texts[i]:
                    s += 0.06
            # hard penalty if anchors exist but doc lacks them
            if anchors and not any(a in self._doc_texts[i] for a in anchors):
                s *= 0.15
            # definitional phrasing boost
            if def_query:
                st = self.points[i]["statement"].lower()
                if st.startswith(("define","recall that","state that","explain what is","explain what is meant by")):
                    s += 0.10
            blended.append((i, s))

        blended.sort(key=lambda t: t[1], reverse=True)
        out: List[Tuple[Dict[str, Any], float]] = []
        for i, s in blended[:max(1, top_k)]:
            p = self.points[i]
            out.append((p, float(round(s, 6))))
        return out


_physics_index: Optional[GCSEPhysicsIndex] = GCSEPhysicsIndex()

# Multi-tutor index cache
_indices: Dict[str, GCSEPhysicsIndex] = {}

async def ensure_index_loaded(tutor_id: str) -> GCSEPhysicsIndex:
    """Load (and cache) a syllabus index for a given tutor_id."""
    if tutor_id in _indices and _indices[tutor_id].loaded:
        return _indices[tutor_id]
    idx = GCSEPhysicsIndex(syllabus_doc_id=tutor_id)
    await idx.load_from_firestore(db)
    _indices[tutor_id] = idx
    return idx


# ==============================================================================
# Optional LLM rerank on top of hybrid (kept small)
# ==============================================================================

RERANK_ENABLED = False
RERANK_CANDIDATES = 20
RERANK_TOP_K = 12
RERANK_BLEND = (0.6, 0.4)  # original, rerank

# --- NEW: Strict LLM selection gate to drop off-topic points ---
SELECT_ENABLED = True
SELECT_TOP_K = 20       # consider up to this many candidates (after rerank)
SELECT_MAX_KEEP = 6     # keep at most this many for grounding/answer
SELECT_MIN_KEEP = 3     # but ensure at least this many survive (fallback to top-N)


def _build_rerank_prompt(query: str, candidates: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, str]]:
    system = (
        "You are a GCSE Physics syllabus reranker. Rank the given specification points by how directly and "
        "precisely they answer the user's query. Prefer exact definitions when the query starts with 'what is' "
        "or 'define'. Avoid tangents. Output ONLY valid JSON like: {\"order\":[0,2,1,...]} with candidate indices."
    )
    items = []
    for i, (p, _s) in enumerate(candidates):
        stmt = p["statement"]
        if len(stmt) > 240:
            stmt = stmt[:237] + "..."
        items.append({
            "idx": i,
            "point_id": p["point_id"],
            "topic_id": p["topic_id"],
            "paper_id": p["paper_id"],
            "statement": stmt,
            "equations": (p.get("equations") or [])[:2],
        })
    user = {"query": query, "top_k": min(RERANK_TOP_K, len(candidates)), "candidates": items}
    return [{"role":"system","content":system}, {"role":"user","content": json.dumps(user, ensure_ascii=False)}]


def _blend_scores(original: float, rank_pos: int, n: int) -> float:
    rr = 1.0 if n <= 1 else (1.0 - (rank_pos / (n - 1)))
    a, b = RERANK_BLEND
    return a * float(original) + b * rr


async def _llm_rerank_points(query: str, matches: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
    if not RERANK_ENABLED or not _oai_ready() or not matches:
        return matches[:RERANK_TOP_K]

    cands = matches[:RERANK_CANDIDATES]
    messages = _build_rerank_prompt(query, cands)

    try:
        client = _oai_client()
        resp = client.chat.completions.create(model=OPENAI_RERANK_MODEL, messages=messages)
        text = (resp.choices[0].message.content or "").strip()
        data = json.loads(text)
        order = data.get("order", [])
        if not isinstance(order, list) or not all(isinstance(i, int) for i in order):
            raise ValueError("bad order format")

        reranked: List[Tuple[Dict[str, Any], float]] = []
        used = set()
        n = min(len(order), len(cands))
        for pos, idx in enumerate(order[:n]):
            if 0 <= idx < len(cands) and idx not in used:
                p, s = cands[idx]
                reranked.append((p, _blend_scores(s, pos, n)))
                used.add(idx)
        for i, (p, s) in enumerate(cands):
            if i not in used:
                reranked.append((p, _blend_scores(s * 0.9, len(reranked), len(cands))))
        reranked.sort(key=lambda t: t[1], reverse=True)
        return reranked[:RERANK_TOP_K]
    except Exception as e:
        logger.warning("LLM rerank failed (%s); using original ordering", e)
        return matches[:RERANK_TOP_K]


# --- NEW: LLM selection/gating step to prevent out-of-context points ---
async def _llm_select_points(
    query: str,
    matches: List[Tuple[Dict[str, Any], float]],
    top_k: int = SELECT_TOP_K,
    max_keep: int = SELECT_MAX_KEEP,
    min_keep: int = SELECT_MIN_KEEP,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Ask OPENAI_MODEL to look at the user query and a small set of candidate points, and
    return ONLY the ones that are directly relevant. This acts as a conservative gate to
    drop weird or off-topic points even if fuzzy recall brought them in.
    """
    if not SELECT_ENABLED or not _oai_ready() or not matches:
        return matches[:max_keep]

    cands = matches[:max(1, top_k)]

    # Build a strict, JSON-only prompt
    system = (
        "You are a strict relevance filter for GCSE Physics specification points. "
        "Given a user query and a numbered list of candidate points, choose only those "
        "that directly help answer the query. Reject tangential or off-topic points. "
        "If the query starts with 'what is' or 'define', prefer definition-style statements. "
        "If it starts with 'how' or 'describe', prefer procedure/steps or law statements. "
        f"Return ONLY compact JSON like: {{\"keep\":[0,2,1]}}. Keep at most {max_keep}."
    )

    items = []
    for i, (p, _s) in enumerate(cands):
        stmt = p.get("statement", "")
        if len(stmt) > 220:
            stmt = stmt[:217] + "..."
        items.append({
            "idx": i,
            "topic": p.get("topic_name", ""),
            "statement": stmt,
            "equations": (p.get("equations") or [])[:2],
        })
    user = {
        "query": query,
        "candidates": items,
        "constraints": {
            "max_keep": max_keep,
            "rules": [
                "drop unrelated topics",
                "avoid generic points that do not move the answer forward",
                "prefer exact definitions for 'what is/define'",
                "prefer law/formula statements when referenced by name",
                "prefer apparatus/steps when the user asks 'how to' or 'describe an experiment'",
            ],
        },
    }

    try:
        client = _oai_client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        keep = data.get("keep", [])
        if not isinstance(keep, list):
            raise ValueError("bad keep list")

        # sanitize + truncate
        keep = [int(i) for i in keep if isinstance(i, int) and 0 <= i < len(cands)]
        # de-dup preserving order
        seen = set()
        keep = [i for i in keep if not (i in seen or seen.add(i))][:max_keep]

        if len(keep) < min_keep:
            # fallback: top-scoring until min_keep
            need = min_keep - len(keep)
            for i in range(len(cands)):
                if i not in keep:
                    keep.append(i)
                    need -= 1
                    if need <= 0:
                        break

        filtered = [cands[i] for i in keep]
        logger.info("LLM selector kept %d/%d points", len(filtered), len(cands))
        return filtered
    except Exception as e:
        logger.warning("LLM select failed (%s); falling back to top-%d", e, max_keep)
        return cands[:max_keep]


# ==============================================================================
# Grounding / smalltalk / helpers
# ==============================================================================

MIN_RELEVANCE = 0.35
GROUNDING_TOP_N = 5
GROUNDING_MIN_SCORE = 0.40

def _is_smalltalk(msg: str) -> bool:
    m = (msg or "").strip().lower()
    if not m:
        return True
    patterns = [
        r"^(hi|hello|hey|yo|howdy)\b[!.?]*$",
        r"^good (morning|afternoon|evening)[!.?]*$",
        r"^(thanks|thank you|cheers)[!.?]*$",
        r"^(ok|okay|cool|nice|great|awesome)[!.?]*$",
        r"^(test|ping|hello there\??)$"
    ]
    if any(re.match(p, m) for p in patterns):
        return True
    toks = re.findall(r"[a-z0-9=]+", m)
    if ("=" not in m) and not any(ch.isdigit() for ch in m) and len(toks) <= 2:
        return True
    return False

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def max_severity(s1: str, s2: str) -> str:
    levels = {"low": 0, "medium": 1, "high": 2}
    return s1 if levels.get(s1.lower(), 0) >= levels.get(s2.lower(), 0) else s2

def handle_openai_error(e: Exception) -> str:
    msg = str(e)
    if "unsupported_value" in msg or "does not support" in msg:
        return "This model doesn’t support that parameter; request was rejected."
    if any(k in msg.lower() for k in ["rate", "quota", "limit"]):
        return "I'm hitting a rate limit right now. Please try again in a moment."
    return "I couldn't reach the model just now. Please try again."

def _topic_from(cmd: str) -> str:
    t = re.sub(r"^@[\w-]+\s*", "", cmd or "").strip()
    return t or "this topic"


# ==============================================================================
# Weak areas: NEVER store 'Unknown topic'
# ==============================================================================

def _is_unknown_topic_name(name: Optional[str]) -> bool:
    if not name:
        return True
    bad = {"unknown","unknown topic","this","that","it","topic"}
    s = (name or "").strip().lower()
    return s in bad or len(s) < 3

def detect_weak_areas(
    message: str,
    response: str,
    matched_points: Optional[List[Dict[str, Any]]] = None
) -> List[WeakArea]:
    """
    Extracts weak areas from the *user* message only.
    Returns [] if no clear topic could be identified (never returns 'unknown topic').
    Optionally uses matched_points[0].topic_name as a fallback when confident.
    """
    weak_areas: List[WeakArea] = []
    m = (message or "").strip().lower()
    if not m:
        return weak_areas

    trig_strict = {
        "help with": "medium",
        "struggling with": "high",
        "confused about": "high",
        "don't understand": "high",
    }
    trig_generic = {
        "not sure": "medium",
        "difficulty": "medium",
        "trouble": "medium",
        "explain again": "low",
    }

    def _clean_topic(t: str) -> str:
        t = (t or "").strip(" .,:;!?\"'()[]{}")
        for lead in ("about ","on ","in ","the ","this ","that ","it "):
            if t.startswith(lead):
                t = t[len(lead):]
        return t.strip()

    def _extract_after(phrase: str) -> str:
        try:
            idx = m.index(phrase)
        except ValueError:
            return ""
        tail = m[idx + len(phrase):]
        stop_idx = len(tail)
        for sep in [".","?","!","\n"]:
            j = tail.find(sep)
            if j != -1:
                stop_idx = min(stop_idx, j)
        return _clean_topic(tail[:stop_idx])[:60].strip()

    now = now_iso()

    # strict phrases
    for phrase, severity in trig_strict.items():
        if phrase in m:
            topic = _extract_after(phrase)
            if not _is_unknown_topic_name(topic):
                weak_areas.append(WeakArea(
                    topic=topic.capitalize(),
                    description=f"User expressed difficulty with {topic}",
                    severity=severity,
                    first_identified=now,
                    last_encountered=now,
                    session_ids=[]
                ))

    # generic phrases: fallback only if confident syllabus topic present
    if not weak_areas:
        for phrase, severity in trig_generic.items():
            if phrase in m and matched_points:
                top = matched_points[0]
                if float(top.get("score", 0.0)) >= 0.45:
                    fallback = (top.get("topic_name") or "").strip()
                    if not _is_unknown_topic_name(fallback):
                        weak_areas.append(WeakArea(
                            topic=fallback,
                            description=f"User expressed difficulty with {fallback}",
                            severity=severity,
                            first_identified=now,
                            last_encountered=now,
                            session_ids=[]
                        ))
                break

    return weak_areas


# ==============================================================================
# Media hint (kept)
# ==============================================================================

def _image_cls_messages(query: str, points: List[dict]) -> List[dict]:
    system = (
        "You are a concise classifier for an educational tutor. "
        "Decide if an illustrative image/diagram would meaningfully help the user's request. "
        "Use the user message plus a few syllabus points to judge. "
        "Return ONLY compact JSON like: {\"show\": true, \"query\": \"<concise search term>\"}. "
        "Rules: Suggest images for things that benefit from diagrams/plots/schemas "
        "(e.g., waveforms, spectra, field lines, ray diagrams, circuits, force/free-body diagrams, "
        "lenses, energy level transitions, nuclear/atomic structure, motion graphs). "
        "Do NOT justify; keep query 2–5 words, noun-ish."
    )
    payload = {
        "query": query,
        "points": [
            {"statement": p.get("statement"), "topic": p.get("topic_name"), "equations": (p.get("equations") or [])[:2]}
            for p in points[:5]
        ]
    }
    return [{"role":"system","content":system}, {"role":"user","content": json.dumps(payload, ensure_ascii=False)}]

async def infer_image_hint_ai(message: str, matched_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        client = _oai_client()
        top = []
        for mp in matched_points[:5]:
            s = mp.get("statement", "")
            if len(s) > 180:
                s = s[:177] + "..."
            top.append({
                "id": f"{mp.get('paper_id','')}-{mp.get('topic_id','')}-{mp.get('point_id','')}",
                "statement": s,
                "eqs": (mp.get("equations") or [])[:2]
            })
        system = (
            "Decide if a SINGLE illustrative image would significantly help a short GCSE Physics answer. "
            "Only show an image if it adds clear conceptual clarity (not decoration). "
            'Return STRICT JSON: {"show":bool,"topic":str,"query":str,"kind":"diagram|photo|chart",'
            '"layout":"inline-right|block-above","reason":str,"confidence":float}.'
        )
        payload = {"user_message": message, "syllabus_points": top}
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system}, {"role":"user","content": json.dumps(payload, ensure_ascii=False)}]
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        if not isinstance(data, dict) or "show" not in data:
            raise ValueError("bad media JSON")
        data["confidence"] = float(max(0.0, min(1.0, data.get("confidence", 0.5))))
        if data.get("show"):
            data.setdefault("layout", "inline-right")
            data.setdefault("kind", "diagram")
            data.setdefault("topic", "")
            data.setdefault("query", "")
            data.setdefault("reason", "Clarifies concept")
        return data
    except Exception as e:
        logger.info("media-hint failed: %s", e)
        return {"show": False, "confidence": 0.0, "reason": "no decision"}


# ==============================================================================
# Answering style
# ==============================================================================

def _style_instructions(style: Any) -> str:
    if isinstance(style, dict):
        tone = style.get("tone", "concise")
        max_lines = style.get("max_lines", 7)
        allow = style.get("allow", ["h1","bold","underline","hr"]) or []
        end_hint = style.get("end_hint", "")
        allow_str = ", ".join(allow)
        tail = f" Finish with: {end_hint}" if end_hint else ""
        return (
            f"Use a {tone} tone. Keep to at most {max_lines} short lines unless a list is necessary. "
            f"Allowed formatting: {allow_str}." + tail
        )
    # fallback to your legacy preset
    return (
        "Answer directly and concisely first (no preamble). "
        "If helpful, use Markdown: headings (#), **bold**, _italics_, horizontal rules (---). "
        "You may use inline HTML for underline/sup/sub: <u>, <sup>, <sub>. "
        "Keep to 4–7 bullets max; no extra sections unless asked. "
        "Finish with one short hint (e.g., 'Want examples or a quick quiz?')."
    )


async def get_tutor_meta(tutor_id: str) -> Dict[str, str]:
    """Return {'id': tutor_id, 'title': str} from Firestore if available; else safe defaults."""
    try:
        ref = db.collection(SYLLABI_COLL).document(tutor_id)
        snap = await ref.get()
        if snap.exists:
            d = snap.to_dict() or {}
            title = d.get("title") or d.get("name") or tutor_id
            return {"id": tutor_id, "title": title}
    except Exception:
        pass
    defaults = {
        "edexcel_gcse_physics_issue4": "Edexcel GCSE Physics",
        "edexcel_igcse_economics_issue2": "Edexcel iGCSE Economics",
    }
    return {"id": tutor_id, "title": defaults.get(tutor_id, tutor_id)}


# ==============================================================================
# Routes
# ==============================================================================

@router.post("/session", response_model=CreateSessionResponse)
async def create_session(
    payload: CreateSessionIn = Body(default=None),
    current_email: str = Depends(get_current_email)
):
    sid = uuid4().hex
    tutor_id = (payload.tutor_id if payload else None) or DEFAULT_TUTOR_ID
    tutor_meta = await get_tutor_meta(tutor_id)

    doc_ref = db.collection(COLL).document(sid)
    seed = [
        {"role": "assistant", "content": "New session ready. Paste a topic or use an @command (e.g. @explain, @examples, @quiz, @hint, @summarize)."}
    ]
    await doc_ref.set({
        "user_email": current_email,
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "title": (payload.title if payload and payload.title else None),
        "last_message_preview": seed[0]["content"][:160],
        "messages": seed,
        "user_msg_count": 0,
        "user_msg_limit": 20,
        "tutor_id": tutor_id,
        "tutor_title": tutor_meta.get("title"),
    })
    return {
        "session_id": sid,
        "seed_messages": seed,
        "remaining_messages": 20,
        "user_msg_limit": 20,
        "tutor_id": tutor_id,
        "tutor_title": tutor_meta.get("title"),
    }


async def _inc_user_msg_count(doc_ref, limit: int):
    try:
        await doc_ref.set({
            "user_msg_count": firestore.Increment(1),
            "user_msg_limit": limit,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)
    except Exception:
        pass

@router.get("/media")
async def media_search(q: str = Query(..., min_length=2), limit: int = 6):
    try:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": q,
            "gsrnamespace": 6,
            "gsrlimit": min(max(limit,1),12),
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": 800,
            "format": "json",
            "origin": "*",
        }
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get("https://commons.wikimedia.org/w/api.php", params=params)
            r.raise_for_status()
        data = r.json()
        pages = (data.get("query", {}) or {}).get("pages", {}) or {}
        images = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            ext = info.get("extmetadata", {}) or {}
            images.append({
                "title": page.get("title"),
                "thumb": info.get("thumburl") or info.get("url"),
                "url": info.get("url"),
                "mime": info.get("mime"),
                "source": info.get("descriptionurl"),
                "license": (ext.get("LicenseShortName") or {}).get("value"),
                "attribution": (ext.get("Artist") or {}).get("value"),
            })
        return {"images": images}
    except Exception as e:
        logger.warning("media_search failed: %s", e)
        return {"images": []}

@router.get("/sessions", response_model=List[SessionsListItem])
async def list_sessions(current_email: str = Depends(get_current_email)):
    try:
        q = (
            db.collection(COLL)
            .where("user_email", "==", current_email)
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(50)
        )
        items: List[SessionsListItem] = []
        async for doc in q.stream():
            data = doc.to_dict() or {}
            ts = data.get("updated_at")
            items.append(SessionsListItem(
                id=doc.id,
                title=data.get("title") or f"Session {doc.id[:6]}",
                last_message_preview=data.get("last_message_preview"),
                updated_at=(ts.isoformat() if hasattr(ts, "isoformat") else None),
                can_edit=True,
                tutor_id=data.get("tutor_id"),
                tutor_title=data.get("tutor_title"),
            ))
        return items
    except Exception as e:
        raise HTTPException(500, f"Error fetching sessions: {str(e)}")


@router.patch("/session/rename")
async def rename_session(request: UpdateSessionRequest, current_email: str = Depends(get_current_email)):
    try:
        doc_ref = db.collection(COLL).document(request.session_id)
        snap = await doc_ref.get()
        if not snap.exists:
            raise HTTPException(404, "Session not found")
        if snap.get("user_email") != current_email:
            raise HTTPException(403, "Forbidden")
        await doc_ref.update({"title": request.new_title, "updated_at": firestore.SERVER_TIMESTAMP})
        return {"ok": True, "new_title": request.new_title}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error renaming session: {str(e)}")

@router.get("/session/{sid}", response_model=SessionOut)
async def get_session(sid: str, current_email: str = Depends(get_current_email)):
    try:
        doc_ref = db.collection(COLL).document(sid)
        snap = await doc_ref.get()
        if not snap.exists:
            raise HTTPException(404, "Session not found")
        d = snap.to_dict() or {}
        if d.get("user_email") != current_email:
            raise HTTPException(403, "Forbidden")
        ts = d.get("updated_at")
        return SessionOut(
            id=sid,
            messages=[Message(**m) for m in d.get("messages", [])],
            title=d.get("title"),
            updated_at=(ts.isoformat() if hasattr(ts, "isoformat") else None),
            tutor_id=d.get("tutor_id") or DEFAULT_TUTOR_ID,
            tutor_title=d.get("tutor_title") or (await get_tutor_meta(d.get("tutor_id") or DEFAULT_TUTOR_ID)).get("title"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error fetching session: {str(e)}")



@router.post("/message")
async def append_message(payload: MessageIn, current_email: str = Depends(get_current_email)):
    try:
        doc_ref = db.collection(COLL).document(payload.session_id)
        snap = await doc_ref.get()
        if not snap.exists:
            raise HTTPException(404, "Session not found")
        session_data = snap.to_dict() or {}
        if session_data.get("user_email") != current_email:
            raise HTTPException(403, "Forbidden")

        messages = session_data.get("messages", [])

        # Derive points from explicit field or meta/meta_json
        points = payload.points
        if points is None and payload.meta and isinstance(payload.meta, dict):
            points = payload.meta.get("points") or payload.meta.get("matched_points") or payload.meta.get("syllabus_points")
        if points is None and payload.meta_json:
            try:
                meta_obj = json.loads(payload.meta_json)
                if isinstance(meta_obj, dict):
                    points = meta_obj.get("points") or meta_obj.get("matched_points") or meta_obj.get("syllabus_points")
            except Exception:
                pass

        new_message = {
            "role": payload.role,
            "content": payload.content,
            "type": payload.type,
            "quiz": payload.quiz
        }
        if points is not None:
            new_message["points"] = points
        if payload.flashcards is not None:             # <— NEW
            # ensure list of dicts (Pydantic models become dicts automatically)
            new_message["flashcards"] = [fc.dict() for fc in payload.flashcards]

        messages.append(new_message)

        title = session_data.get("title")
        if not title and payload.role == "user":
            clean = payload.content.strip()
            title = (clean[1:] if clean.startswith("@") else clean)[:48] or "Session"

        update_data = {
            "messages": messages,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "last_message_preview": payload.content[:160],
        }
        if not session_data.get("title") and title:
            update_data["title"] = title

        await doc_ref.set(update_data, merge=True)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error saving message: %s", e)
        raise HTTPException(500, "Failed to save message. Please try again.")


# ------------------------------------------------------------------------------
# Chat (hybrid fuzzy → optional LLM rerank → LLM selection gate → answer)
# ------------------------------------------------------------------------------

# --- helper (place near your other utilities) ---
_BULLET_BLOCK_RE = re.compile(
    r'^[ \t]*(?:[-*•\u2013\u2014]\s*)?\[[^\]]+\].*(?:\r?\n)?',
    re.MULTILINE
)
def _strip_grounding_from_text(text: str) -> str:
    """Remove any syllabus-point bullet lines the model might have echoed."""
    cleaned = _BULLET_BLOCK_RE.sub("", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ------------------------------------------------------------------------------
# Chat (hybrid fuzzy → optional LLM rerank → LLM selection gate → answer)
# ------------------------------------------------------------------------------

@router.post("/chat")
async def chat_handler(data: Dict[str, Any] = Body(...), current_email: str = Depends(get_current_email)):
    quick = bool((data.get("quick") or data.get("mode") == "scan"))
    session_id: str = data.get("session_id")
    message: str = (data.get("message") or "").strip()

    if not session_id:
        raise HTTPException(400, "session_id required")
    if not message:
        raise HTTPException(400, "message required")
    if not _oai_ready():
        raise HTTPException(500, "OpenAI client not available on server")

    try:
        # session + recent history
        doc_ref = db.collection(COLL).document(session_id)
        snap = await doc_ref.get()
        if not snap.exists:
            raise HTTPException(404, "Session not found")
        if snap.get("user_email") != current_email:
            raise HTTPException(403, "Forbidden")
        data0 = snap.to_dict() or {}

        # Tutor info
        tutor_id = data0.get("tutor_id") or DEFAULT_TUTOR_ID
        tutor_meta = await get_tutor_meta(tutor_id)
        tutor_title = tutor_meta.get("title") or DEFAULT_TUTOR_TITLE

        history = data0.get("messages", [])[-10:]

        # message limit
        limit = int(data0.get("user_msg_limit", 20))
        count = int(data0.get("user_msg_count", 0))
        remaining_before = max(0, limit - count)
        if remaining_before <= 0:
            brief = "You’ve reached the 20-message limit for this session. Please start a new session to continue."
            return {
                "reply": brief,
                "matched_points": [],
                "citations": [],
                "remaining_messages": 0,
                "user_msg_limit": limit,
                "limit_reached": True
            }

        # small talk → brief prompt (still counts)
        if _is_smalltalk(message):
            await _inc_user_msg_count(doc_ref, limit)
            await _append_user_message(session_id, message)
            brief = f"Hello — how can I help you with {tutor_title} today?"
            await _append_assistant_message(session_id, brief)
            return {
                "reply": brief,
                "matched_points": [],
                "citations": [],
                "remaining_messages": max(0, remaining_before - 1),
                "user_msg_limit": limit,
                "limit_reached": False
            }

        # ensure index and match (per tutor)
        idx = await ensure_index_loaded(tutor_id)
        matches = idx.match(message, top_k=RERANK_CANDIDATES) if idx and idx.loaded else []

        # optional LLM rerank
        if matches:
            try:
                matches = await _llm_rerank_points(message, matches)
            except Exception as e:
                logger.warning("LLM rerank failed (%s); using original ordering", e)

        # LLM selection gate
        if matches:
            try:
                matches = await _llm_select_points(
                    message, matches,
                    top_k=SELECT_TOP_K,
                    max_keep=SELECT_MAX_KEEP,
                    min_keep=SELECT_MIN_KEEP
                )
            except Exception as e:
                logger.warning("LLM selection failed (%s); keeping original top-%d", e, SELECT_MAX_KEEP)
                matches = matches[:SELECT_MAX_KEEP]

        max_score = max((float(s) for (_p, s) in matches), default=0.0)
        if not matches or max_score < MIN_RELEVANCE:
            brief = (
                f"I couldn’t confidently match your request to the {tutor_title} syllabus. "
                "Try naming the topic or formula (e.g., “@explain Newton’s second law” or “supply and demand diagram”)."
            )
            await _append_assistant_message(session_id, brief)
            return {
                "reply": brief,
                "matched_points": [],
                "citations": [],
                "remaining_messages": max(0, remaining_before - 1),
                "user_msg_limit": limit,
                "limit_reached": False
            }

        # flatten to API-friendly points
        matched_points: List[Dict[str, Any]] = []
        for (p, score) in matches:
            matched_points.append({
                "paper_id": p["paper_id"],
                "topic_id": p["topic_id"],
                "topic_name": p.get("topic_name"),
                "point_id": p["point_id"],
                "statement": p["statement"],
                "equations": p.get("equations") or [],
                "maths_skills": p.get("maths_skills") or [],
                "tags": p.get("tags") or [],
                "score": round(float(score), 3)
            })

        # ---------------------------
        # Topic mode for vague prompts
        # ---------------------------
        def _decide_topic_mode(mps: List[Dict[str, Any]], msg: str) -> Tuple[bool, List[Dict[str, Any]]]:
            if not mps:
                return False, []
            # Heuristics: short/vague query or diffuse matches across many topics with low spread
            token_count = len(re.findall(r"[A-Za-z0-9]+", msg))
            by_topic: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
            for mp in mps:
                k = (mp.get("paper_id") or "", mp.get("topic_id") or "", mp.get("topic_name") or "")
                by_topic[k].append(mp)

            topic_count = len(by_topic)
            scores = sorted([mp["score"] for mp in mps], reverse=True)
            top = scores[0] if scores else 0.0
            k3 = scores[2] if len(scores) >= 3 else (scores[-1] if scores else 0.0)
            spread = top - k3

            vague = (
                token_count <= 3
                or (topic_count >= 3 and top < 0.62)
                or (top < 0.58 and spread < 0.12)
            )
            if not vague:
                return False, []

            topics: List[Dict[str, Any]] = []
            for (paper_id, topic_id, topic_name), items in by_topic.items():
                items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
                topics.append({
                    "is_topic": True,
                    "paper_id": paper_id,
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "score": round(max((x["score"] for x in items), default=0.0), 3),
                    "points_preview": [
                        {
                            "point_id": it["point_id"],
                            "statement": it["statement"],
                            "equations": it.get("equations") or [],
                            "tags": it.get("tags") or [],
                            "score": it["score"],
                        } for it in items_sorted[:6]
                    ],
                })
            topics.sort(key=lambda t: t["score"], reverse=True)
            return True, topics

        topic_mode, topic_matches = _decide_topic_mode(matched_points, message)

        # @quiz short-circuit (counts)
        if message.lower().startswith("@quiz"):
            await _inc_user_msg_count(doc_ref, limit)
            topic = _topic_from(message)
            quiz = _make_quiz(topic)
            reply = f"Here's a quick 3-question quiz on {topic}."
            await _append_user_message(session_id, message)
            await _append_assistant_message(session_id, reply, msg_type="quiz", quiz=quiz, points=matched_points[:4])
            resp = {
                "reply": reply,
                "type": "quiz",
                "quiz": quiz,
                "matched_points": matched_points,
                "citations": [f"firestore:{tutor_id}"],
                "remaining_messages": max(0, remaining_before - 1),
                "user_msg_limit": limit,
                "limit_reached": False,
            }
            if topic_mode:
                resp["topic_mode"] = True
                resp["topics"] = topic_matches
            return resp

        # style + model call (no echo of syllabus bullets)
        style = data.get("style") or DEFAULT_STYLE
        style_text = _style_instructions(style)
        system_preamble = (
            f"You are Atlas, a personal tutor. Use '{tutor_title}' specification wording for definitions when applicable. "
            "Do NOT list or quote syllabus points in your answer; the UI will render them separately. "
            "Never include bracketed codes like [1.1.3d] in the reply."
            + style_text
        )

        oa_messages = [{"role": "system", "content": system_preamble}]
        for m in history:
            role = "assistant" if m.get("role") == "assistant" else "user"
            oa_messages.append({"role": role, "content": m.get("content", "")})

        # Choose grounding: if topic mode, use top topic's preview points
        if topic_mode and topic_matches:
            top_topic = topic_matches[0]
            use_points = []
            for pp in top_topic.get("points_preview", [])[:GROUNDING_TOP_N]:
                use_points.append({
                    "paper_id": top_topic.get("paper_id"),
                    "topic_id": top_topic.get("topic_id"),
                    "topic_name": top_topic.get("topic_name"),
                    "point_id": pp.get("point_id"),
                    "statement": pp.get("statement"),
                    "equations": pp.get("equations") or [],
                    "maths_skills": [],
                    "tags": pp.get("tags") or [],
                    "score": pp.get("score", 0.0),
                })
        else:
            use_points = matched_points[:GROUNDING_TOP_N]

        grounding_payload = [
            {
                "id": mp["point_id"],
                "topic": mp.get("topic_name"),
                "eqs": (mp.get("equations") or [])[:2],
                "statement": (mp.get("statement") or "")[:200],
            }
            for mp in use_points
        ]
        oa_messages.append({
            "role": "system",
            "content": "GROUNDING (do not echo; use only for internal reasoning): "
                       + json.dumps(grounding_payload, ensure_ascii=False)
        })

        # Actual user message
        oa_messages.append({"role": "user", "content": message})

        client = _oai_client()
        completion = client.chat.completions.create(model=OPENAI_MODEL, messages=oa_messages)
        model_reply = (completion.choices[0].message.content or "").strip()

        # Final guard: strip any echoed point lines if the model slips
        model_reply = _strip_grounding_from_text(model_reply)

        # persist + count (store points for hover; if topic mode, store the points we grounded on)
        await _inc_user_msg_count(doc_ref, limit)
        await _append_user_message(session_id, message)
        await _append_assistant_message(session_id, model_reply, points=use_points[:4])

        media = await infer_image_hint_ai(message, matched_points)

        resp = {
            "reply": model_reply,
            "matched_points": [] if topic_mode else matched_points,
            "citations": [f"firestore:{tutor_id}"],
            "remaining_messages": max(0, remaining_before - 1),
            "user_msg_limit": limit,
            "limit_reached": False,
            "media": media,
        }
        if topic_mode:
            resp["topic_mode"] = True
            resp["topics"] = topic_matches
        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat error: %s", e)
        return {
            "reply": handle_openai_error(e),
            "matched_points": [],
            "citations": [f"firestore:{DEFAULT_TUTOR_ID}"],
            "remaining_messages": None,
            "user_msg_limit": 20,
            "limit_reached": False
        }



# ==============================================================================
# Persistence helpers
# ==============================================================================

async def _append_user_message(session_id: str, content: str):
    doc_ref = db.collection(COLL).document(session_id)
    snap = await doc_ref.get()
    if not snap.exists:
        return
    data = snap.to_dict() or {}
    msgs = data.get("messages", [])
    msgs.append({"role": "user", "content": content})
    await doc_ref.set({
        "messages": msgs,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "last_message_preview": content[:160],
    }, merge=True)

async def _append_assistant_message(
    session_id: str,
    content: str,
    msg_type: Optional[str] = None,
    quiz: Optional[Dict[str, Any]] = None,
    points: Optional[List[Dict[str, Any]]] = None,
    flashcards: Optional[List[Dict[str, Any]]] = None,
):
    doc_ref = db.collection(COLL).document(session_id)
    snap = await doc_ref.get()
    if not snap.exists:
        return
    data = snap.to_dict() or {}
    msgs = data.get("messages", [])

    entry: Dict[str, Any] = {"role": "assistant", "content": content}
    if msg_type:
        entry["type"] = msg_type
    if quiz:
        entry["quiz"] = quiz
    if points is not None:
        entry["points"] = points
    if flashcards is not None:
        entry["flashcards"] = flashcards
        entry["type"] = entry.get("type") or "flashcards"

    msgs.append(entry)  # <-- the missing line

    await doc_ref.set({
        "messages": msgs,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "last_message_preview": content[:160],
    }, merge=True)

# ==============================================================================
# Weak areas storage & cleanup
# ==============================================================================

@router.get("/weak-areas", response_model=WeakAreasResponse)
async def get_weak_areas(current_email: str = Depends(get_current_email)):
    try:
        doc_ref = db.collection("user_weak_areas").document(current_email)
        doc = await doc_ref.get()
        if not doc.exists:
            return {"weak_areas": []}
        return {"weak_areas": doc.to_dict().get("weak_areas", [])}
    except Exception as e:
        raise HTTPException(500, f"Error fetching weak areas: {str(e)}")

@router.post("/weak-areas", response_model=WeakAreasResponse)
async def update_weak_areas(payload: WeakAreasUpdate, current_email: str = Depends(get_current_email)):
    try:
        doc_ref = db.collection("user_weak_areas").document(current_email)
        existing = []
        doc = await doc_ref.get()
        if doc.exists:
            existing = doc.to_dict().get("weak_areas", [])
        existing_areas = [WeakArea(**area) for area in existing]
        updated_areas: List[WeakArea] = []
        for new_area in payload.weak_areas:
            # never allow unknowns in
            if _is_unknown_topic_name(new_area.topic):
                continue
            found = next((a for a in existing_areas if a.topic.lower() == new_area.topic.lower()), None)
            if found:
                found.last_encountered = new_area.last_encountered
                found.severity = max_severity(found.severity, new_area.severity)
                sid = new_area.session_ids[0] if new_area.session_ids else None
                if sid and sid not in found.session_ids:
                    found.session_ids.append(sid)
                updated_areas.append(found)
            else:
                updated_areas.append(new_area)
        areas_dict = [area.dict() for area in updated_areas]
        await doc_ref.set({"weak_areas": areas_dict}, merge=True)
        return {"weak_areas": areas_dict}
    except Exception as e:
        raise HTTPException(500, f"Error updating weak areas: {str(e)}")

@router.delete("/weak-areas/cleanup-unknown")
async def cleanup_unknown_weak_areas(current_email: str = Depends(get_current_email)):
    try:
        ref = db.collection("user_weak_areas").document(current_email)
        snap = await ref.get()
        if not snap.exists:
            return {"removed": 0}
        data = snap.to_dict() or {}
        items = data.get("weak_areas", [])
        kept = [a for a in items if not _is_unknown_topic_name(a.get("topic"))]
        removed = len(items) - len(kept)
        if removed:
            await ref.set({"weak_areas": kept}, merge=True)
        return {"removed": removed}
    except Exception as e:
        raise HTTPException(500, f"Cleanup failed: {str(e)}")


# ==============================================================================
# Simple quiz generator (kept from your version)
# ==============================================================================

def _make_quiz(topic: str) -> Dict[str, Any]:
    topic = topic.lower().strip()
    science_questions = [
        {"template": "What is the primary purpose of {topic}?", "options": ["Energy production","Information storage","Structural support","Waste removal"], "correct": 0},
        {"template": "Which of these is essential for {topic}?", "options": ["Oxygen","Carbon dioxide","Water","Nitrogen"], "correct": None},
        {"template": "Where does {topic} primarily occur in cells?", "options": ["Mitochondria","Chloroplasts","Nucleus","Ribosomes"], "correct": None}
    ]
    if "photosynthesis" in topic:
        science_questions[1]["correct"] = 1
        science_questions[2]["correct"] = 1
    elif "respiration" in topic:
        science_questions[1]["correct"] = 0
        science_questions[2]["correct"] = 0
    quiz = {"topic": topic.capitalize(), "items": []}
    for q in science_questions[:3]:
        quiz["items"].append({
            "prompt": q["template"].format(topic=topic),
            "choices": q["options"],
            "answerIdx": q["correct"],
            "feedback": {"correct": f"Correct! Good understanding of {topic}.", "wrong": f"Review the fundamentals of {topic}."}
        })
    return quiz

# ==============================================================================
# Tutors
# ==============================================================================



@router.get("/tutors", response_model=List[Tutor])
async def get_tutors(current_email: str = Depends(get_current_email)):
    items: List[Tutor] = []
    try:
        async for doc in db.collection(SYLLABI_COLL).stream():
            d = doc.to_dict() or {}
            items.append(Tutor(id=doc.id, title=d.get("title") or d.get("name") or doc.id))
    except Exception as e:
        logger.info("get_tutors: fallback defaults (%s)", e)

    # ensure defaults exist
    defaults = {
        "edexcel_gcse_physics_issue4": "Edexcel GCSE Physics",
        "edexcel_igcse_economics_issue2": "Edexcel iGCSE Economics",
    }
    have = {t.id for t in items}
    for k, v in defaults.items():
        if k not in have:
            items.append(Tutor(id=k, title=v))

    items.sort(key=lambda t: t.title.lower())
    return items

@router.post("/tutors", response_model=Tutor)
async def upsert_tutor(t: TutorIn, current_email: str = Depends(get_current_email)):
    try:
        ref = db.collection(SYLLABI_COLL).document(t.id)
        await ref.set({"title": t.title, "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)
        return Tutor(id=t.id, title=t.title)
    except Exception as e:
        raise HTTPException(500, f"Error creating/updating tutor: {str(e)}")

class SessionTutorUpdate(BaseModel):
    session_id: str
    tutor_id: str

@router.patch("/session/tutor")
async def set_session_tutor(req: SessionTutorUpdate, current_email: str = Depends(get_current_email)):
    try:
        doc_ref = db.collection(COLL).document(req.session_id)
        snap = await doc_ref.get()
        if not snap.exists:
            raise HTTPException(404, "Session not found")
        data = snap.to_dict() or {}
        if data.get("user_email") != current_email:
            raise HTTPException(403, "Forbidden")
        meta = await get_tutor_meta(req.tutor_id)
        await doc_ref.update({
            "tutor_id": req.tutor_id,
            "tutor_title": meta.get("title"),
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        return {"ok": True, "tutor_id": req.tutor_id, "tutor_title": meta.get("title")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error updating session tutor: {str(e)}")

@router.post("/chat/stream")
async def chat_stream_handler(
    data: Dict[str, Any] = Body(...),
    current_email: str = Depends(get_current_email),
):
    """
    Streaming variant of /chat that emits SSE events:
      - event: meta   { tutor_title, topic_mode, topics?, matched_points? }
      - event: token  { "text": "..." }             # incremental model tokens
      - event: final  { ...full payload... }
      - event: error  { "message": "..." }
    Produces highly structured, beautified Markdown answers (headings, bullets, callouts, KaTeX-friendly math).
    """

    quick = bool((data.get("quick") or data.get("mode") == "scan"))

    import json, re, asyncio, threading
    from collections import defaultdict
    from typing import List, Dict, Tuple

    # ========== Small helpers (server-side beautify) ==========

    # TIP/WARN/INFO/NOTE -- lines → callouts your renderer styles nicely
    def _auto_callouts(md: str) -> str:
        if not md:
            return md
        # Match start-of-line labels (lenient dashes/colons)
        rx = re.compile(r'^(?:\s*)(TIP|INFO|NOTE|WARN|WARNING)\s*(?:—|-|:)?\s*(.*)$', re.IGNORECASE | re.MULTILINE)
        def rep(m):
            kind = m.group(1).upper()
            rest = (m.group(2) or "").strip()
            return f"> [!{kind}] {rest}"
        return rx.sub(rep, md)

    # Conservatively mathify common log/ln equations if the user/model wrote plain ascii
    # - Wrap equation-ish lines with $...$
    # - Convert log_/log/ln → \log_/ \log / \ln
    # - Special-case change-of-base into a pretty fraction
    def _auto_mathify(md: str) -> str:
        if not md:
            return md

        # 1) Change-of-base canonicalization (use callable to avoid backslash escapes)
        cob_rx = re.compile(
            r'(?im)^\s*Change of base:\s*log_?[a-zA-Z0-9]+\s*x\s*=\s*log_?[a-zA-Z0-9]+\s*x\s*/\s*log_?[a-zA-Z0-9]+\s*[a-zA-Z0-9]+\s*(?:.*)$'
        )
        md = cob_rx.sub(
            lambda _m: r"Change of base: $$\log_b x = \dfrac{\log_k x}{\log_k b}\quad (k>0,\;k\neq 1)$$",
            md,
        )

        # 2) Do not touch fenced code blocks
        parts = re.split(r"(```[\s\S]*?```)", md)

        def _mathify_chunk(chunk: str) -> str:
            lines = chunk.split("\n")
            out = []
            for line in lines:
                if ("=" in line) and re.search(r'(^|\s)(log_|log\b|ln\b)', line, re.IGNORECASE) and ("$" not in line):
                    t = line
                    # Use callables so backslashes remain literal
                    t = re.sub(r'log_',   lambda m: r'\log_', t)
                    t = re.sub(r'\blog\b', lambda m: r'\log',  t)
                    t = re.sub(r'\bln\b',  lambda m: r'\ln',   t)
                    out.append(f"${t}$")
                else:
                    out.append(line)
            return "\n".join(out)

        for i in range(0, len(parts), 2):
            parts[i] = _mathify_chunk(parts[i])

        return "".join(parts)

    # Remove any internal GROUNDING echoes if a model ever leaks them
    def _strip_grounding_from_text(text: str) -> str:
        if not text:
            return text
        # Known headers we never want to show
        text = re.sub(
            r'(^|\n)\*{0,2}Relevant .*?Syllabus Points.*?\n(?:- .*?\n)+',
            r'\1',
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()

    def _postprocess_markdown(text: str) -> str:
        if not text:
            return text
        t = _strip_grounding_from_text(text)
        t = _auto_callouts(t)
        t = _auto_mathify(t)
        return t

    # ===== Input & basic checks =====
    session_id: str = (data.get("session_id") or "").strip()
    message: str = (data.get("message") or "").strip()

    if not session_id:
        return StreamingResponse(
            iter([_sse("error", {"message": "session_id required"})]),
            media_type="text/event-stream",
        )
    if not message:
        return StreamingResponse(
            iter([_sse("error", {"message": "message required"})]),
            media_type="text/event-stream",
        )
    if not _oai_ready():
        return StreamingResponse(
            iter([_sse("error", {"message": "OpenAI client not available on server"})]),
            media_type="text/event-stream",
        )

    # ===== Strong structure & math/callout contract =====
    FORMATTING_RULES = (
        "FORMAT RULES:\n"
        "- Be concise: max 4–6 bullets and/or 1 short paragraph.\n"
        "- Start with a 1–2 sentence direct answer titled '**Answer:**'.\n"
        "- Use Markdown lists and short lines; keep it scannable.\n"
        "- Use KaTeX LaTeX for math: inline `$...$`, display `$$...$$`, and ```math blocks.\n"
        "- Prefer `\\log`, `\\ln`, subscripts like `\\log_b`, and fractions `\\dfrac{...}{...}`.\n"
        "- Use one `[!TIP]` or `[!WARN]` callout for key advice/warnings.\n"
        "- Do NOT include syllabus codes like [1.1.3d].\n"
    )

    STRUCTURE_TEMPLATE = (
        "**Answer:** ${short_answer}\n\n"
        "- ${point1}\n"
        "- ${point2}\n"
        "- ${point3}\n"
        "- ${point4}\n\n"
        "> [!TIP] ${tip}\n"
    )

    # ===== Fetch session + limits =====
    doc_ref = db.collection(COLL).document(session_id)
    snap = await doc_ref.get()
    if not snap.exists:
        return StreamingResponse(
            iter([_sse("error", {"message": "Session not found"})]),
            media_type="text/event-stream",
        )
    if snap.get("user_email") != current_email:
        return StreamingResponse(
            iter([_sse("error", {"message": "Forbidden"})]),
            media_type="text/event-stream",
        )

    data0 = snap.to_dict() or {}

    tutor_id = data0.get("tutor_id") or DEFAULT_TUTOR_ID
    tutor_meta = await get_tutor_meta(tutor_id)
    tutor_title = tutor_meta.get("title") or DEFAULT_TUTOR_TITLE

    history = data0.get("messages", [])[-10:]
    limit = int(data0.get("user_msg_limit", 20))
    count = int(data0.get("user_msg_count", 0))
    remaining_before = max(0, limit - count)
    if remaining_before <= 0:
        brief = "You’ve reached the 20-message limit for this session. Please start a new session to continue."
        return StreamingResponse(
            iter(
                [
                    _sse(
                        "final",
                        {
                            "reply": brief,
                            "matched_points": [],
                            "citations": [],
                            "remaining_messages": 0,
                            "user_msg_limit": limit,
                            "limit_reached": True,
                        },
                    )
                ]
            ),
            media_type="text/event-stream",
        )

    # ===== Smalltalk (streamed) =====
    if _is_smalltalk(message):
        async def smalltalk_gen():
            brief = f"Hello — how can I help you with {tutor_title} today?"
            await _inc_user_msg_count(doc_ref, limit)
            await _append_user_message(session_id, message)
            await _append_assistant_message(session_id, brief)
            yield _sse("token", {"text": brief})
            yield _sse(
                "final",
                {
                    "reply": brief,
                    "matched_points": [],
                    "citations": [],
                    "remaining_messages": max(0, remaining_before - 1),
                    "user_msg_limit": limit,
                    "limit_reached": False,
                },
            )

        return StreamingResponse(smalltalk_gen(), media_type="text/event-stream")

    # ===== Build matches (recall → optional rerank → selection) =====
    idx = await ensure_index_loaded(tutor_id)
    matches = idx.match(message, top_k=RERANK_CANDIDATES) if idx and idx.loaded else []

    if matches and not quick:
        try:
            matches = await _llm_rerank_points(message, matches)
        except Exception as e:
            logger.warning("LLM rerank failed (%s); using original ordering", e)

    # LLM selection gate (skip for quick)
    if matches and not quick:
        try:
            matches = await _llm_select_points(
                message, matches,
                top_k=SELECT_TOP_K,
                max_keep=SELECT_MAX_KEEP,
                min_keep=SELECT_MIN_KEEP
            )
        except Exception as e:
            logger.warning("LLM selection failed (%s); keeping original top-%d", e, SELECT_MAX_KEEP)
            matches = matches[:SELECT_MAX_KEEP]

    max_score = max((float(s) for (_p, s) in matches), default=0.0)
    if not matches or max_score < MIN_RELEVANCE:

        async def no_match_gen():
            brief = (
                f"I couldn’t confidently match your request to the {tutor_title} syllabus. "
                "Try naming the topic or formula (e.g., “@explain Newton’s second law” or “supply and demand diagram”)."
            )
            await _inc_user_msg_count(doc_ref, limit)
            await _append_user_message(session_id, message)
            await _append_assistant_message(session_id, brief)
            yield _sse("token", {"text": brief})
            yield _sse(
                "final",
                {
                    "reply": brief,
                    "matched_points": [],
                    "citations": [f"firestore:{tutor_id}"],
                    "remaining_messages": max(0, remaining_before - 1),
                    "user_msg_limit": limit,
                    "limit_reached": False,
                },
            )

        return StreamingResponse(no_match_gen(), media_type="text/event-stream")

    # ===== Flatten points for payload/UI =====
    matched_points: List[Dict[str, Any]] = []
    for (p, score) in matches:
        matched_points.append(
            {
                "paper_id": p["paper_id"],
                "topic_id": p["topic_id"],
                "topic_name": p.get("topic_name"),
                "point_id": p["point_id"],
                "statement": p["statement"],
                "equations": p.get("equations") or [],
                "maths_skills": p.get("maths_skills") or [],
                "tags": p.get("tags") or [],
                "score": round(float(score), 3),
            }
        )

    if quick:
        async def quick_gen():
            # send meta right away so UI can render chips
            yield _sse("meta", {
                "tutor_title": tutor_title,
                "topic_mode": False,
                "topics": None,
                "matched_points": matched_points,
            })
            # optionally give a tiny token so the text area isn’t empty
            yield _sse("token", {"text": "Scan complete.\n"})
            # final without LLM/media
            yield _sse("final", {
                "reply": "Scan complete.",
                "matched_points": matched_points,
                "citations": [f"firestore:{tutor_id}"],
                "remaining_messages": max(0, remaining_before),
                "user_msg_limit": limit,
                "limit_reached": False,
                "topic_mode": False,
                "topics": None,
                "media": {"show": False, "confidence": 0.0, "reason": "quick"}
            })
        return StreamingResponse(quick_gen(), media_type="text/event-stream")


    # ===== Topic mode decision =====
    def _decide_topic_mode(mps: List[Dict[str, Any]], msg: str) -> Tuple[bool, List[Dict[str, Any]]]:
        if not mps:
            return False, []
        token_count = len(re.findall(r"[A-Za-z0-9]+", msg))
        by_topic: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for mp in mps:
            k = (mp.get("paper_id") or "", mp.get("topic_id") or "", mp.get("topic_name") or "")
            by_topic[k].append(mp)

        topic_count = len(by_topic)
        scores = sorted([mp["score"] for mp in mps], reverse=True)
        top = scores[0] if scores else 0.0
        k3 = scores[2] if len(scores) >= 3 else (scores[-1] if scores else 0.0)
        spread = top - k3

        vague = (token_count <= 3) or (topic_count >= 3 and top < 0.62) or (top < 0.58 and spread < 0.12)
        if not vague:
            return False, []

        topics: List[Dict[str, Any]] = []
        for (paper_id, topic_id, topic_name), items in by_topic.items():
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
            topics.append(
                {
                    "is_topic": True,
                    "paper_id": paper_id,
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "score": round(max((x["score"] for x in items), default=0.0), 3),
                    "points_preview": [
                        {
                            "point_id": it["point_id"],
                            "statement": it["statement"],
                            "equations": it.get("equations") or [],
                            "tags": it.get("tags") or [],
                            "score": it["score"],
                        }
                        for it in items_sorted[:6]
                    ],
                }
            )
        topics.sort(key=lambda t: t["score"], reverse=True)
        return True, topics

    topic_mode, topic_matches = _decide_topic_mode(matched_points, message)

    # ===== Style + system preamble (beautified & bullet-guided) =====
    style = data.get("style") or DEFAULT_STYLE
    style_text = _style_instructions(style)

    system_preamble = (
        f"You are Atlas, a concise, structured tutor. Use '{tutor_title}' definitions "
        "when relevant, but never quote syllabus bullets or codes.\n\n"
        + style_text
        + "\n"
        + FORMATTING_RULES
        + "\nFill the following template naturally; omit unused sections:\n"
        + STRUCTURE_TEMPLATE
        + "\nIMPORTANT OUTPUT CONTRACT:\n"
          "* Use LaTeX for math: `$...$` inline, `$$...$$` for display, or fenced ```math blocks.\n"
          "* Prefer `\\log`, `\\ln`, subscripts `\\log_b`, and `\\dfrac{...}{...}` for fractions.\n"
          "* If giving tips/warnings, write them as callouts using `> [!TIP] ...` or `> [!WARN] ...`.\n"
          "* Keep the entire answer visually scannable in chat (short lines, compact lists).\n"
    )

    oa_messages = [{"role": "system", "content": system_preamble}]
    for m in history:
        role = "assistant" if m.get("role") == "assistant" else "user"
        oa_messages.append({"role": role, "content": m.get("content", "")})
    oa_messages.append({"role": "user", "content": message})

    # ===== Grounding set =====
    if topic_mode and topic_matches:
        top_topic = topic_matches[0]
        use_points = []
        for pp in top_topic.get("points_preview", [])[:GROUNDING_TOP_N]:
            use_points.append(
                {
                    "paper_id": top_topic.get("paper_id"),
                    "topic_id": top_topic.get("topic_id"),
                    "topic_name": top_topic.get("topic_name"),
                    "point_id": pp.get("point_id"),
                    "statement": pp.get("statement"),
                    "equations": pp.get("equations") or [],
                    "maths_skills": [],
                    "tags": pp.get("tags") or [],
                    "score": pp.get("score", 0.0),
                }
            )
    else:
        use_points = matched_points[:GROUNDING_TOP_N]

    grounding_payload = [
        {
            "id": mp["point_id"],
            "topic": mp.get("topic_name"),
            "eqs": (mp.get("equations") or [])[:2],
            "statement": (mp.get("statement") or "")[:200],
        }
        for mp in use_points
    ]
    oa_messages.insert(
        1,
        {
            "role": "system",
            "content": "GROUNDING (do not echo; use only for internal reasoning): "
            + json.dumps(grounding_payload, ensure_ascii=False),
        },
    )

    client = _oai_client()

    # ===== Stream generator =====
    async def gen():
        full: List[str] = []

        # Pre-meta so the UI can render chips/peek
        yield _sse(
            "meta",
            {
                "tutor_title": tutor_title,
                "topic_mode": topic_mode,
                "topics": topic_matches if topic_mode else None,
                "matched_points": [] if topic_mode else matched_points,
            },
        )

        # Try true streaming; fallback to chunking if needed
        try:
            loop = asyncio.get_running_loop()
            q: asyncio.Queue = asyncio.Queue()

            def _producer():
                try:
                    for chunk in client.chat.completions.create(
                        model=OPENAI_MODEL, messages=oa_messages, stream=True
                    ):
                        delta = (chunk.choices[0].delta.content or "")
                        if delta:
                            full.append(delta)
                            loop.call_soon_threadsafe(q.put_nowait, ("token", {"text": delta}))
                except Exception as e:
                    logger.exception("OpenAI stream failed: %s", e)
                    loop.call_soon_threadsafe(q.put_nowait, ("__error__", {"message": str(e)}))
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, ("__done__", None))

            threading.Thread(target=_producer, daemon=True).start()

            while True:
                evt, payload = await q.get()
                if evt == "__done__":
                    break
                if evt == "__error__":
                    raise RuntimeError(payload.get("message") if isinstance(payload, dict) else "stream error")
                if evt == "token":
                    yield _sse("token", payload)

        except Exception as e:
            # Fallback: non-stream call, then fake streaming by chunking
            try:
                completion = client.chat.completions.create(model=OPENAI_MODEL, messages=oa_messages)
                txt = (completion.choices[0].message.content or "")
            except Exception as e2:
                yield _sse("error", {"message": f"Model call failed: {e2}"})
                return
            for piece in _chunk_text(txt, 28):
                full.append(piece)
                yield _sse("token", {"text": piece})

        # Finalize, clean, beautify, persist
        final_text_raw = "".join(full)
        # Post-process first, then apply LaTeX fixes
        beautified = _postprocess_markdown(final_text_raw)
        final_text = fix_latex(beautified)

        await _inc_user_msg_count(doc_ref, limit)
        await _append_user_message(session_id, message)
        await _append_assistant_message(session_id, final_text, points=use_points[:4])

        media = {"show": False, "confidence": 0.0, "reason": "skipped"}
        if not quick:
            media = await infer_image_hint_ai(message, matched_points)

        yield _sse(
            "final",
            {
                "reply": final_text,
                "matched_points": [] if topic_mode else matched_points,
                "citations": [f"firestore:{tutor_id}"],
                "remaining_messages": max(0, remaining_before - 1),
                "user_msg_limit": limit,
                "limit_reached": False,
                "topic_mode": topic_mode,
                "topics": topic_matches if topic_mode else None,
                "media": media,
            },
        )

    return StreamingResponse(gen(), media_type="text/event-stream")

def _auto_flashcard_title(session_title: Optional[str], history: List[Dict[str, Any]]) -> str:
    # Prefer the latest non-empty user message as topic
    topic = None
    for m in reversed(history):
        if m.get("role") == "user":
            txt = (m.get("content") or "").strip()
            if txt:
                topic = " ".join(txt.split()[:6])
                break
    if not topic:
        topic = (session_title or "Study set")
    return f"Flashcards — {topic[:40]}"

def _normalize_cards(raw_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in raw_cards:
        q = (c.get("q") or c.get("question") or "").strip()
        a = (c.get("a") or c.get("answer") or "").strip()
        if not q or not a:
            continue
        hint = (c.get("hint") or "").strip() or None
        cloze = bool(c.get("cloze")) if isinstance(c.get("cloze"), (bool, int)) else False
        tags = c.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        out.append({"q": q[:400], "a": a[:800], "hint": (hint[:300] if hint else None), "cloze": cloze, "tags": tags[:5]})
    return out[:20]

async def _make_flashcards_from_llm(title: str, convo: str, num_cards: int) -> Dict[str, Any]:
    """
    Use OPENAI_MODEL to create a JSON flashcard set from the conversation.
    """
    system = (
        "You are an expert GCSE tutor. Create high‑yield flashcards from the conversation. "
        "Each card should be exam‑useful, concise, and unambiguous. Use Markdown for math (e.g., $F=ma$). "
        f"Return STRICT JSON only: {{\"title\":\"{title}\",\"cards\":[{{\"q\":\"...\",\"a\":\"...\",\"hint\":\"...\",\"cloze\":false,\"tags\":[\"physics\"]}}, ...]}} "
        f"Generate {num_cards} cards. Avoid duplication."
    )
    client = _oai_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Conversation:\n{convo[:6000]}"},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        # best‑effort JSON recovery
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end+1])
        else:
            raise HTTPException(500, "Model returned invalid JSON for flashcards.")

    title_out = (data.get("title") or title or "Flashcards").strip()[:60]
    cards = _normalize_cards(data.get("cards") or [])
    if len(cards) < 4:
        raise HTTPException(500, "Model returned too few flashcards.")
    return {"title": title_out, "cards": cards}

@router.post("/flashcards/generate", response_model=FlashcardGenerateOut)
async def generate_flashcards(payload: FlashcardGenerateIn, current_email: str = Depends(get_current_email)):
    # Load session
    sid = (payload.session_id or "").strip()
    if not sid:
        raise HTTPException(400, "session_id required")

    doc_ref = db.collection(COLL).document(sid)
    snap = await doc_ref.get()
    if not snap.exists:
        raise HTTPException(404, "Session not found")

    sess = snap.to_dict() or {}
    if sess.get("user_email") != current_email:
        raise HTTPException(403, "Forbidden")

    history: List[Dict[str, Any]] = sess.get("messages", [])
    if sum(1 for m in history if m.get("role") == "user") < 1:
        raise HTTPException(400, "At least one user message is required to generate flashcards.")

    # Build STRICT corpus from this session only (messages + any points already attached)
    msgs, pts, corpus = _gather_session_corpus(sess)

    # Title (prefer latest user utterance; fallback to session title)
    title = _auto_flashcard_title(sess.get("title"), history)

    # Generate strictly from corpus + validate answers against corpus
    try:
        fc = await _generate_flashcards_strict_from_corpus(
            num_cards=payload.num_cards,
            messages=msgs,
            points=pts,
            corpus_text=corpus
        )
    except Exception as e:
        logger.exception("Flashcard generation failed: %s", e)
        raise HTTPException(500, handle_openai_error(e))

    # Normalise and clip
    cards = _normalize_cards(fc.get("cards") or [])
    if len(cards) < 4:
        raise HTTPException(400, "Not enough in-session material to make reliable flashcards yet.")

    # Persist (only if requested)
    fid = uuid4().hex
    tutor_id = sess.get("tutor_id") or DEFAULT_TUTOR_ID
    tutor_meta = await get_tutor_meta(tutor_id)
    now_iso_str = now_iso()

    set_doc = {
        "id": fid,
        "user_email": current_email,
        "session_id": sid,
        "tutor_id": tutor_id,
        "tutor_title": tutor_meta.get("title"),
        "title": title if fc.get("title") == "Flashcards — Session" else fc.get("title", title),
        "cards": cards,
        "card_count": len(cards),
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }

    saved = False
    if bool(payload.save_to_progress):
        try:
            await db.collection(FLASHCARDS_COLL).document(fid).set(set_doc)
            saved = True
        except Exception as e:
            logger.exception("Failed saving flashcards: %s", e)
            raise HTTPException(500, "Could not save flashcards.")

    # Append a chat message that includes the cards (LIST) so the UI can render the deck inline
    await _append_assistant_message(
        sid,
        content=f"Flashcards ready: **{set_doc['title']}** ({len(cards)} cards).",
        msg_type="flashcards",
        flashcards=cards,  # <-- list, matches Message.flashcards (schema fix)
    )

    # Build response payload (mirror Firestore timestamps best-effort)
    out = dict(set_doc)
    out["created_at"] = now_iso_str if not saved else None  # replaced below if Firestore obj includes .isoformat
    out["updated_at"] = now_iso_str if not saved else None
    out["id"] = fid

    # If we saved, fetch back iso timestamps if present on the local object (best effort)
    try:
        if saved:
            for k in ("created_at", "updated_at"):
                ts = out.get(k)
                if hasattr(ts, "isoformat"):
                    out[k] = ts.isoformat()
                else:
                    out[k] = None
    except Exception:
        pass

    return {"set": out, "saved_to_progress": saved}


@router.get("/flashcards/sets", response_model=List[FlashcardSetSummary])
async def list_flashcard_sets(current_email: str = Depends(get_current_email)):
    try:
        # Preferred: server-side order (requires composite index)
        q = (
            db.collection(FLASHCARDS_COLL)
            .where("user_email", "==", current_email)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(100)
        )
        items: List[FlashcardSetSummary] = []
        async for doc in q.stream():
            d = doc.to_dict() or {}
            ts = d.get("created_at")
            items.append(FlashcardSetSummary(
                id=d.get("id") or doc.id,
                session_id=d.get("session_id"),
                title=d.get("title") or "Flashcards",
                card_count=int(d.get("card_count") or len(d.get("cards") or [])),
                tutor_id=d.get("tutor_id"),
                tutor_title=d.get("tutor_title"),
                created_at=(ts.isoformat() if hasattr(ts, "isoformat") else None),
            ))
        return items

    except FailedPrecondition:
        # Fallback: run without order_by, then sort in-memory
        q = (
            db.collection(FLASHCARDS_COLL)
            .where("user_email", "==", current_email)
            .limit(200)  # pull a bit more, then slice
        )
        docs = []
        async for doc in q.stream():
            docs.append(doc)

        def _sort_key(doc):
            d = doc.to_dict() or {}
            ts = d.get("created_at")
            try:
                return ts.timestamp() if hasattr(ts, "timestamp") else 0.0
            except Exception:
                return 0.0

        docs.sort(key=_sort_key, reverse=True)
        items: List[FlashcardSetSummary] = []
        for doc in docs[:100]:
            d = doc.to_dict() or {}
            ts = d.get("created_at")
            items.append(FlashcardSetSummary(
                id=d.get("id") or doc.id,
                session_id=d.get("session_id"),
                title=d.get("title") or "Flashcards",
                card_count=int(d.get("card_count") or len(d.get("cards") or [])),
                tutor_id=d.get("tutor_id"),
                tutor_title=d.get("tutor_title"),
                created_at=(ts.isoformat() if hasattr(ts, "isoformat") else None),
            ))
        return items

@router.get("/flashcards/session/{sid}", response_model=List[FlashcardSetSummary])
async def list_flashcard_sets_for_session(sid: str, current_email: str = Depends(get_current_email)):
    # Authz
    doc_ref = db.collection(COLL).document(sid)
    snap = await doc_ref.get()
    if not snap.exists:
        raise HTTPException(404, "Session not found")
    sess = snap.to_dict() or {}
    if sess.get("user_email") != current_email:
        raise HTTPException(403, "Forbidden")

    try:
        # Preferred: equality on two fields + order (needs composite index with 3 fields)
        q = (
            db.collection(FLASHCARDS_COLL)
            .where("user_email", "==", current_email)
            .where("session_id", "==", sid)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(20)
        )
        items: List[FlashcardSetSummary] = []
        async for doc in q.stream():
            d = doc.to_dict() or {}
            ts = d.get("created_at")
            items.append(FlashcardSetSummary(
                id=d.get("id") or doc.id,
                session_id=d.get("session_id"),
                title=d.get("title") or "Flashcards",
                card_count=int(d.get("card_count") or len(d.get("cards") or [])),
                tutor_id=d.get("tutor_id"),
                tutor_title=d.get("tutor_title"),
                created_at=(ts.isoformat() if hasattr(ts, "isoformat") else None),
            ))
        return items

    except FailedPrecondition:
        # Fallback: filter by user_email on server, then filter session_id + sort locally
        q = (
            db.collection(FLASHCARDS_COLL)
            .where("user_email", "==", current_email)
            .limit(200)
        )
        docs = []
        async for doc in q.stream():
            docs.append(doc)

        rows = []
        for doc in docs:
            d = doc.to_dict() or {}
            if d.get("session_id") == sid:
                rows.append(doc)

        def _sort_key(doc):
            d = doc.to_dict() or {}
            ts = d.get("created_at")
            try:
                return ts.timestamp() if hasattr(ts, "timestamp") else 0.0
            except Exception:
                return 0.0

        rows.sort(key=_sort_key, reverse=True)
        out: List[FlashcardSetSummary] = []
        for doc in rows[:20]:
            d = doc.to_dict() or {}
            ts = d.get("created_at")
            out.append(FlashcardSetSummary(
                id=d.get("id") or doc.id,
                session_id=d.get("session_id"),
                title=d.get("title") or "Flashcards",
                card_count=int(d.get("card_count") or len(d.get("cards") or [])),
                tutor_id=d.get("tutor_id"),
                tutor_title=d.get("tutor_title"),
                created_at=(ts.isoformat() if hasattr(ts, "isoformat") else None),
            ))
        return out

@router.get("/flashcards/set/{fid}", response_model=FlashcardSet)
async def get_flashcard_set(fid: str, current_email: str = Depends(get_current_email)):
    ref = db.collection(FLASHCARDS_COLL).document(fid)
    snap = await ref.get()
    if not snap.exists:
        raise HTTPException(404, "Flashcard set not found")
    d = snap.to_dict() or {}
    if d.get("user_email") != current_email:
        raise HTTPException(403, "Forbidden")
    out = FlashcardSet(
        id=d.get("id") or fid,
        user_email=d.get("user_email"),
        session_id=d.get("session_id"),
        tutor_id=d.get("tutor_id"),
        tutor_title=d.get("tutor_title"),
        title=d.get("title") or "Flashcards",
        cards=[Flashcard(**c) for c in d.get("cards") or []],
        card_count=int(d.get("card_count") or len(d.get("cards") or [])),
        created_at=(d.get("created_at").isoformat() if hasattr(d.get("created_at"), "isoformat") else None),
        updated_at=(d.get("updated_at").isoformat() if hasattr(d.get("updated_at"), "isoformat") else None),
    )
    return out

# ================================
# Flashcards (STRICT: chat + syllabus points only)
# ================================

def _sentences(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = re.split(r"(?<=[.!?])\s+", s)
    return [p.strip() for p in parts if 8 <= len(p.strip()) <= 240]

def _gather_session_corpus(session_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """
    Returns (messages, points, joined_text_corpus) using ONLY what exists in the session.
    - messages: [{"id": str(idx), "role": "user|assistant", "text": str}]
    - points:   list of point dicts pulled from assistant messages' 'points' field
    - corpus:   plain text of all message texts + syllabus statements (already attached)
    """
    msgs_raw = session_data.get("messages", [])
    msgs: List[Dict[str, Any]] = []
    pts: List[Dict[str, Any]] = []

    for idx, m in enumerate(msgs_raw):
        role = m.get("role")
        txt = (m.get("content") or "").strip()
        if role in ("user", "assistant") and txt:
            msgs.append({"id": str(idx), "role": role, "text": txt})
        if role == "assistant":
            for mp in (m.get("points") or []):
                st = (mp.get("statement") or "").strip()
                pid = (mp.get("point_id") or "").strip()
                if st and pid:
                    pts.append(mp)

    # dedupe by point_id
    seen = set()
    uniq_pts = []
    for p in pts:
        pid = p.get("point_id")
        if pid not in seen:
            uniq_pts.append(p); seen.add(pid)

    parts = []
    for m in msgs:
        parts.extend(_sentences(m["text"]))
    for p in uniq_pts:
        parts.extend(_sentences(p.get("statement") or ""))

    return msgs, uniq_pts, "\n".join(parts)

async def _generate_flashcards_strict_from_corpus(
    num_cards: int,
    messages: List[Dict[str, Any]],
    points: List[Dict[str, Any]],
    corpus_text: str
) -> Dict[str, Any]:
    """
    Generate flashcards using ONLY the provided corpus.
    Returns {"title": str, "cards": [Flashcard]}
    """
    # If model unavailable, fallback to simple cards straight from syllabus statements
    if not _oai_ready():
        basics = []
        for p in points[:num_cards]:
            stmt = (p.get("statement") or "").strip()
            if not stmt:
                continue
            basics.append({"q": f"State/define: {stmt[:80]}...", "a": stmt, "hint": None, "cloze": False, "tags": ["physics"]})
        title = "Flashcards — Session"
        return {"title": title, "cards": basics[:max(4, num_cards)]}

    system = (
        "You are generating GCSE study flashcards strictly from the given CORPUS. "
        "You MUST NOT add any information that is not present in the corpus. "
        "Prefer clear definitions, laws, equations, key steps, and short worked facts. "
        "Return STRICT JSON only: "
        "{\"cards\":[{\"q\":str,\"a\":str,\"hint\":str|null,\"cloze\":false,\"tags\":[\"physics\"]}, ...]} "
        f"Create up to {num_cards} useful cards. Keep answers concise (≤3 short lines)."
    )

    trimmed = "\n".join((corpus_text or "").split("\n")[:900])
    payload = {
        "corpus": {
            "messages": messages[:60],
            "syllabus_points": [
                {"point_id": p.get("point_id"), "statement": p.get("statement")}
                for p in points[:60]
            ]
        },
        "corpus_text": trimmed
    }

    client = _oai_client()
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system},
                      {"role":"user","content": json.dumps(payload, ensure_ascii=False)}]
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        items = data.get("cards", [])
    except Exception as e:
        logger.warning("flashcards strict LLM failed: %s", e)
        items = []

    # Normalise -> Flashcard shape
    cards = _normalize_cards(items)

    # Validate against corpus using fuzzy match (drop hallucinations)
    lines = _sentences(corpus_text)
    validated: List[Dict[str, Any]] = []
    for c in cards:
        a = c["a"]
        best = process.extractOne(a, lines, scorer=fuzz.partial_ratio) if lines else None
        score = best[1] if best else 0
        if score >= 70:
            validated.append(c)
        if len(validated) >= num_cards:
            break

    # If not enough, synthesize simple cards from existing syllabus points
    if len(validated) < max(4, min(8, num_cards)):
        for p in points:
            if len(validated) >= num_cards:
                break
            stmt = (p.get("statement") or "").strip()
            if not stmt:
                continue
            if any(fuzz.partial_ratio(stmt, v["a"]) >= 85 for v in validated):
                continue
            validated.append({"q": f"State or define: {stmt[:80]}...", "a": stmt, "hint": None, "cloze": False, "tags": ["physics"]})

    title = "Flashcards — Session"
    return {"title": title, "cards": validated[:num_cards]}
