import os
import re
import unicodedata
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from rapidfuzz import fuzz

MODEL_NAME = os.getenv('MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
API_TOKEN = os.getenv('LINKAGE_API_TOKEN', '').strip()

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

app = FastAPI(title='Linkage Semantic API', version='1.0.0')

STORE_SYNONYMS = {
    'PIZARIA': 'PIZZARIA',
    'PIZZARIAS': 'PIZZARIA',
    'PIZZAARIA': 'PIZZARIA',
    'BURGUER': 'BURGER',
    'BURGUER': 'BURGER',
    'GOURMETT': 'GOURMET'
}
UNIT_TERMS = {'LOJA', 'UNIDADE', 'FILIAL', 'BOX', 'PONTO'}
ADDRESS_SYNONYMS = {
    'R': 'RUA', 'R.': 'RUA', 'AV': 'AVENIDA', 'AV.': 'AVENIDA',
    'ROD': 'RODOVIA', 'ROD.': 'RODOVIA', 'TRV': 'TRAVESSA', 'TRV.': 'TRAVESSA'
}
STOPWORDS = {'DE', 'DA', 'DO', 'DAS', 'DOS', 'E', 'LTDA', 'ME', 'EIRELI', 'SA'}
COMMON_SURNAMES = {'SILVA','SANTOS','OLIVEIRA','SOUZA','SOUSA','PEREIRA','FERREIRA','LIMA','ALVES','RODRIGUES','COSTA','GOMES','NASCIMENTO'}


class EntityPayload(BaseModel):
    store_name: str = ''
    address: str = ''
    partner_names: List[str] = []


class OptionsPayload(BaseModel):
    check_store_name: bool = True
    check_address: bool = True
    check_partner_names: bool = True
    check_kinship_hint: bool = True


class LinkagePayload(BaseModel):
    case_id: Optional[str] = ''
    principal: EntityPayload
    linked: EntityPayload
    options: OptionsPayload = OptionsPayload()


def _strip_accents(text: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFKD', text or '') if not unicodedata.combining(ch))


def normalize_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def normalize_store_name(text: str) -> str:
    s = normalize_spaces(_strip_accents(text).upper())
    s = re.sub(r'[^A-Z0-9\s]', ' ', s)
    tokens = []
    for tok in s.split():
        tok = STORE_SYNONYMS.get(tok, tok)
        if tok in UNIT_TERMS:
            continue
        if re.fullmatch(r'(LOJA|UNIDADE)?\d+', tok):
            continue
        tokens.append(tok)
    return ' '.join(tokens)


def canonical_store_name(a: str, b: str) -> str:
    na = normalize_store_name(a)
    nb = normalize_store_name(b)
    ta = na.split()
    tb = nb.split()
    common = []
    tb_set = set(tb)
    for tok in ta:
        if tok in tb_set and tok not in common:
            common.append(tok)
    if common:
        return normalize_spaces(' '.join(common)).title()
    return normalize_spaces(na or nb).title()


def normalize_address(text: str) -> str:
    s = normalize_spaces(_strip_accents(text).upper())
    s = re.sub(r'[,;\-]', ' ', s)
    s = re.sub(r'\bN[ºO]?\b', ' ', s)
    out = []
    for tok in s.split():
        out.append(ADDRESS_SYNONYMS.get(tok, tok))
    return ' '.join(out)


def canonical_address(a: str, b: str) -> str:
    # escolhe o mais completo dentre os originais, mantendo o texto mais legível
    a = normalize_spaces(a)
    b = normalize_spaces(b)
    return a if len(a) >= len(b) else b


def normalize_person_name(text: str) -> str:
    s = normalize_spaces(_strip_accents(text).upper())
    s = re.sub(r'[^A-Z\s]', ' ', s)
    return ' '.join(tok for tok in s.split() if tok not in STOPWORDS)


def person_tokens(text: str) -> List[str]:
    return [t for t in normalize_person_name(text).split() if len(t) > 1]


@lru_cache(maxsize=1)
def get_model():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(MODEL_NAME)


def semantic_score(a: str, b: str) -> float:
    a = (a or '').strip()
    b = (b or '').strip()
    if not a or not b:
        return 0.0
    model = get_model()
    if model is not None and util is not None:
        emb = model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        score = float(util.cos_sim(emb[0], emb[1]).item())
        return max(0.0, min(1.0, score))
    return max(0.0, min(1.0, fuzz.token_sort_ratio(a, b) / 100.0))


def confidence_label(score: float) -> str:
    if score >= 0.92:
        return 'Alta'
    if score >= 0.78:
        return 'Média'
    return 'Baixa'


def analyze_store_name(a: str, b: str):
    na, nb = normalize_store_name(a), normalize_store_name(b)
    lexical = fuzz.token_sort_ratio(na, nb) / 100.0 if na and nb else 0.0
    semantic = semantic_score(na or a, nb or b)
    score = max(lexical, semantic)
    match = score >= 0.82
    canonical = canonical_store_name(a, b) if match else ''
    reason = 'Diferença apenas por numeração de unidade e pequena variação ortográfica.' if match else 'A IA não encontrou semelhança forte o suficiente no nome da loja.'
    return {
        'match': match,
        'score': round(score, 4),
        'confidence_label': confidence_label(score),
        'canonical_name': canonical,
        'best_display': canonical,
        'reason': reason,
        'suggestion_text': canonical or 'Sem sugestão relevante'
    }


def analyze_address(a: str, b: str):
    na, nb = normalize_address(a), normalize_address(b)
    lexical = fuzz.token_sort_ratio(na, nb) / 100.0 if na and nb else 0.0
    semantic = semantic_score(na or a, nb or b)
    score = max(lexical, semantic)
    match = score >= 0.88
    canonical = canonical_address(a, b) if match else ''
    reason = 'Mesmo endereço-base com abreviação, pontuação ou variação ortográfica.' if match else 'A IA não encontrou semelhança forte o suficiente no endereço.'
    return {
        'match': match,
        'score': round(score, 4),
        'confidence_label': confidence_label(score),
        'canonical_address': canonical,
        'best_display': canonical,
        'reason': reason,
        'suggestion_text': canonical or 'Sem sugestão relevante'
    }


def analyze_partner_names(principal_names: List[str], linked_names: List[str]):
    best = None
    for p in principal_names or []:
        for l in linked_names or []:
            np, nl = normalize_person_name(p), normalize_person_name(l)
            lexical = fuzz.token_sort_ratio(np, nl) / 100.0 if np and nl else 0.0
            semantic = semantic_score(np or p, nl or l)
            score = max(lexical, semantic)
            item = (score, p, l)
            if best is None or item[0] > best[0]:
                best = item
    if not best:
        return {'match': False, 'score': 0.0, 'confidence_label': 'Baixa', 'canonical_name': '', 'best_principal': '', 'best_linked': '', 'reason': 'Não há nomes societários suficientes para análise.', 'suggestion_text': 'Sem sugestão relevante'}
    score, p, l = best
    match = score >= 0.9
    canonical = canonical_store_name(p, l) if match else ''
    return {
        'match': match,
        'score': round(score, 4),
        'confidence_label': confidence_label(score),
        'canonical_name': canonical,
        'best_principal': p,
        'best_linked': l,
        'reason': 'Nomes societários muito próximos após normalização.' if match else 'A IA não encontrou semelhança forte o suficiente nos nomes societários.',
        'suggestion_text': canonical or 'Sem sugestão relevante'
    }


def analyze_kinship_hint(principal_names: List[str], linked_names: List[str]):
    best = None
    for p in principal_names or []:
        tp = [t for t in person_tokens(p) if t not in COMMON_SURNAMES]
        if not tp:
            tp = person_tokens(p)
        for l in linked_names or []:
            tl = [t for t in person_tokens(l) if t not in COMMON_SURNAMES]
            if not tl:
                tl = person_tokens(l)
            common = [t for t in tp if t in tl]
            common_total = [t for t in person_tokens(p) if t in person_tokens(l)]
            score = 0.0
            if len(common) >= 1:
                score = 0.78
            elif len(common_total) >= 2:
                score = 0.72
            elif len(common_total) == 1:
                score = 0.52
            item = (score, p, l, common_total)
            if best is None or item[0] > best[0]:
                best = item
    if not best or best[0] < 0.72:
        return {'match': False, 'score': round(best[0],4) if best else 0.0, 'confidence_label': 'Baixa', 'principal_name': '', 'linked_name': '', 'display_text': 'Sem alerta relevante', 'reason': 'Não houve elementos suficientes para sugerir vínculo familiar provável.', 'suggestion_text': 'Sem alerta relevante'}
    score, p, l, common_total = best
    return {
        'match': True,
        'score': round(score, 4),
        'confidence_label': confidence_label(score),
        'principal_name': p,
        'linked_name': l,
        'display_text': 'Possível vínculo familiar',
        'reason': f'Sobrenomes/nomes em comum sugerem possível vínculo familiar ({", ".join(common_total[:3])}).',
        'suggestion_text': 'Possível vínculo familiar'
    }


def check_token(auth: Optional[str]):
    if not API_TOKEN:
        return
    if not auth or not auth.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Token não informado.')
    if auth.split(' ', 1)[1].strip() != API_TOKEN:
        raise HTTPException(status_code=401, detail='Token inválido.')


@app.get('/health')
def health(authorization: Optional[str] = Header(default=None)):
    check_token(authorization)
    return {'status': 'ok', 'model': MODEL_NAME}


@app.post('/v1/linkage/semantic-check')
def semantic_check(payload: LinkagePayload, authorization: Optional[str] = Header(default=None)):
    check_token(authorization)
    result = {
        'store_name': analyze_store_name(payload.principal.store_name, payload.linked.store_name) if payload.options.check_store_name else {},
        'address': analyze_address(payload.principal.address, payload.linked.address) if payload.options.check_address else {},
        'partner_names': analyze_partner_names(payload.principal.partner_names, payload.linked.partner_names) if payload.options.check_partner_names else {},
        'kinship_hint': analyze_kinship_hint(payload.principal.partner_names, payload.linked.partner_names) if payload.options.check_kinship_hint else {}
    }
    return result
