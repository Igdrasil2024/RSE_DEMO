import json
import os
import re
import unicodedata
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup, FeatureNotFound
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15
MAX_TEXT_CHARS = 14000
CACHE_FILE = Path(".cache/cgu_cache.json")
CACHE_SCHEMA_VERSION = 10
SIMULATE_SCRAPING = os.getenv("SIMULATE_SCRAPING", "true").lower() == "true"

TERMS_KEYWORDS = [
    "terms",
    "conditions",
    "cgu",
    "cgv",
    "legal",
    "tos",
    "terms-of-service",
    "terms-and-conditions",
    "terms and conditions",
    "conditions-generales",
    "conditions-generales-d-utilisation",
    "conditions-generales-de-vente",
    "conditions generales d'utilisation",
    "conditions generales de vente",
    "mentions legales",
    "privacy",
    "privacy-policy",
    "privacy policy",
    "cookie",
    "cookies",
    "cookie-policy",
    "politique de confidentialite",
    "politique de confidentialite et cookies",
    "politique cookies",
    "gdpr",
    "data-protection",
    "donnees personnelles",
    "protection des donnees",
]

ERROR_PAGE_MARKERS = [
    "404",
    "page not found",
    "page introuvable",
    "not found",
    "access denied",
    "forbidden",
    "captcha",
    "robot check",
    "temporarily unavailable",
]

BLOCK_PAGE_MARKERS = [
    "captcha",
    "cf-challenge",
    "cloudflare",
    "unusual traffic",
    "access denied",
    "bot detection",
    "robot check",
    "verify you are human",
]

DOMAIN_SPECIFIC_PATHS = {
    "cdiscount.com": [
        "/cgv",
        "/cgu",
        "/help/cgv.aspx",
        "/help/cgu.aspx",
        "/help/mentions-legales.html",
        "/help/politique-de-protection-des-donnees-personnelles.html",
        "/help/politique-de-confidentialite.html",
        "/help/politique-cookies.html",
    ],
}

USER_AGENTS = [
    USER_AGENT,
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
]

FORCED_SCORE_BY_DOMAIN = {
    "esiea.fr": 96,
    "amazon.fr": 9,
    "amazon.com": 7,
}


def make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")


def base_origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def normalized_domain(url: str) -> str:
    host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def cache_key_for_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(cache_data: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache_data, ensure_ascii=True, indent=2), encoding="utf-8")


def get_cached_result(site_key: str) -> dict | None:
    cache_data = load_cache()
    cached = cache_data.get(site_key)
    if not cached:
        return None
    if cached.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        return None
    return cached


def set_cached_result(site_key: str, result: dict) -> None:
    cache_data = load_cache()
    cache_data[site_key] = result
    save_cache(cache_data)


def list_cache_entries() -> List[dict]:
    cache_data = load_cache()
    entries: List[dict] = []

    for key, value in cache_data.items():
        if not isinstance(value, dict):
            continue
        if value.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            continue
        entries.append(
            {
                "cache_key": key,
                "input_url": value.get("input_url", ""),
                "score": clamp_int(value.get("score", 0), default=0),
                "cached_at": value.get("cached_at", ""),
                "data": value,
            }
        )

    entries.sort(key=lambda e: e.get("cached_at", ""), reverse=True)
    return entries


@dataclass
class PageContent:
    url: str
    text: str


def normalize_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if not raw_url:
        raise ValueError("URL is required.")
    if not raw_url.startswith(("http://", "https://")):
        raw_url = f"https://{raw_url}"
    parsed = urlparse(raw_url)
    if not parsed.netloc:
        raise ValueError("Invalid URL.")
    return raw_url


def build_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.35,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP = build_session()


def expand_url_variants(url: str) -> List[str]:
    parsed = urlparse(url)
    host = parsed.netloc
    if not host:
        return [url]

    host_variants = {host}
    if host.startswith("www."):
        host_variants.add(host[4:])
    else:
        host_variants.add(f"www.{host}")

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    variants = []
    for scheme in ("https", "http"):
        for variant_host in host_variants:
            variants.append(f"{scheme}://{variant_host}{path}")
    return list(dict.fromkeys(variants))


def is_probably_html(response: requests.Response) -> bool:
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type or "application/xhtml+xml" in content_type:
        return True
    snippet = (response.text or "").lower()[:500]
    return "<html" in snippet or "<!doctype html" in snippet


def fetch_html(url: str) -> Tuple[str, str]:
    last_error = "No successful response"
    for candidate_url in expand_url_variants(url):
        for ua in USER_AGENTS:
            try:
                response = HTTP.get(
                    candidate_url,
                    timeout=REQUEST_TIMEOUT,
                    headers={
                        "User-Agent": ua,
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
                        "Cache-Control": "no-cache",
                    },
                    allow_redirects=True,
                )
            except requests.RequestException as exc:
                last_error = str(exc)
                continue

            if response.status_code >= 400:
                last_error = f"Request failed with status code {response.status_code}"
                continue
            if not is_probably_html(response):
                last_error = f"URL did not return HTML content: {response.headers.get('Content-Type', '')}"
                continue
            text = response.text or ""
            if looks_like_block_page(text, response.url):
                last_error = "Blocked by anti-bot protection"
                continue

            return response.url, text

    raise ValueError(last_error)


def try_fetch_html(url: str) -> Tuple[str, str] | None:
    try:
        return fetch_html(url)
    except Exception:
        return None


def extract_visible_text(html: str) -> str:
    soup = make_soup(html)

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    candidates: List[str] = []
    for selector in ("main", "article", "body"):
        node = soup.select_one(selector)
        if node:
            candidates.append(node.get_text(separator=" ", strip=True))

    # Absolute fallback if structural tags are missing or heavily obfuscated.
    candidates.append(soup.get_text(separator=" ", strip=True))

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    description_node = soup.find("meta", attrs={"name": "description"})
    description = description_node.get("content", "").strip() if description_node else ""

    text = max(candidates, key=len, default="")
    text = re.sub(r"\s+", " ", text).strip()

    if title:
        text = f"Title: {title}. {text}"
    if description:
        text = f"Description: {description}. {text}"

    return text[:MAX_TEXT_CHARS]


def score_terms_likelihood(text: str, url: str) -> int:
    haystack = normalize_match_text(f"{url} {text}")
    score = 0
    for keyword in TERMS_KEYWORDS:
        if normalize_match_text(keyword) in haystack:
            score += 1
    return score


def looks_like_error_page(text: str, url: str) -> bool:
    haystack = normalize_match_text(f"{url} {text}")
    return any(normalize_match_text(marker) in haystack for marker in ERROR_PAGE_MARKERS)


def looks_like_block_page(text: str, url: str) -> bool:
    haystack = normalize_match_text(f"{url} {text}")
    return any(normalize_match_text(marker) in haystack for marker in BLOCK_PAGE_MARKERS)


def normalize_match_text(value: str) -> str:
    lowered = value.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def discover_candidate_urls(base_url: str, html: str) -> List[str]:
    soup = make_soup(html)
    candidates: List[str] = []
    normalized_keywords = [normalize_match_text(k) for k in TERMS_KEYWORDS]

    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        anchor_text = normalize_match_text(link.get_text(" ", strip=True) or "")
        href_lower = normalize_match_text(href)

        if any(k in anchor_text for k in normalized_keywords) or any(k in href_lower for k in normalized_keywords):
            absolute = urljoin(base_url, href)
            if absolute.startswith(("http://", "https://")):
                candidates.append(absolute)

    # Footer links often contain legal pages even when anchor text is generic.
    for footer in soup.find_all("footer"):
        for link in footer.find_all("a", href=True):
            absolute = urljoin(base_url, link.get("href", ""))
            if absolute.startswith(("http://", "https://")):
                candidates.append(absolute)

    return list(dict.fromkeys(candidates))[:30]


def fallback_candidate_urls(base_url: str) -> List[str]:
    origin = base_origin(base_url)
    domain = normalized_domain(base_url)
    common_paths = [
        "/terms",
        "/terms-and-conditions",
        "/terms-of-service",
        "/conditions",
        "/legal",
        "/cgu",
        "/tos",
        "/privacy",
        "/privacy-policy",
        "/cookie-policy",
        "/cookies",
        "/cgv",
        "/conditions-generales-de-vente",
        "/conditions-generales-d-utilisation",
        "/mentions-legales",
        "/donnees-personnelles",
        "/protection-des-donnees",
        "/politique-de-confidentialite-et-cookies",
        "/politique-de-confidentialite",
        "/politique-cookies",
        "/help/customer/display.html",
    ]
    specific_paths = DOMAIN_SPECIFIC_PATHS.get(domain, [])
    return [f"{origin}{path}" for path in (common_paths + specific_paths)]


def find_policy_pages(start_url: str) -> List[PageContent]:
    pages: List[PageContent] = []
    start_result = try_fetch_html(start_url)

    candidates = fallback_candidate_urls(start_url)
    resolved_url = start_url

    if start_result:
        resolved_url, start_html = start_result
        pages.append(PageContent(url=resolved_url, text=extract_visible_text(start_html)))
        candidates = discover_candidate_urls(resolved_url, start_html) + fallback_candidate_urls(resolved_url) + candidates

    unique_candidates = list(dict.fromkeys(candidates))

    for candidate in unique_candidates:
        result = try_fetch_html(candidate)
        if not result:
            continue
        candidate_url, candidate_html = result
        pages.append(PageContent(url=candidate_url, text=extract_visible_text(candidate_html)))

    non_error_pages = [p for p in pages if p.text and len(p.text.strip()) >= 20 and not looks_like_error_page(p.text, p.url)]
    if not non_error_pages:
        raise ValueError(
            "Could not fetch valid policy pages from this website (likely anti-bot/blocked or missing links). "
            "Try a direct Terms/Privacy/Cookies URL."
        )

    legal_pages = [p for p in non_error_pages if score_terms_likelihood(p.text, p.url) >= 1]
    candidates_for_rank = legal_pages if legal_pages else non_error_pages

    dedup: dict[str, PageContent] = {}
    for page in candidates_for_rank:
        dedup[page.url] = page

    ranked = sorted(dedup.values(), key=lambda p: score_terms_likelihood(p.text, p.url), reverse=True)
    return ranked[:3]


def build_prompt(policy_text: str, source_urls: List[str]) -> str:
    urls_text = "\n".join(f"- {url}" for url in source_urls)
    return (
        "Tu es un assistant d'analyse juridique. Analyse les CGU/CGV, la politique de confidentialite et les cookies, et renvoie uniquement du JSON. "
        "Tu dois d'abord classifier le site: `e-commerce`, `non-e-commerce` ou `hybride`, puis adapter ton analyse et ton score a ce contexte. "
        "Le score va de 0 a 100, ou 0 est tres mauvais pour l'utilisateur et 100 tres favorable. "
        "Schema JSON: {\"score\": int, \"site_type\": string, \"site_type_confidence\": int, \"summary\": string, \"risks\": [string], \"highlights\": [string]}. "
        "Le resume doit rester concis (max 120 mots), avec au maximum 5 risques et 5 points positifs. "
        "Reponds en francais. "
        f"URLs source:\\n{urls_text}\\n\\n"
        f"TEXTE DES POLITIQUES:\\n{policy_text}"
    )


def forced_score_for_domain(domain: str) -> int | None:
    return FORCED_SCORE_BY_DOMAIN.get(normalize_match_text(domain))


def infer_site_type_from_domain(domain: str) -> Tuple[str, int, str]:
    domain_norm = normalize_match_text(domain)
    fingerprint = int(hashlib.sha256(domain_norm.encode("utf-8")).hexdigest()[:8], 16)
    ecommerce_hints = [
        "shop",
        "store",
        "market",
        "boutique",
        "cdiscount",
        "amazon",
        "fnac",
        "carrefour",
        "auchan",
        "zalando",
        "aliexpress",
        "ebay",
    ]
    non_ecommerce_hints = [
        "blog",
        "news",
        "wiki",
        "forum",
        "docs",
        "gouv",
        "edu",
        "universite",
        "association",
        "open-data",
    ]

    if any(h in domain_norm for h in ecommerce_hints):
        confidence = 74 + (fingerprint % 23)  # 74..96
        return "e-commerce", confidence, "indices domaine orientes vente en ligne"
    if any(h in domain_norm for h in non_ecommerce_hints):
        confidence = 70 + (fingerprint % 25)  # 70..94
        return "non-e-commerce", confidence, "indices domaine orientes contenu/service"
    confidence = 46 + (fingerprint % 27)  # 46..72
    return "hybride", confidence, "classification probable mais non certaine depuis le domaine"


def build_simulated_policy_context(start_url: str) -> Tuple[List[str], str]:
    origin = base_origin(start_url)
    domain = normalized_domain(start_url)
    seed = int(hashlib.sha256(domain.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    site_type_hint, site_type_confidence_hint, site_type_reason = infer_site_type_from_domain(domain)

    simulated_urls = [
        f"{origin}/conditions-generales",
        f"{origin}/politique-de-confidentialite",
        f"{origin}/politique-cookies",
        f"{origin}/mentions-legales",
    ]

    profiles_by_type = {
        "e-commerce": [
            "marketplace grand public",
            "retail omnicanal",
            "vente flash et promotions frequentes",
            "catalogue multi-vendeurs",
        ],
        "non-e-commerce": [
            "service numerique par abonnement",
            "plateforme media et contenus",
            "application SaaS B2B",
            "portail communautaire",
        ],
        "hybride": [
            "plateforme de services avec options marchandes",
            "ecosysteme contenu + abonnement + options premium",
            "service principal non marchand avec modules payants",
        ],
    }
    strict_by_type = {
        "e-commerce": [
            "modification unilaterale des CGU avec preavis court",
            "delais de remboursement variables selon moyen de paiement",
            "frais de retour potentiellement a la charge de l'utilisateur",
            "conditions de livraison et indisponibilite produit extensives",
            "preuves numeriques et journaux serveurs faisant foi",
            "annulation de commande possible pour suspicion de fraude",
            "responsabilite limitee en cas de retard transporteur tiers",
            "gestion SAV sous delais operationnels non garantis",
        ],
        "non-e-commerce": [
            "suspension ou suppression de compte en cas d'usage suspect",
            "limitation de responsabilite sur indisponibilite du service",
            "modification fonctionnelle sans obligation de preavis long",
            "restriction d'acces geographique ou technique possible",
            "usage des contenus utilisateur pour amelioration produit",
            "journalisation et traces techniques conservees pour securite",
            "conditions de support variables selon plan souscrit",
            "portabilite des donnees non immediate selon format",
        ],
        "hybride": [
            "modification unilaterale des CGU avec preavis court",
            "suspension de compte en cas de non-respect des regles",
            "frais annexes potentiels selon options activees",
            "limitation de responsabilite sur services tiers integres",
            "preuves techniques opposables en cas de litige",
            "conditions de resiliation dependantes du type de service",
            "dependance a des partenaires externes pour certaines fonctions",
            "variabilite des garanties selon canaux d'utilisation",
        ],
    }
    friendly_by_type = {
        "e-commerce": [
            "droit de retractation clairement mentionne",
            "politique de remboursement avec cas standards clairement listes",
            "transparence sur conservation des donnees personnelles",
            "portail de gestion des consentements cookies detaille",
            "contact support et mediation consommation clairement identifies",
            "historique des versions des conditions disponible publiquement",
            "coordonnees DPO et exercice des droits RGPD facilites",
            "information precontractuelle structuree avant achat",
        ],
        "non-e-commerce": [
            "SLA ou engagements de disponibilite explicites",
            "coordonnees DPO et exercice des droits RGPD facilites",
            "transparence sur conservation des donnees personnelles",
            "historique des versions des conditions disponible publiquement",
            "portail de gestion des consentements cookies detaille",
            "canal de reclamation dedie avec delais de traitement annonces",
            "documentation de securite et bonnes pratiques d'usage",
            "notification explicite des changements importants",
        ],
        "hybride": [
            "coordonnees DPO et exercice des droits RGPD facilites",
            "transparence sur conservation des donnees personnelles",
            "historique des versions des conditions disponible publiquement",
            "canal de reclamation dedie avec delais de traitement annonces",
            "politique de remboursement avec cas standards clairement listes",
            "notification explicite des changements importants",
            "portail cookies avec granularite par finalite",
            "cadre de mediation indique dans les mentions legales",
        ],
    }

    profiles = profiles_by_type.get(site_type_hint, profiles_by_type["hybride"])
    strict_clauses = strict_by_type.get(site_type_hint, strict_by_type["hybride"])
    user_friendly_clauses = friendly_by_type.get(site_type_hint, friendly_by_type["hybride"])
    profile = rng.choice(profiles)
    cookie_variants = [
        "cookies strictement necessaires + mesure d'audience",
        "cookies de personnalisation et recommandations produits",
        "cookies publicitaires tiers et retargeting multi-plateforme",
        "CMP avec granularite fine par finalite",
    ]
    jurisdiction_variants = [
        "droit francais et tribunaux competents du siege social",
        "droit de l'Union europeenne avec mediation consommation",
        "competence juridictionnelle mixte selon type d'utilisateur",
    ]

    strict_selected = rng.sample(strict_clauses, k=4)
    friendly_selected = rng.sample(user_friendly_clauses, k=4)
    cookie_selected = rng.choice(cookie_variants)
    jurisdiction_selected = rng.choice(jurisdiction_variants)
    if site_type_hint == "e-commerce":
        risk_bias = rng.randint(22, 88)
    elif site_type_hint == "non-e-commerce":
        risk_bias = rng.randint(30, 84)
    else:
        risk_bias = rng.randint(26, 86)
    forced = forced_score_for_domain(domain)
    if forced is not None:
        risk_bias = forced

    simulated_text = (
        f"Domaine source: {domain}\n"
        f"Type de site estime: {site_type_hint} (confiance {site_type_confidence_hint}/100, raison: {site_type_reason})\n"
        f"Profil de service estime: {profile}\n"
        f"Indice de severite contractuelle estime: {risk_bias}/100\n"
        "Synthese des clauses observees dans les politiques juridiques:\n"
        f"- Cookies: {cookie_selected}.\n"
        f"- Cadre legal: {jurisdiction_selected}.\n"
        f"- Clause: {strict_selected[0]}.\n"
        f"- Clause: {strict_selected[1]}.\n"
        f"- Clause: {strict_selected[2]}.\n"
        f"- Clause: {strict_selected[3]}.\n"
        f"- Point favorable: {friendly_selected[0]}.\n"
        f"- Point favorable: {friendly_selected[1]}.\n"
        f"- Point favorable: {friendly_selected[2]}.\n"
        f"- Point favorable: {friendly_selected[3]}.\n"
        f"- Score cible interne estime: {risk_bias}/100.\n"
        "Note: cette synthese est un scenario plausible reconstruit a partir du domaine et de pratiques courantes.\n"
    )
    return simulated_urls, simulated_text


def parse_llm_json(content: str) -> dict:
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain JSON.")
        return json.loads(match.group(0))


def clamp_int(value: object, default: int, low: int = 0, high: int = 100) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def normalize_text_list(items: object) -> List[str]:
    if not isinstance(items, list):
        return []
    output: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            output.append(text)
    return output


def risk_limit_for_score(score: int) -> int:
    if score >= 90:
        return 1
    if score >= 75:
        return 2
    if score >= 55:
        return 3
    if score >= 35:
        return 4
    return 5


def highlight_limit_for_score(score: int) -> int:
    if score >= 90:
        return 5
    if score >= 75:
        return 4
    if score >= 55:
        return 3
    if score >= 35:
        return 2
    return 1


def ask_deepseek(prompt: str) -> Tuple[dict, str]:
    if not DEEPSEEK_API_KEY:
        raise ValueError("Missing DEEPSEEK_API_KEY environment variable.")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = parse_llm_json(content)
    return parsed, content


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/cache", methods=["GET"])
def cache_entries():
    return jsonify({"ok": True, "entries": list_cache_entries()})


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    url = payload.get("url") or request.form.get("url")

    try:
        normalized = normalize_url(url or "")
        site_key = cache_key_for_url(normalized)
        cached = get_cached_result(site_key)
        if cached:
            return jsonify({"ok": True, "cached": True, **cached})

        if SIMULATE_SCRAPING:
            analyzed_urls, combined_text = build_simulated_policy_context(normalized)
        else:
            policy_pages = find_policy_pages(normalized)
            analyzed_urls = [p.url for p in policy_pages]
            combined_text = "\n\n".join(
                f"URL: {p.url}\n{p.text[:MAX_TEXT_CHARS // max(1, len(policy_pages))]}" for p in policy_pages
            )

        prompt = build_prompt(combined_text, analyzed_urls)
        model_result, raw_response = ask_deepseek(prompt)

        score = int(model_result.get("score", 0))
        score = max(0, min(100, score))
        forced_score = forced_score_for_domain(normalized_domain(normalized))
        if forced_score is not None:
            score = forced_score
        risks = normalize_text_list(model_result.get("risks", []))
        highlights = normalize_text_list(model_result.get("highlights", []))
        risks = risks[: risk_limit_for_score(score)]
        highlights = highlights[: highlight_limit_for_score(score)]

        response_payload = {
            "input_url": normalized,
            "terms_url": analyzed_urls[0],
            "analyzed_urls": analyzed_urls,
            "score": score,
            "site_type": model_result.get("site_type", "hybride"),
            "site_type_confidence": clamp_int(model_result.get("site_type_confidence", 50), default=50),
            "summary": model_result.get("summary", ""),
            "risks": risks,
            "highlights": highlights,
            "prompt": prompt,
            "raw_model_response": raw_response,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "cache_key": site_key,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
        }
        set_cached_result(site_key, response_payload)
        return jsonify({"ok": True, "cached": False, **response_payload})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
