import re
from urllib.parse import urlparse

URL_REGEX = re.compile(r"(?:https?://|www\.)[\w\-\.\@:%_\+~#?&//=]+", re.IGNORECASE)
SUSPICIOUS_KEYWORDS = [
    "verify", "login", "signin", "account", "update", "password",
    "secure", "billing", "confirm", "urgent", "alert", "limited",
    "unauthorized", "suspend", "prize", "reward",
]
ACTION_PHRASES = [
    "click here", "verify your", "update your", "confirm your", "reset your",
    "sign in to", "login to", "provide your", "submit your",
]


def find_urls(text: str):
    return URL_REGEX.findall(text)


def is_ip_address_host(host: str) -> bool:
    return bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", host))


def parse_url_host(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.hostname or ""
    except Exception:
        return ""


def get_rule_based_warnings(text: str):
    warnings = []
    normalized = text.lower()

    if any(phrase in normalized for phrase in ACTION_PHRASES):
        warnings.append("Contains urgent or action-oriented phrasing commonly used in phishing attacks.")

    if any(keyword in normalized for keyword in SUSPICIOUS_KEYWORDS):
        warnings.append("Contains phishing-related keywords such as verify, account, login, or urgent.")

    urls = find_urls(text)
    for url in urls:
        host = parse_url_host(url)
        if host:
            if is_ip_address_host(host):
                warnings.append("Includes a URL that uses a raw IP address instead of a domain.")
            if host.count("-") >= 2:
                warnings.append("URL host contains multiple hyphens, which is often suspicious.")
            if any(token in host for token in ["secure", "login", "verify", "account", "update", "confirm"]):
                warnings.append("URL contains suspicious keywords in the hostname.")

        if "/secure" in url.lower() or "/login" in url.lower() or "/verify" in url.lower():
            warnings.append("URL path contains login or verification keywords.")

    if not urls and any(keyword in normalized for keyword in ["your account", "your password", "your billing"]):
        warnings.append("Text references account or billing details without a clear website URL.")

    if not warnings and len(urls) == 0 and len(normalized.split()) > 50:
        warnings.append("Long email-like text with no clear source URL should be reviewed carefully.")

    return warnings


def summarize_confidence(score: float) -> str:
    if score >= 0.85:
        return "High confidence"
    if score >= 0.65:
        return "Medium confidence"
    return "Low confidence"
