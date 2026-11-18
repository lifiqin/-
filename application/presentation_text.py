"""Text-only, bounded projection for the unprivileged JavaScript view."""
import re

_SECRET = re.compile(r'sk-[A-Za-z0-9]{12,}', re.IGNORECASE)
_URL = re.compile(r'https?://\S+', re.IGNORECASE)


def clean_text(value, maximum=1200):
    """Do not publish keys, service URLs, control or bidi-override characters."""
    if not isinstance(value, str):
        return ''
    value = _SECRET.sub('[已隐藏]', value)
    value = _URL.sub('[地址已隐藏]', value)
    value = ''.join(c for c in value if c in '\n\t' or (
        ord(c) >= 32 and not 0x7f <= ord(c) <= 0x9f
        and not 0x202a <= ord(c) <= 0x202e and not 0x2066 <= ord(c) <= 0x2069
        and ord(c) not in (0xfffe, 0xffff)))
    if len(value) > maximum:
        value = value[:maximum] + '…（内容已截断）'
    return value.encode('utf-8', 'replace').decode('utf-8')
