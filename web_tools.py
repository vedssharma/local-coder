"""Bounded public-web retrieval using the environment's proxy and TLS trust."""
from html.parser import HTMLParser
import ipaddress
import json
import os
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener, getproxies, proxy_bypass


from contextlib import nullcontext
from execution_context import CURRENT_CONTEXT

MAX_BYTES = 1_000_000


def public_url(url):
    if not isinstance(url, str) or len(url) > 8192 or any(c.isspace() for c in url):
        raise ValueError('Use a valid HTTP(S) URL without whitespace')
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Only public HTTP(S) URLs without embedded credentials are supported')
    if parsed.port not in (None, 80, 443):
        raise ValueError('Only standard HTTP(S) ports are supported')
    host = parsed.hostname.rstrip('.').lower()
    if host == 'localhost' or host.endswith(('.localhost', '.local')):
        raise ValueError('Local, private, and non-public network destinations are unavailable')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError('Local, private, and non-public network destinations are unavailable')
    # A managed HTTP proxy resolves destination names itself. Local DNS may be
    # unavailable; retain that proxy rather than trying to bypass its policy.
    # For proxied hostnames the proxy's destination policy enforces DNS access.
    if getproxies().get(parsed.scheme) and not proxy_bypass(parsed.hostname):
        return url
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80),
                                   type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('Local, private, and non-public network destinations are unavailable')
    return url


class PublicRedirect(HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        # Search-provider credentials must never travel to another origin.
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            req.remove_header('X-subscription-token')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def retrieve(url, timeout_seconds=20, headers=None):
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 30:
        raise ValueError('timeout_seconds must be between 1 and 30')
    context = CURRENT_CONTEXT.get()
    if context:
        context.check()
    public_url(url)
    request = Request(url, headers={'User-Agent': 'local-coder/1.0', 'Accept-Encoding': 'identity', **(headers or {})})
    try:
        timeout = context.timeout(timeout_seconds) if context else timeout_seconds
        with build_opener(PublicRedirect()).open(request, timeout=timeout) as response, (context.response_guard(response) if context else nullcontext()):
            data = response.read(MAX_BYTES + 1)
            encoding = response.headers.get_content_charset() or 'utf-8'
            try:
                body = data[:MAX_BYTES].decode(encoding, errors='replace')
            except LookupError:
                body = data[:MAX_BYTES].decode('utf-8', errors='replace')
            return {'url': response.geturl(), 'status': response.status,
                    'content_type': response.headers.get_content_type(),
                    'body': body,
                    'truncated': len(data) > MAX_BYTES}
    except HTTPError as exc:
        raise ValueError(f'Web request returned HTTP {exc.code}; check destination access and provider availability') from None
    except URLError as exc:
        raise ValueError(f'Web request failed: {exc.reason}') from None


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.in_title = False
        self.text = []
        self.title = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript', 'template'):
            self.hidden += 1
        if tag == 'title':
            self.in_title = True
        if tag in ('p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'pre', 'tr') and not self.hidden:
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'template'):
            self.hidden = max(0, self.hidden - 1)
        if tag == 'title':
            self.in_title = False
        if tag in ('p', 'div', 'li', 'h1', 'h2', 'h3', 'pre', 'tr') and not self.hidden:
            self.text.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            (self.title if self.in_title else self.text).append(data)


def fetch(url, max_chars=12000, timeout_seconds=20):
    if type(max_chars) is not int or not 100 <= max_chars <= 50000:
        raise ValueError('max_chars must be between 100 and 50000')
    result = retrieve(url, timeout_seconds)
    body = result.pop('body')
    title = ''
    if result['content_type'] in ('text/html', 'application/xhtml+xml'):
        parser = PageParser()
        parser.feed(body)
        title = ' '.join(''.join(parser.title).split())
        body = '\n'.join(line.strip() for line in ''.join(parser.text).splitlines() if line.strip())
    elif not (result['content_type'].startswith('text/') or result['content_type'] in ('application/json', 'application/xml')):
        raise ValueError('Unsupported content type; web_fetch supports HTML and text, not binary files')
    result.update(title=title[:500], text=body[:max_chars], truncated=result['truncated'] or len(body) > max_chars,
                  untrusted_content=True)
    return result


class SearchParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.capture = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get('class', '').split()
        if tag == 'a' and 'result__a' in classes:
            href = urljoin('https://html.duckduckgo.com', attrs.get('href', ''))
            destination = parse_qs(urlsplit(href).query).get('uddg', [href])[0]
            if urlsplit(destination).scheme in ('http', 'https'):
                self.results.append({'title': '', 'url': destination, 'snippet': ''})
                self.capture = ('a', 'title')
        elif 'result__snippet' in classes and self.results:
            self.capture = (tag, 'snippet')

    def handle_endtag(self, tag):
        if self.capture and self.capture[0] == tag:
            self.capture = None

    def handle_data(self, data):
        if self.capture and self.results:
            self.results[-1][self.capture[1]] += data


def search(query, max_results=5, timeout_seconds=20):
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise ValueError('query must contain 1 to 1000 characters')
    if type(max_results) is not int or not 1 <= max_results <= 10:
        raise ValueError('max_results must be between 1 and 10')
    key = os.environ.get('BRAVE_SEARCH_API_KEY')
    if key:
        response = retrieve('https://api.search.brave.com/res/v1/web/search?' + urlencode({'q': query, 'count': max_results}),
                            timeout_seconds, {'X-Subscription-Token': key, 'Accept': 'application/json'})
        if response['truncated']:
            raise ValueError('Search provider response exceeded the size limit')
        items = json.loads(response['body']).get('web', {}).get('results', [])
        results = [{'title': i.get('title', ''), 'url': i['url'], 'snippet': i.get('description', '')} for i in items]
        provider = 'brave'
    else:
        response = retrieve('https://html.duckduckgo.com/html/?' + urlencode({'q': query}), timeout_seconds)
        parser = SearchParser()
        parser.feed(response['body'])
        results = parser.results
        if not results and not re.search(r'no results|no more results', response['body'], re.I):
            raise ValueError('Search provider returned no recognizable results (possibly a bot challenge); '
                             'retry or configure BRAVE_SEARCH_API_KEY')
        provider = 'duckduckgo'
    for item in results:
        item['title'] = ' '.join(item['title'].split())[:500]
        item['snippet'] = ' '.join(item['snippet'].split())[:2000]
    return {'query': query, 'provider': provider, 'results': results[:max_results],
            'truncated': response['truncated'], 'untrusted_content': True}
