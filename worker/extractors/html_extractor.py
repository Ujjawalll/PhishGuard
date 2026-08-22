from bs4 import BeautifulSoup
import re
import urllib.parse

def extract_html(html: str, base_url: str) -> dict:
    result = {
        "form_count": 0,
        "password_input_count": 0,
        "input_count": 0,
        "script_count": 0,
        "iframe_count": 0,
        "external_link_ratio": 0.0,
        "has_redirect": 0,
        "title_length": 0,
        "suspicious_text_count": 0
    }
    
    if not html:
        return result
        
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        result["form_count"] = len(soup.find_all('form'))
        result["input_count"] = len(soup.find_all('input'))
        result["password_input_count"] = len(soup.find_all('input', type='password'))
        result["script_count"] = len(soup.find_all('script'))
        result["iframe_count"] = len(soup.find_all('iframe'))
        
        if soup.title and soup.title.string:
            result["title_length"] = len(soup.title.string)
            
        # Redirect check (meta refresh)
        meta_refresh = soup.find('meta', attrs={'http-equiv': re.compile(r'^refresh$', re.I)})
        if meta_refresh:
            result["has_redirect"] = 1
            
        # Links
        links = soup.find_all('a', href=True)
        total_links = len(links)
        if total_links > 0:
            parsed_base = urllib.parse.urlparse(base_url)
            external = 0
            for link in links:
                href = link['href']
                if href.startswith('http'):
                    parsed_href = urllib.parse.urlparse(href)
                    if parsed_href.netloc != parsed_base.netloc:
                        external += 1
            result["external_link_ratio"] = round(external / total_links, 4)
            
        # Suspicious text
        text = soup.get_text().lower()
        suspicious_words = ['verify', 'suspended', 'confirm', 'account', 'login', 'secure']
        result["suspicious_text_count"] = sum(text.count(w) for w in suspicious_words)
        
    except Exception:
        pass
        
    return result
