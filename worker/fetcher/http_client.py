import httpx
from worker.fetcher.security import validate_url
from typing import Optional, Tuple

class SafeHTTPClient:
    def __init__(self, timeout: int = 10, max_redirects: int = 5, max_size: int = 5 * 1024 * 1024):
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.max_size = max_size

    def fetch_html(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns (html_content, error_message)"""
        if not validate_url(url):
            return None, "SSRF or invalid URL detected"

        try:
            # We don't use follow_redirects=True directly because we want to check sizes and SSRF at each step
            client = httpx.Client(timeout=self.timeout)
            current_url = url
            
            for _ in range(self.max_redirects + 1):
                # Stream the response so we can check Content-Length and actual size
                with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        current_url = response.headers.get("Location")
                        if not current_url:
                            return None, "Invalid redirect"
                        # Make absolute if relative
                        current_url = httpx.URL(url).join(current_url)
                        if not validate_url(str(current_url)):
                            return None, "Redirected to unsafe URL (SSRF)"
                        continue
                        
                    # Target reached
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" not in content_type:
                        return None, f"Invalid Content-Type: {content_type}"
                        
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > self.max_size:
                        return None, "Content too large"
                        
                    content = b""
                    for chunk in response.iter_bytes():
                        content += chunk
                        if len(content) > self.max_size:
                            return None, "Content exceeded max size during read"
                            
                    return content.decode('utf-8', errors='ignore'), None
                    
            return None, "Too many redirects"
        except httpx.TimeoutException:
            return None, "Request timed out"
        except Exception as e:
            return None, f"Request failed: {str(e)}"
