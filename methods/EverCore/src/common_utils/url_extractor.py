"""
URL content extraction tool
Used to extract metadata such as title, description, and images from web pages
"""

import re
import aiohttp
from typing import Optional, Dict, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Tag

from core.observation.logger import get_logger

logger = get_logger(__name__)

# Request configuration
DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; MemsysBot/1.0; +https://memsys.ai/bot)"


class URLExtractor:
    """URL content extractor"""

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        """
        Initialize URL extractor

        Args:
            timeout: Request timeout duration (seconds)
            max_content_length: Maximum content length
            user_agent: User agent string
        """
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.user_agent = user_agent

    async def extract_metadata(
        self, url: str, need_redirect: bool = True
    ) -> Dict[str, Any]:
        """
        Extract metadata information from URL

        Args:
            url: URL to extract
            need_redirect: Whether to follow redirects to obtain the final URL

        Returns:
            Dict[str, Any]: Extracted metadata information
        """
        try:
            # Get final URL (if redirection is needed)
            final_url = url
            if need_redirect:
                final_url = await self._get_final_url(url)

            # Get webpage content
            html_content = await self._fetch_html_content(final_url)
            if not html_content:
                return self._create_empty_metadata(url, final_url)

            # Parse HTML and extract metadata
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata = self._extract_metadata_from_soup(soup, final_url)
            metadata['original_url'] = url
            metadata['final_url'] = final_url

            return metadata

        except Exception as e:
            logger.error("Failed to extract URL metadata: %s, error: %s", url, str(e))
            return self._create_error_metadata(url, str(e))

    async def _get_final_url(self, url: str) -> str:
        """
        Get the final URL after redirection

        Args:
            url: Original URL

        Returns:
            str: Final URL
        """
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {'User-Agent': self.user_agent}

            # Create SSL context, skip certificate verification (relatively safe for content extraction)
            import ssl

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers, connector=connector
            ) as session:
                # Send only HEAD request to get final URL, without downloading content
                async with session.head(url, allow_redirects=True) as response:
                    return str(response.url)

        except Exception as e:
            logger.warning("Failed to get final URL: %s, error: %s", url, str(e))
            return url

    async def _fetch_html_content(self, url: str) -> Optional[str]:
        """
        Get HTML content

        Args:
            url: URL to retrieve

        Returns:
            Optional[str]: HTML content, returns None on failure
        """
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            # Create SSL context, skip certificate verification (relatively safe for content extraction)
            import ssl

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers, connector=connector
            ) as session:
                async with session.get(url) as response:
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' not in content_type:
                        logger.warning(
                            "Non-HTML content: %s, content-type: %s", url, content_type
                        )
                        return None

                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_content_length:
                        logger.warning(
                            "Content too large: %s, size: %s", url, content_length
                        )
                        return None

                    # Read content
                    content = await response.text()
                    if len(content) > self.max_content_length:
                        logger.warning(
                            "Content too large: %s, size: %d", url, len(content)
                        )
                        return None

                    return content

        except Exception as e:
            logger.error("Failed to get HTML content: %s, error: %s", url, str(e))
            return None

    def _extract_metadata_from_soup(
        self, soup: BeautifulSoup, url: str
    ) -> Dict[str, Any]:
        """
        Extract metadata from BeautifulSoup object

        Args:
            soup: BeautifulSoup object
            url: Page URL

        Returns:
            Dict[str, Any]: Extracted metadata
        """
        metadata = {
            'title': None,
            'description': None,
            'image': None,
            'site_name': None,
            'url': url,
            'type': None,
            'favicon': None,
            'og_tags': {},
            'twitter_tags': {},
            'meta_tags': {},
        }

        try:
            # Extract Open Graph tags
            og_tags = self._extract_og_tags(soup)
            metadata['og_tags'] = og_tags

            # Extract Twitter Card tags
            twitter_tags = self._extract_twitter_tags(soup)
            metadata['twitter_tags'] = twitter_tags

            # Extract basic meta tags
            meta_tags = self._extract_meta_tags(soup)
            metadata['meta_tags'] = meta_tags

            # Prioritize Open Graph information, but skip values containing template variables
            metadata['title'] = (
                self._get_safe_value(og_tags.get('title'))
                or self._get_safe_value(twitter_tags.get('title'))
                or self._get_safe_value(self._extract_title(soup))
                or self._get_safe_value(meta_tags.get('title'))
            )

            metadata['description'] = (
                self._get_safe_value(og_tags.get('description'))
                or self._get_safe_value(twitter_tags.get('description'))
                or self._get_safe_value(meta_tags.get('description'))
            )

            metadata['image'] = self._get_safe_value(
                og_tags.get('image')
            ) or self._get_safe_value(twitter_tags.get('image'))

            metadata['site_name'] = self._get_safe_value(og_tags.get('site_name'))
            metadata['type'] = self._get_safe_value(og_tags.get('type'))
            metadata['favicon'] = self._extract_favicon(soup, url)

            # Clean and validate data
            metadata = self._clean_metadata(metadata)

        except Exception as e:
            logger.error("Failed to parse metadata: %s, error: %s", url, str(e))

        return metadata

    def _extract_og_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract Open Graph tags"""
        og_tags = {}

        for tag in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
            if tag.get('content'):
                property_name = tag['property'][3:]  # Remove 'og:' prefix
                og_tags[property_name] = tag['content'].strip()

        return og_tags

    def _extract_twitter_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract Twitter Card tags"""
        twitter_tags = {}

        for tag in soup.find_all(
            'meta', attrs={'name': lambda x: x and x.startswith('twitter:')}
        ):
            if tag.get('content'):
                name = tag['name'][8:]  # Remove 'twitter:' prefix
                twitter_tags[name] = tag['content'].strip()

        return twitter_tags

    def _extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract basic meta tags"""
        meta_tags = {}

        # Extract title
        title_tag = soup.find('meta', attrs={'name': 'title'})
        if title_tag and title_tag.get('content'):
            meta_tags['title'] = title_tag['content'].strip()

        # Extract description
        description_tag = soup.find('meta', attrs={'name': 'description'})
        if description_tag and description_tag.get('content'):
            meta_tags['description'] = description_tag['content'].strip()

        # Extract keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag and keywords_tag.get('content'):
            meta_tags['keywords'] = keywords_tag['content'].strip()

        # Extract author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag and author_tag.get('content'):
            meta_tags['author'] = author_tag['content'].strip()

        return meta_tags

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        return None

    def _extract_first_image(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Extract the first meaningful image"""
        # Find img tags
        img_tags = soup.find_all('img', src=True)

        for img in img_tags:
            src = img['src'].strip()
            if not src:
                continue

            # Convert to absolute URL
            absolute_url = urljoin(base_url, src)

            # Simple filter: skip obvious decorative images
            if self._is_meaningful_image(img, src):
                return absolute_url

        return None

    def _is_meaningful_image(self, img_tag: Tag, src: str) -> bool:
        """Determine if the image is meaningful (non-decorative)"""
        # Skip obvious decorative images
        skip_patterns = [
            'icon',
            'logo',
            'avatar',
            'button',
            'pixel',
            'spacer',
            'blank',
            'transparent',
            '1x1',
            'tracking',
        ]

        src_lower = src.lower()
        if any(pattern in src_lower for pattern in skip_patterns):
            return False

        # Check image size attributes
        width = img_tag.get('width')
        height = img_tag.get('height')

        if width and height:
            try:
                w, h = int(width), int(height)
                # Skip very small images
                if w < 100 or h < 100:
                    return False
                # Skip obvious decorative sizes
                if w == 1 or h == 1:
                    return False
            except (ValueError, TypeError):
                pass

        return True

    def _extract_favicon(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Extract website icon"""
        # Find icon in link tags
        icon_links = soup.find_all('link', rel=lambda x: x and 'icon' in x.lower())

        for link in icon_links:
            href = link.get('href')
            if href:
                return urljoin(base_url, href.strip())

        # Default favicon path
        parsed_url = urlparse(base_url)
        default_favicon = f"{parsed_url.scheme}://{parsed_url.netloc}/favicon.ico"
        return default_favicon

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate metadata"""
        # Clean string fields
        string_fields = [
            'title',
            'description',
            'image',
            'site_name',
            'type',
            'favicon',
            'url',
        ]
        for field in string_fields:
            if metadata.get(field):
                # Clean extra whitespace
                cleaned_value = re.sub(r'\s+', ' ', str(metadata[field])).strip()
                metadata[field] = cleaned_value

                # Limit length
                if field == 'title' and len(metadata[field]) > 200:
                    metadata[field] = metadata[field][:200] + '...'
                elif field == 'description' and len(metadata[field]) > 500:
                    metadata[field] = metadata[field][:500] + '...'

        # Validate URL format
        url_fields = ['image', 'favicon', 'url']
        for field in url_fields:
            if metadata.get(field) and not self._is_valid_url(metadata[field]):
                metadata[field] = None

        return metadata

    def _contains_template_variables(self, text: str) -> bool:
        """
        Check if text contains template variables

        Check the following template variable formats:
        - ${variable}
        - {{variable}}
        - {variable}
        - #{variable}
        - @{variable}

        Args:
            text: Text to check

        Returns:
            bool: Returns True if contains template variables, otherwise False
        """
        if not text or not isinstance(text, str):
            return False

        # Define regular expression patterns for template variables
        template_patterns = [
            r'\$\{[^}]+\}',  # ${variable}
            r'\{\{[^}]+\}\}',  # {{variable}}
            r'#\{[^}]+\}',  # #{variable}
            r'@\{[^}]+\}',  # @{variable}
            # {variable} - Only match variable names containing letters, digits, dots, underscores
            r'\{[a-zA-Z_][a-zA-Z0-9_.]*\}',
        ]

        # Check each pattern
        for pattern in template_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _get_safe_value(self, value: str) -> Optional[str]:
        """
        Get a safe value, return None if it contains template variables

        Args:
            value: Value to check

        Returns:
            Optional[str]: Returns original value if valid and does not contain template variables, otherwise returns None
        """
        if not value or not isinstance(value, str):
            return None

        # Clean whitespace
        cleaned_value = value.strip()
        if not cleaned_value:
            return None

        # Check if contains template variables
        if self._contains_template_variables(cleaned_value):
            return None

        return cleaned_value

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _create_empty_metadata(
        self, original_url: str, final_url: str
    ) -> Dict[str, Any]:
        """Create empty metadata"""
        return {
            'title': None,
            'description': None,
            'image': None,
            'site_name': None,
            'url': final_url,
            'original_url': original_url,
            'final_url': final_url,
            'type': None,
            'favicon': None,
            'og_tags': {},
            'twitter_tags': {},
            'meta_tags': {},
            'error': None,
        }

    def _create_error_metadata(self, url: str, error: str) -> Dict[str, Any]:
        """Create error metadata"""
        return {
            'title': None,
            'description': None,
            'image': None,
            'site_name': None,
            'url': url,
            'original_url': url,
            'final_url': url,
            'type': None,
            'favicon': None,
            'og_tags': {},
            'twitter_tags': {},
            'meta_tags': {},
            'error': error,
        }


# Global instance
_url_extractor = URLExtractor()


async def extract_url_metadata(url: str, need_redirect: bool = True) -> Dict[str, Any]:
    """
    Convenience function to extract URL metadata

    Args:
        url: URL to extract
        need_redirect: Whether to follow redirects

    Returns:
        Dict[str, Any]: Metadata information
    """
    return await _url_extractor.extract_metadata(url, need_redirect)
