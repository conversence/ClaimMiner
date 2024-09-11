"""
Extract metadata from a HTML file.
Translated from `Hypothesis client code<https://github.com/hypothesis/client/blob/main/src/annotator/integrations/html-metadata.ts>`_
"""

"""
This translation to Python is Copyright 2024 Conversence and SocietyLibrary,
under the same terms as the original copyright notice below:

Copyright 2012 Aron Carroll, Rufus Pollock, and Nick Stenning.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from contextlib import suppress
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Dict, List
from urllib.parse import urlparse, ParseResult, quote_plus, unquote
import re

from lxml.html import HtmlElement

from .uri import normalize


@dataclass
class Link:
    href: str
    rel: Optional[str] = None
    type: Optional[str] = None


# Extension of the `Metadata` type with non-optional fields for `dc`, `eprints` etc.
@dataclass
class HTMLDocumentMetadata:
    title: str
    link: List[Link]
    dc: Dict[str, List[str]]
    eprints: Dict[str, List[str]]
    facebook: Dict[str, List[str]]
    highwire: Dict[str, List[str]]
    prism: Dict[str, List[str]]
    twitter: Dict[str, List[str]]
    favicon: Optional[str] = None
    documentFingerprint: Optional[str] = None


# HTMLMetadata reads metadata/links from the current HTML document.
class HTMLMetadata:
    document: HtmlElement
    location: str

    def __init__(self, document: HtmlElement, location: str):
        self.document = document
        self.location = location

    # Returns the primary URI for the document being annotated
    def uri(self) -> str:
        uri = unquote(self._getDocumentHref())

        # Use the `link[rel=canonical]` element's href as the URL if present.
        links = self._getLinks()
        for link in links:
            if link.rel == "canonical":
                uri = link.href
        return uri

    # Return metadata for the current page.
    def getDocumentMetadata(self) -> HTMLDocumentMetadata:
        title_el = self.document.find("/head/title")
        title = title_el.text if title_el is not None else ""
        metadata = HTMLDocumentMetadata(
            title=title,
            link=[],
            dc=self._getMetaTags("name", "dc."),
            eprints=self._getMetaTags("name", "eprints."),
            facebook=self._getMetaTags("property", "og:"),
            highwire=self._getMetaTags("name", "citation_"),
            prism=self._getMetaTags("name", "prism."),
            twitter=self._getMetaTags("name", "twitter:"),
        )

        if favicon := self._getFavicon():
            metadata.favicon = favicon

        metadata.title = self._getTitle(metadata)
        metadata.link = self._getLinks(metadata)

        for dcLink in filter(
            lambda link: link.href.startswith("urn:x-dc"), metadata.link
        ):
            metadata.documentFingerprint = dcLink.href
            break

        return metadata

    # Return an array of all the `content` values of `<meta>` tags on the page
    # where the value of the attribute begins with `<prefix>`.
    #
    # @param prefix - it is interpreted as a regex
    def _getMetaTags(self, attribute: str, prefix: str) -> Dict[str, List[str]]:
        tags: Dict[str, List[str]] = defaultdict(list)
        for meta in self.document.findall(".//meta"):
            name = meta.attrib.get(attribute, None)
            content = meta.attrib.get("content", None)
            if name and content:
                if match := re.match(rf"^{prefix}(.+)$", name, re.I):
                    key = match[1].lower()
                    tags[key].append(content)
        return tags

    def _getTitle(self, metadata: HTMLDocumentMetadata) -> str:
        if "title" in metadata.highwire:
            return metadata.highwire["title"][0]
        elif "title" in metadata.eprints:
            return metadata.eprints["title"][0]
        elif "title" in metadata.prism:
            return metadata.prism["title"][0]
        elif "title" in metadata.facebook:
            return metadata.facebook["title"][0]
        elif "title" in metadata.twitter:
            return metadata.twitter["title"][0]
        elif "title" in metadata.dc:
            return metadata.dc["title"][0]
        else:
            text_el = self.document.find("/head/title")
            if text_el is not None:
                return text_el.text
            h1 = self.document.find("/body//h1")
            if h1 is not None:
                return h1.text
        return ""

    # Get document URIs from `<link>` and `<meta>` elements on the page.
    #
    # @param [metadata] - Dublin Core and Highwire metadata parsed from `<meta>` tags.

    def _getLinks(self, metadata: Optional[HTMLDocumentMetadata] = None) -> List[Link]:
        links: List[Link] = [Link(href=self._getDocumentHref())]

        # Extract links from `<link>` tags with certain `rel` values.
        linkElements = self.document.findall(".//link")
        for link in linkElements:
            rel = link.attrib.get("rel", None)
            if rel not in ["alternate", "canonical", "bookmark", "shortlink"]:
                continue

            if rel == "alternate":
                # Ignore RSS feed links.
                type_ = link.attrib.get("type", None)
                if type_ and re.match(r"^application\/(rss|atom)\+xml", type_):
                    continue
                # Ignore alternate languages.
                if link.attrib.get("hreflang", None):
                    continue
            href = link.attrib.get("href", None)
            if href:
                with suppress(Exception):
                    # Ignore URIs which cannot be parsed.
                    href = self._absoluteUrl(href)
                    links.append(Link(href=href, rel=rel, type=link.type))

        if not metadata:
            return links

        # Look for links in scholar metadata
        for name in metadata.highwire.keys():
            values = metadata.highwire[name]
            if name == "pdf_url":
                for url in values:
                    with suppress(Exception):
                        links.append(
                            Link(href=self._absoluteUrl(url), type="application/pdf")
                        )

            # Kind of a hack to express DOI identifiers as links but it's a
            # convenient place to look them up later, and somewhat sane since
            # they don't have a type.
            if name == "doi":
                for doi in values:
                    if not doi.startswith("doi:"):
                        doi = f"doi:{doi}"
                        links.append(Link(href=doi))

        # Look for links in Dublin Core data
        for name in metadata.dc.keys():
            values = metadata.dc[name]
            if name == "identifier":
                for id in values:
                    if doi.startswith("doi:"):
                        links.append(Link(href=id))

        # Look for a link to identify the resource in Dublin Core metadata
        dcRelationValues = metadata.dc["relation.ispartof"]
        dcIdentifierValues = metadata.dc["identifier"]
        if dcRelationValues and dcIdentifierValues:
            dcUrnRelationComponent = dcRelationValues[dcRelationValues.length - 1]
            dcUrnIdentifierComponent = dcIdentifierValues[dcIdentifierValues.length - 1]
            dcUrn = (
                "urn:x-dc:"
                + quote_plus(dcUrnRelationComponent)
                + "/"
                + quote_plus(dcUrnIdentifierComponent)
            )
            links.append(Link(href=dcUrn))

        return links

    def _getFavicon(self) -> Optional[str]:
        favicon = None
        for link in self.document.findall(".//link"):
            rel = link.attrib.get("rel", None)
            if rel in {"shortcut icon", "icon"}:
                if href := link.attrib.get("href", None):
                    with suppress(Exception):
                        favicon = self._absoluteUrl(href)
        return favicon

    def _baseUri(self):
        base = self.document.find("/head/base")
        if base is not None and urlparse(base.text).scheme in self._allowedSchemes:
            return base.text

    # Convert a possibly relative URI to an absolute one. This will throw an
    # exception if the URL cannot be parsed.
    def _absoluteUrl(self, url: str) -> str:
        return normalize(url, self._baseUri())

    # Get the true URI record when it's masked via a different protocol.
    # This happens when an href is set with a uri using the 'blob:' protocol
    # but the document can set a different uri through a <base> tag.

    _allowedSchemes = {"http:", "https:", "file:"}

    def _getDocumentHref(self) -> str:
        href = self.location

        # Use the current document location if it has a recognized scheme.
        scheme = urlparse(href).scheme
        if scheme in self._allowedSchemes:
            return href

        # Otherwise, try using the location specified by the <base> element.
        base = self._baseUri()
        if base:
            return base

        # Fall back to returning the document URI, even though the scheme is not
        # in the allowed list.
        return href
