"""Document parsing for the document repository."""

import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List, Optional, Union


_REQUIRED_KEYS = ("ord_no", "topic", "title")
_DOC_EXTENSIONS = ('.txt', '.md')


@dataclass
class Document:
    """Metadata as key-values"""
    document_metadata: Dict
    """Body of text - organized by lines"""
    body: List[str] = field(default_factory=list)

    def match_audience(self, agent_name: str) -> bool:
        """Check if a specific agent is the target audience.

        If "audience" is not specified, the document targets all agents.
        """
        audience = self.document_metadata.get("audience")
        if audience is None:
            return True
        return agent_name in audience


@dataclass
class Topic:
    """A group of documents sharing the same topic name."""
    ord_no: int
    topic: str
    documents: List[Document] = field(default_factory=list)

    def count(self, agent_name: str) -> int:
        """Count documents accessible to a given agent.

        Args:
            agent_name: the agent name to check audience against

        Returns:
            Number of documents matching the agent's audience.
        """
        return sum(1 for doc in self.documents if doc.match_audience(agent_name))


def parse_document(document: str) -> Document:
    """Parse a document string into a Document instance.

    Metadata lines have the form ``# key: value`` and must appear before
    the body.  The keys ``ord_no``, ``topic`` and ``title`` are required.

    Args:
        document: contents of the input file

    Returns:
        Document: a properly parsed Document instance

    Raises:
        ValueError: if required headers are missing
    """
    metadata: Dict = {}
    body_lines: List[str] = []
    in_body = False

    for line in document.split('\n'):
        if not in_body and line.startswith('# '):
            key, sep, value = line[2:].partition(': ')
            if sep:
                metadata[key] = _parse_metadata_value(key, value)
                continue
        in_body = True
        body_lines.append(line)

    # Strip leading blank lines from body
    while body_lines and not body_lines[0]:
        body_lines.pop(0)
    # Strip trailing blank lines from body
    while body_lines and not body_lines[-1]:
        body_lines.pop()

    missing = [k for k in _REQUIRED_KEYS if k not in metadata]
    if missing:
        raise ValueError(f"Missing required headers: {', '.join(missing)}")

    return Document(document_metadata=metadata, body=body_lines)


@lru_cache
def load_documents(path: str) -> List[Topic]:
    """Load all document files from a directory (including subdirectories).

    Only ``.txt`` and ``.md`` files are considered.  Documents are grouped by
    their ``topic`` metadata and sorted by ``ord_no`` within each topic.
    Topics receive consecutive ``ord_no`` values in the order they are first
    encountered.

    Args:
        path: root directory to search for document files

    Returns:
        List of Topic instances, each containing its sorted documents.

    Raises:
        ValueError: if any document file cannot be parsed
    """
    topics_map: Dict[str, Topic] = {}

    for root, dirs, files in os.walk(path):
        dirs.sort()
        for filename in sorted(files):
            if not filename.endswith(_DOC_EXTENSIONS):
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            doc = parse_document(content)
            topic_name = doc.document_metadata["topic"]
            if topic_name not in topics_map:
                topics_map[topic_name] = Topic(
                    ord_no=len(topics_map),
                    topic=topic_name,
                )
            topics_map[topic_name].documents.append(doc)

    for topic in topics_map.values():
        topic.documents.sort(key=lambda d: d.document_metadata["ord_no"])

    return list(topics_map.values())


def find_topic(key: Union[int, str], agent_name: str,
               all_topics: List[Topic]) -> Optional[Topic]:
    """Find a topic by index, name, or name fragment.

    A topic is only considered if it contains at least one document accessible
    to the given agent.

    Args:
        key: topic index (int), exact name, or case-insensitive name fragment
        agent_name: the requesting agent name (for audience filtering)
        all_topics: the full list of topics from ``load_documents``

    Returns:
        The matching Topic, or None if no topic matches.

    Raises:
        ValueError: if more than one topic matches the key, listing the
            matching topic names and indexes.
    """
    accessible = [t for t in all_topics if t.count(agent_name) > 0]

    if isinstance(key, int):
        for topic in accessible:
            if topic.ord_no == key:
                return topic
        return None

    key_lower = key.lower()

    # Exact name match takes priority
    for topic in accessible:
        if topic.topic.lower() == key_lower:
            return topic

    matches = [t for t in accessible if key_lower in t.topic.lower()]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None

    listing = ", ".join(f"{t.topic} (#{t.ord_no})" for t in matches)
    raise ValueError(f"Ambiguous topic '{key}', matches: {listing}")


def find_document(key: Union[int, str], agent_name: str,
                  topic: Topic) -> Optional[Document]:
    """Find a document within a topic by index, title, or title fragment.

    Only documents accessible to the given agent are considered.  An exact
    (case-insensitive) title match takes priority over fragment matches.

    Args:
        key: document index (int), exact title, or case-insensitive title fragment
        agent_name: the requesting agent name (for audience filtering)
        topic: the topic to search within

    Returns:
        The matching Document, or None if no document matches.

    Raises:
        ValueError: if more than one document matches, listing matching
            document titles and indexes.
    """
    accessible = [d for d in topic.documents if d.match_audience(agent_name)]

    if isinstance(key, int):
        for doc in accessible:
            if doc.document_metadata["ord_no"] == key:
                return doc
        return None

    key_lower = key.lower()

    # Exact title match takes priority
    for doc in accessible:
        if doc.document_metadata["title"].lower() == key_lower:
            return doc

    matches = [d for d in accessible
               if key_lower in d.document_metadata["title"].lower()]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        listing = ", ".join(
            f"{d.document_metadata['title']} (#{d.document_metadata['ord_no']})"
            for d in matches
        )
        raise ValueError(f"Ambiguous document '{key}', matches: {listing}")

    # Fuzzy match: pick the highest-similarity title above 90%
    best_doc = None
    best_ratio = 0.0
    for doc in accessible:
        ratio = SequenceMatcher(
            None, key_lower, doc.document_metadata["title"].lower()
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_doc = doc

    if best_ratio >= 0.9:
        return best_doc
    return None


def _parse_metadata_value(key: str, value: str):
    """Convert metadata value based on key type."""
    if key == "audience":
        return [name.strip() for name in value.split(',')]
    try:
        return int(value)
    except ValueError:
        return value
