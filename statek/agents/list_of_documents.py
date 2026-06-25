# Copyright 2026 Statek authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
list_of_documents meta-tool for STATEK agents.

Provides a tool for listing topics and documents from the configured
documents directory. The base directory is read from StatekSettings.documents_dir
(env: STATEK_DOCUMENTS_DIR).
"""

# pylint: disable=unused-argument

import logging
from typing import Optional, Union

from statek.document import load_documents, find_topic, find_document
from statek.settings import get_statek_settings
from statek.system import tool
from statek.utils import perm_ctx_set, perm_ctx_get

log = logging.getLogger('statek')


def _get_documents_dir() -> Optional[str]:
    """Return the documents base directory from StatekSettings, or None if not set."""
    return get_statek_settings().documents_dir


@tool(system=True)
def list_of_documents(agent_name: str, documents_dir: str,
                      topic: Optional[Union[int, str]] = None,
                      start_index: int = 0, limit: int = 25, **kwargs):  # pylint: disable=unused-argument
    """Lists available topics or documents within a topic.

    When called without a topic, lists all topics accessible to the agent.
    When called with a topic (ID, name, or fragment), lists documents in that topic.

    Args:
        agent_name: The agent role (for audience filtering).
        documents_dir: Path to the documents directory.
        topic: Optional topic ID, name, or name fragment.
        start_index: Index of the first item to show (default: 0).
        limit: Maximum number of items to show (default: 25).

    Returns:
        None. Prints the list to console.
    """
    if not documents_dir:
        print("# No documents found")
        return

    all_topics = load_documents(documents_dir)
    if not all_topics:
        print("# No documents found")
        return

    if topic is None:
        _list_topics(agent_name, all_topics, start_index, limit)
    else:
        _list_documents(topic, agent_name, all_topics, start_index, limit)


def _list_topics(agent_name, all_topics, start_index, limit):
    accessible = [t for t in all_topics if t.count(agent_name) > 0]
    total = len(accessible)
    print(f"# Topic ID: Topic name ({total} total)")
    for t in accessible[start_index:start_index + limit]:
        print(f"{t.ord_no}: {t.topic}")


def _list_documents(topic_key, agent_name, all_topics, start_index, limit):
    matched = find_topic(topic_key, agent_name, all_topics)
    if matched is None:
        print(f"# Topic '{topic_key}' not found")
        return

    perm_ctx_set(last_topic_id=matched.ord_no)

    docs = [d for d in matched.documents if d.match_audience(agent_name)]
    total = len(docs)
    print(f"# Document ID: Document name ({total} total)")
    for d in docs[start_index:start_index + limit]:
        print(f"{d.document_metadata['ord_no']}: {d.document_metadata['title']}")


def _resolve_topic(topic_key, agent_name, all_topics):
    """Resolve a topic key, falling back to last_topic_id from context.

    Raises:
        ValueError: if topic_key is None and no last_topic_id exists.
    """
    if topic_key is None:
        topic_key = perm_ctx_get("last_topic_id", None)
        if topic_key is None:
            raise ValueError(
                "No topic specified and no last_topic_id in context"
            )
    matched = find_topic(topic_key, agent_name, all_topics)
    if matched is not None:
        perm_ctx_set(last_topic_id=matched.ord_no)
    return matched


@tool(system=True)
def show_document(agent_name: str, documents_dir: str,  # pylint: disable=too-many-positional-arguments,too-many-arguments
                  key: Union[int, str], topic: Optional[Union[int, str]] = None,
                  start_from: int = 0, limit: int = 50, **kwargs):
    """Shows the contents of a specific document.

    Locates and prints a document from a specific topic or the last accessed
    topic.  By default the first 50 lines are printed.

    Args:
        agent_name: The agent role (for audience filtering).
        documents_dir: Path to the documents directory.
        key: Document ID, title, or title fragment (case-insensitive).
        topic: Optional topic ID, name, or fragment.  Falls back to the
            last accessed topic (last_topic_id) if not specified.
        start_from: First line number to print (default: 0).
        limit: Maximum number of lines to print (default: 50).

    Returns:
        None. Prints the document contents to console.

    Raises:
        ValueError: if no topic is specified and no last_topic_id exists.
    """
    if not documents_dir:
        print("# No documents found")
        return

    all_topics = load_documents(documents_dir)
    if not all_topics:
        print("# No documents found")
        return

    matched = _resolve_topic(topic, agent_name, all_topics)
    if matched is None:
        print(f"# Topic '{topic}' not found")
        return

    doc = find_document(key, agent_name, matched)
    if doc is None:
        print(f"# Document '{key}' not found in topic '{matched.topic}'")
        return

    body_slice = doc.body[start_from:start_from + limit]
    print(f"# {doc.document_metadata['title']}"
          f" (lines {start_from}-{start_from + len(body_slice)}"
          f"/{len(doc.body)})")
    for line in body_slice:
        print(line)
