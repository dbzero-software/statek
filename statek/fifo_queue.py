"""FiFoQueue - a first-in first-out queue backed by a db0.index."""
# pylint: disable=no-member
from typing import Callable, Dict, List, Optional

import dbzero as db0


@db0.memo
class FQ_Item:  # pylint: disable=invalid-name
    """Single queue entry; stores kwargs and its own integer key."""

    def __init__(self, key: int, prefix=None, **kwargs):
        db0.set_prefix(self, prefix)
        self.__key = key
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def queue_key(self) -> int:
        return self.__key

    def to_dict(self) -> Dict:
        return {
            key: value
            for key, value in vars(self).items()
            if key != "_FQ_Item__key"
        }


@db0.memo
class FiFoQueue:
    """FIFO queue container backed by a db0.index with ever-incrementing integer keys."""

    def __init__(self):
        self.__items = db0.index()
        self.__next_key = 0

    def is_empty(self) -> bool:
        return len(self.__items) == 0

    def push_back(self, **kwargs):
        """Append a single element to the back of the queue."""
        prefix = db0.get_prefix_of(self).name
        self.__items.add(self.__next_key, FQ_Item(self.__next_key,prefix=prefix, **kwargs))
        self.__next_key += 1

    def has_item(self, filter: Callable, max_scan: int = 100) -> Optional[bool]:  # pylint: disable=redefined-builtin
        """Check whether any queued item matches *filter* without mutating the queue.

        Items are scanned in the underlying index iteration order. Returns
        ``None`` when determining the answer would require scanning beyond
        *max_scan* queue entries.
        """
        scanned = 0
        for item in self.__items.select():
            if not isinstance(item, FQ_Item):
                continue

            if scanned >= max_scan:
                return None

            scanned += 1
            if filter(**item.to_dict()):
                return True

        return False

    def pop_front(self, count, filter: Callable = None) -> List[Dict]:  # pylint: disable=redefined-builtin
        """Retrieve and remove up to *count* first elements from the queue.

        Items are visited in FIFO order. Each item that passes *filter* (or all
        items when *filter* is None) is removed from the queue and included in
        the result. Items that do not pass *filter* are left in the queue.
        Scanning stops once *count* matching items have been collected.

        Args:
            count: Maximum number of elements to retrieve.
            filter: Optional callable invoked as ``filter(**item_kwargs)``.
                    Items for which it returns a falsy value are skipped.
        """
        if filter is not None:
            # Guard against non-FQ_Item objects that db0.filter may encounter in
            # multi-prefix contexts (cross-prefix queries can include unexpected types).
            def _item_filter(item):
                return isinstance(item, FQ_Item) and filter(**item.to_dict())
            query = db0.filter(_item_filter, self.__items.select())
        else:
            query = self.__items.select()

        results = []
        items_to_remove = []

        for item in self.__items.sort(query):
            if len(results) >= count:
                break
            items_to_remove.append(item)
            results.append(item.to_dict())

        for item in items_to_remove:
            self.__items.remove(item.queue_key, item)

        return results
