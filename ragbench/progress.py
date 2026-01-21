from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def progress_iter(items: Iterable[T], desc: str | None = None, total: int | None = None) -> Iterator[T]:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return iter(items)
    return tqdm(items, desc=desc, total=total, ascii=True, leave=False, dynamic_ncols=True, mininterval=0.5)
