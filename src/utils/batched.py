from typing import TypeVar, Iterator, List
from tqdm import tqdm

T = TypeVar('T', covariant=True)


def batched(xs: Iterator[T], batch_size: int, total:int=0) -> Iterator[List[T]]:
    buf: List[T] = []

    if total:
        for x in tqdm(xs, total=total):
            buf.append(x)
            if len(buf) == batch_size:
                yield buf
                buf.clear()
    else:
        for x in xs:
            buf.append(x)
            if len(buf) == batch_size:
                yield buf
                buf.clear()

    if len(buf) != 0:
        yield buf


def chunks(l, n):
    n = len(l) // n
    return list(l[i:i + n] for i in range(0, len(l), n))