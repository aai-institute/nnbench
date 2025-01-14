from collections.abc import Generator

import pytest

from nnbench.memo import Memo, cached_memo, clear_memo_cache, memo_cache_size


@pytest.fixture
def clear_memos() -> Generator[None, None, None]:
    try:
        clear_memo_cache()
        yield
    finally:
        clear_memo_cache()


class MyMemo(Memo[int]):
    @cached_memo
    def __call__(self):
        return 0


def test_memo_caching(clear_memos):
    m = MyMemo()
    assert memo_cache_size() == 0
    m()
    assert memo_cache_size() == 1
    m()
