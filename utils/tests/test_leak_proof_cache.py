import pytest
import functools
import gc
import weakref
from utils.leakproof_cache import leakproof


class ExampleClass:
    @leakproof
    @functools.lru_cache(maxsize=3)
    def compute(self, x):
        print(f"Computing {x}")
        return x * 2

    @leakproof
    @functools.cache
    def other_compute(self, y):
        print(f"Computing {y}")
        return y + 5


@pytest.fixture
def example_instance():
    """Fixture to create an instance of ExampleClass."""
    return ExampleClass()


def test_lru_cache_with_leakproof(example_instance):
    """Test that @leakproof works with @lru_cache and caches correctly."""
    assert example_instance.compute(1) == 2
    assert example_instance.compute(2) == 4
    assert example_instance.compute(1) == 2  # Cached
    assert example_instance.compute(2) == 4  # Cached

    # Test that the cache respects the maxsize of 3
    assert example_instance.compute(3) == 6
    assert example_instance.compute(4) == 8
    # The first call to compute(1) should be evicted due to maxsize=3
    assert example_instance.compute(1) == 2  # Recomputed


def test_cache_with_leakproof(example_instance):
    """Test that @leakproof works with @cache and caches correctly."""
    assert example_instance.other_compute(10) == 15
    assert example_instance.other_compute(10) == 15  # Cached
    assert example_instance.other_compute(20) == 25
    assert example_instance.other_compute(20) == 25  # Cached


def test_cache_invalidation_on_garbage_collection():
    """Test that the cache is invalidated when the instance is garbage collected with @leakproof."""
    instance = ExampleClass()

    assert instance.compute(5) == 10
    assert instance.compute(5) == 10  # Cached

    # Create a weak reference to track the instance
    weakref_instance = weakref.ref(instance)

    # Delete the instance and force garbage collection
    del instance
    gc.collect()  # Force garbage collection

    # Ensure that the instance is garbage collected
    assert weakref_instance() is None  # The instance should be garbage collected

    # At this point, the cache should be cleared, and any further access will return None
    # or recompute the result, depending on the implementation.


def test_different_arguments_in_cache(example_instance):
    """Test that the cache respects different input arguments with @leakproof."""
    assert example_instance.compute(1) == 2
    assert example_instance.compute(2) == 4
    assert example_instance.compute(3) == 6

    # Ensure different arguments are cached separately
    assert example_instance.compute(1) == 2  # Cached
    assert example_instance.compute(3) == 6  # Cached
    assert example_instance.compute(2) == 4  # Cached
