import weakref
import functools

def leakproof(wrapped):
    """
    A decorator that wraps around caching decorators (like @lru_cache or @cache)
    to ensure that the instance is weakly referenced. This prevents memory leaks
    by allowing the instance to be garbage collected if no strong references remain.

    The cache will be invalidated when the instance is garbage collected.

    Arguments
        wrapped (callable): The method to wrap with memory-safe caching.

    Returns
        callable: A wrapped method that uses weak references to the instance.
    """
    @functools.wraps(wrapped)
    def wrapper(self, *args, **kwargs):
        if not hasattr(wrapper, 'weak_self'):
            # Create a weak reference to the instance
            wrapper.weak_self = weakref.ref(self)
            # Set up a finalizer to clear the cache when the instance is garbage collected
            weakref.finalize(self, wrapped.cache_clear)

        # Get the strong reference to the instance
        strong_self = wrapper.weak_self()
        if strong_self is None:
            # If the instance is garbage collected, clear the cache and return None
            wrapped.cache_clear()
            return None

        # Call the wrapped function with the strong instance, *args, and **kwargs
        return wrapped(strong_self, *args, **kwargs)

    return wrapper