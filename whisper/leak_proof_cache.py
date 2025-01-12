import functools
import weakref
from typing import Callable, Generic, ParamSpec, TypeVar

# Type variables for generic type annotations
T = TypeVar("T")  # Return type of the cached function
P = ParamSpec("P")  # Parameters of the cached function


class leakproofcache(Generic[P, T]):
    """
    A decorator to create a per-instance cache for instance methods using
    a weak reference to `self`, preventing memory leaks.

    Supports customization by allowing the choice of cache function (e.g., functools.lru_cache or functools.cache).
    """

    def __init__(self, cache_func: Callable = functools.lru_cache, *cache_args, **cache_kwargs):
        """
        Initialize the decorator with a cache function and its arguments.

        Args:
            cache_func (Callable): The caching function to use (e.g., functools.lru_cache).
            *cache_args: Positional arguments for the cache function.
            **cache_kwargs: Keyword arguments for the cache function.
        """
        self.cache_func = cache_func
        self.cache_args = cache_args
        self.cache_kwargs = cache_kwargs

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """
        Wrap the target function with the caching logic.

        Args:
            func (Callable): The instance method to cache.

        Returns:
            Callable: The wrapped function with caching applied.
        """
        @functools.wraps(func)
        def wrapped_func(self, *args: P.args, **kwargs: P.kwargs) -> T:
            # Use a weak reference to the instance (self) to prevent memory leaks
            self_weak = weakref.ref(self)

            # Define the actual cached method, using the chosen cache function
            @functools.wraps(func)
            @self.cache_func(*self.cache_args, **self.cache_kwargs)
            def cached_method(*args: P.args, **kwargs: P.kwargs) -> T:
                # Dereference the weak reference to get the instance
                instance = self_weak()
                if instance is None:
                    raise ReferenceError("Instance has been garbage collected")
                return func(instance, *args, **kwargs)

            # Store the cached method on the instance (per-instance cache)
            setattr(self, func.__name__, cached_method)

            # Call the cached method on first invocation
            return cached_method(*args, **kwargs)

        return wrapped_func