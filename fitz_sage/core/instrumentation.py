# fitz_sage/core/instrumentation.py
"""
Instrumentation system for benchmarking plugin performance.

This module provides a transparent proxy system that can intercept plugin
method calls for timing, cost tracking, and other metrics collection.

Design:
    - Hooks are registered by enterprise/benchmarking code
    - When no hooks are registered, there is zero overhead (no proxies created)
    - Proxies are transparent - callers don't know they're using a proxy
    - Thread-safe hook registration

Usage (enterprise side):
    from fitz_sage.core.instrumentation import register_hook, BenchmarkHook

    class TimingHook(BenchmarkHook):
        def on_call_start(self, layer, plugin_name, method, args, kwargs):
            return {"start": time.perf_counter()}

        def on_call_end(self, context, result, error):
            duration = time.perf_counter() - context["start"]
            print(f"Call took {duration*1000:.2f}ms")

    register_hook(TimingHook())
    # Now all plugin calls will be timed
"""

from __future__ import annotations

import logging
import threading
from functools import wraps
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# =============================================================================
# Hook Protocol
# =============================================================================


@runtime_checkable
class BenchmarkHook(Protocol):
    """
    Protocol for benchmark hooks.

    Hooks are notified before and after each instrumented method call.
    The on_call_start method returns a context object that is passed to
    on_call_end, allowing hooks to track state across a call.

    Example:
        class TimingHook:
            def on_call_start(self, layer, plugin_name, method, args, kwargs):
                return {"start": time.perf_counter(), "layer": layer}

            def on_call_end(self, context, result, error):
                duration = time.perf_counter() - context["start"]
                self.metrics.append({
                    "layer": context["layer"],
                    "duration_ms": duration * 1000,
                    "error": str(error) if error else None,
                })
    """

    def on_call_start(
        self,
        layer: str,
        plugin_name: str,
        method: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """
        Called before method execution.

        Args:
            layer: Plugin layer (e.g., "engine", "llm.chat", "chunking")
            plugin_name: Name of the plugin (e.g., "fitz_krag", "openai", "pinecone")
            method: Method being called (e.g., "answer", "chat", "search")
            args: Positional arguments to the method
            kwargs: Keyword arguments to the method

        Returns:
            Context object passed to on_call_end (can be any type)
        """
        ...

    def on_call_end(
        self,
        context: Any,
        result: Any,
        error: Exception | None,
    ) -> None:
        """
        Called after method execution.

        Args:
            context: Context object returned by on_call_start
            result: Return value of the method (None if error occurred)
            error: Exception raised by the method (None if successful)
        """
        ...


# Sentinel value to indicate no cached result
_NO_CACHE = object()


class CachingHook(Protocol):
    """
    Extended hook protocol that supports result caching.

    CachingHooks can intercept method calls and return cached results,
    bypassing the actual method execution. This is useful for caching
    expensive operations during benchmarks.

    Example:
        class ChatCacheHook:
            def __init__(self):
                self._cache = {}

            def get_cached_result(self, layer, plugin_name, method, args, kwargs):
                if layer == "llm.chat" and method == "chat":
                    key = (plugin_name, args[0])  # (plugin, prompt)
                    return self._cache.get(key, _NO_CACHE)
                return _NO_CACHE

            def cache_result(self, layer, plugin_name, method, args, kwargs, result):
                if layer == "llm.chat" and method == "chat":
                    key = (plugin_name, args[0])
                    self._cache[key] = result
    """

    def get_cached_result(
        self,
        layer: str,
        plugin_name: str,
        method: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """
        Check if a cached result is available.

        Args:
            layer: Plugin layer
            plugin_name: Plugin name
            method: Method being called
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Cached result if available, or _NO_CACHE sentinel if not cached
        """
        ...

    def cache_result(
        self,
        layer: str,
        plugin_name: str,
        method: str,
        args: tuple,
        kwargs: dict,
        result: Any,
    ) -> None:
        """
        Store a result in the cache.

        Args:
            layer: Plugin layer
            plugin_name: Plugin name
            method: Method being called
            args: Positional arguments
            kwargs: Keyword arguments
            result: Result to cache
        """
        ...


# =============================================================================
# Global Hook Registry
# =============================================================================

_hooks: list[BenchmarkHook] = []
_lock = threading.Lock()


def register_hook(hook: BenchmarkHook) -> None:
    """
    Register a benchmark hook.

    Once registered, the hook will be notified of all instrumented method calls.
    Hooks are called in the order they were registered.

    Args:
        hook: Hook implementing the BenchmarkHook protocol

    Example:
        hook = TimingHook()
        register_hook(hook)
        # ... run queries ...
        print(hook.metrics)
    """
    with _lock:
        if hook not in _hooks:
            _hooks.append(hook)


def unregister_hook(hook: BenchmarkHook) -> None:
    """
    Unregister a benchmark hook.

    Args:
        hook: Previously registered hook

    Raises:
        ValueError: If hook was not registered
    """
    with _lock:
        _hooks.remove(hook)


def clear_hooks() -> None:
    """Remove all registered hooks."""
    with _lock:
        _hooks.clear()


def has_hooks() -> bool:
    """Check if any hooks are registered."""
    return len(_hooks) > 0


def get_hooks() -> list[BenchmarkHook]:
    """Get a copy of the registered hooks list."""
    with _lock:
        return list(_hooks)


# =============================================================================
# Instrumented Proxy
# =============================================================================


class InstrumentedProxy:
    """
    Transparent proxy that intercepts method calls for benchmarking.

    This proxy wraps any object and notifies registered hooks when methods
    are called. It is transparent to callers - the proxy behaves exactly
    like the wrapped object.

    Args:
        target: Object to wrap
        layer: Plugin layer name (e.g., "engine", "llm.chat")
        plugin_name: Plugin name (e.g., "fitz_krag", "openai")
        methods_to_track: If provided, only these methods are instrumented.
                         If None, all public methods are instrumented.

    Example:
        chat = get_chat("openai")
        proxy = InstrumentedProxy(chat, "llm.chat", "openai", {"chat"})
        proxy.chat(messages)  # Hook is notified
    """

    __slots__ = ("_target", "_layer", "_plugin_name", "_methods_to_track")

    def __init__(
        self,
        target: Any,
        layer: str,
        plugin_name: str,
        methods_to_track: set[str] | None = None,
    ):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_plugin_name", plugin_name)
        object.__setattr__(self, "_methods_to_track", methods_to_track)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)

        # Only wrap callable methods
        if not callable(attr):
            return attr

        # Skip private/dunder methods
        if name.startswith("_"):
            return attr

        # If specific methods listed, only track those
        if self._methods_to_track is not None and name not in self._methods_to_track:
            return attr

        return self._wrap_method(attr, name)

    def _wrap_method(self, method: Callable, method_name: str) -> Callable:
        """Wrap a method to notify hooks before/after execution."""

        @wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Fast path: no hooks registered
            if not _hooks:
                return method(*args, **kwargs)

            # Check caching hooks first - if any returns a cached result, use it
            for hook in _hooks:
                if hasattr(hook, "get_cached_result"):
                    try:
                        cached = hook.get_cached_result(
                            layer=self._layer,
                            plugin_name=self._plugin_name,
                            method=method_name,
                            args=args,
                            kwargs=kwargs,
                        )
                        if cached is not _NO_CACHE:
                            # Cache hit - return without calling actual method
                            return cached
                    except Exception as e:
                        logger.debug(f"Hook get_cached_result failed: {e}")

            # Notify all hooks: start
            contexts: list[tuple[BenchmarkHook, Any]] = []
            for hook in _hooks:
                try:
                    ctx = hook.on_call_start(
                        layer=self._layer,
                        plugin_name=self._plugin_name,
                        method=method_name,
                        args=args,
                        kwargs=kwargs,
                    )
                    contexts.append((hook, ctx))
                except Exception as e:
                    # Don't let hook errors break the actual call
                    logger.debug(f"Hook on_call_start failed: {e}")

            # Execute the actual method
            error: Exception | None = None
            result: Any = None
            try:
                result = method(*args, **kwargs)

                # Store result in caching hooks
                for hook in _hooks:
                    if hasattr(hook, "cache_result"):
                        try:
                            hook.cache_result(
                                layer=self._layer,
                                plugin_name=self._plugin_name,
                                method=method_name,
                                args=args,
                                kwargs=kwargs,
                                result=result,
                            )
                        except Exception as e:
                            logger.debug(f"Hook cache_result failed: {e}")

                return result
            except Exception as e:
                error = e
                raise
            finally:
                # Notify all hooks: end
                for hook, ctx in contexts:
                    try:
                        hook.on_call_end(ctx, result, error)
                    except Exception as e:
                        # Don't let hook errors break execution
                        logger.debug(f"Hook on_call_end failed: {e}")

        return wrapper

    def __repr__(self) -> str:
        return f"InstrumentedProxy({self._target!r}, layer={self._layer!r}, plugin={self._plugin_name!r})"

    def __str__(self) -> str:
        return str(self._target)


# =============================================================================
# Convenience Functions
# =============================================================================


def maybe_wrap(
    target: Any,
    layer: str,
    plugin_name: str,
    methods_to_track: set[str] | None = None,
) -> Any:
    """
    Wrap target in a proxy only if hooks are registered.

    This is the main entry point for instrumentation. Call this when
    returning plugins from registries to enable benchmarking.

    Args:
        target: Plugin instance to potentially wrap
        layer: Plugin layer (e.g., "engine", "llm.chat")
        plugin_name: Plugin name (e.g., "fitz_krag", "openai")
        methods_to_track: If provided, only these methods are instrumented

    Returns:
        Either the original target (if no hooks) or an InstrumentedProxy

    Example:
        def get_llm_plugin(name, type):
            plugin = _load_plugin(name, type)
            return maybe_wrap(plugin, f"llm.{type}", name, {"chat"})
    """
    if not has_hooks():
        return target
    return InstrumentedProxy(target, layer, plugin_name, methods_to_track)


def wrap(
    target: Any,
    layer: str,
    plugin_name: str,
    methods_to_track: set[str] | None = None,
) -> InstrumentedProxy:
    """
    Always wrap target in a proxy (even if no hooks registered).

    Use this when you want consistent proxy behavior regardless of hooks.

    Args:
        target: Plugin instance to wrap
        layer: Plugin layer
        plugin_name: Plugin name
        methods_to_track: Methods to instrument (None = all public methods)

    Returns:
        InstrumentedProxy wrapping the target
    """
    return InstrumentedProxy(target, layer, plugin_name, methods_to_track)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Hook protocols
    "BenchmarkHook",
    "CachingHook",
    "_NO_CACHE",
    # Hook management
    "register_hook",
    "unregister_hook",
    "clear_hooks",
    "has_hooks",
    "get_hooks",
    # Proxy
    "InstrumentedProxy",
    # Convenience
    "maybe_wrap",
    "wrap",
]
