"""Request-scoped cancellation; tokens never grant model or process control."""
import re
import threading
import time


class GenerationCancelled(RuntimeError):
    pass


def check_cancelled(event):
    if event is not None and event.is_set():
        raise GenerationCancelled("Generation cancelled")


class GenerationJobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.active: dict[str, threading.Event] = {}
        self.pending: dict[str, float] = {}

    def _prune(self):
        now = time.monotonic()
        self.pending = {key: expiry for key, expiry in self.pending.items() if expiry > now}

    def start(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9-]{32,64}", identifier):
            raise ValueError("request_id must be an unpredictable hexadecimal request token")
        with self.lock:
            self._prune()
            if identifier in self.active:
                raise ValueError("request_id is already active")
            event = threading.Event()
            if self.pending.pop(identifier, None) is not None:
                event.set()
            self.active[identifier] = event
            return event

    def cancel(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9-]{32,64}", identifier):
            raise ValueError("request_id must be an unpredictable hexadecimal request token")
        with self.lock:
            self._prune()
            event = self.active.get(identifier)
            if event is None:
                # Stop can overtake the original POST on another connection.
                if identifier not in self.pending and len(self.pending) >= 32:
                    raise ValueError("Pending cancellation limit reached")
                self.pending[identifier] = time.monotonic() + 30
            else:
                event.set()
            return True

    def finish(self, identifier):
        with self.lock:
            self.active.pop(identifier, None)
            self.pending.pop(identifier, None)
