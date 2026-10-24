"""Explicit request budgets shared by local and private-endpoint serving."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RequestLimits:
    max_seq: int = 8192
    max_tokens: int = 1024
    max_image_size: int = 1024
    max_images: int = 4
    max_text_chars: int = 200_000
    max_messages: int = 64

    def validate(self):
        bounds = {"max_seq": (2, 262144), "max_tokens": (1, 32768),
                  "max_image_size": (1, 4096), "max_images": (1, 32),
                  "max_text_chars": (1, 4_000_000), "max_messages": (1, 512)}
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
        if self.max_tokens >= self.max_seq:
            raise ValueError("max_tokens must be smaller than max_seq")

    def describe(self):
        return asdict(self)


def limits_from(config):
    limits = RequestLimits(**{name: getattr(config, name) for name in RequestLimits.__dataclass_fields__})
    limits.validate()
    return limits
