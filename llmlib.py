import os
import textwrap

import openai
import tiktoken
from diskcache import Cache
import appdirs
import re

def split_separator(text, separator):
    """Split a text using a separator, but keep the separator in the result.
    
    Separator must be a regex with two capture groups. The first one is kept
    with the text before the split, the second one is kept with the text after
    the split."""
    parts = []
    remainder = text
    before_remainder = ""
    while remainder:
        split = re.split(separator, remainder, maxsplit=1, flags=re.MULTILINE)
        if len(split) == 1:
            parts.append(before_remainder + remainder)
            break
        part, after_part, next_before_remainder, next_remainder = split
        parts.append(before_remainder + part + after_part)
        remainder = next_remainder
        before_remainder = next_before_remainder
    return parts

class Api:
    def ask(self, prompt):
        raise NotImplementedError

    def token_count(self, prompt):
        raise NotImplementedError

    def max_token_count(self):
        raise NotImplementedError

class Openai(Api):
    def __init__(self, model="gpt-3.5-turbo", verbose=False, api_key=None):
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.verbose = verbose

    def ask(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0]['message']['content']
        except openai.error.InvalidRequestError as e:
            e._message += f"; computed token length={self.token_count(prompt)}"
            raise
        return result

    def token_count(self, prompt):
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(prompt))

    def max_token_count(self):
        return {
                "gpt-4": 8192,
                "gpt-3.5-turbo": 4097
            }.get(self.model, 4096)

    def __repr__(self) -> str:
        return f"Openai(model={self.model})"

class Llm:
    def __init__(self, api : Api, verbose=False):
        self.api = api
        self.verbose = verbose
        self.cache = Cache(appdirs.user_cache_dir("llmlib"))
        self.counters = {}
        log_dir = appdirs.user_log_dir("llmlib")
        log_path = os.path.join(log_dir, "log.txt")
        if self.verbose:
            print(f"Logging to {log_path}")
        os.makedirs(log_dir, exist_ok=True)
        self.log_fd = open(log_path, "a")

    def log(self, text):
        self.log_fd.write(text)
        if not text.endswith("\n"):
            self.log_fd.write("\n")

    def ask(self, prompt : str):
        self.log("\n".join(textwrap.wrap(f"Ask {self.api!r}: {prompt}", subsequent_indent="    ")))
        if self.verbose:
            print(f"Ask {self.api!r}: {prompt[:60]!r}")

        assert len(prompt) > 25

        cache_key = ("ask", repr(self.api), prompt)
        result = self.cache.get(cache_key)
        self.increment_counter(f"ask-{self.api!r}")

        if result:
            self.increment_counter(f"ask-{self.api!r}-hit")
            cached = " (cached)"
        else:
            self.increment_counter(f"ask-{self.api!r}-miss")
            result = self.api.ask(prompt)
            cached = ""

        self.log("\n".join(textwrap.wrap(f"Response{cached}: {result}", subsequent_indent="    ")))
        if self.verbose:
            print(f"Response{cached}: {result[:60]!r}")

        self.cache[cache_key] = result

        return result

    def increment_counter(self, name):
        self.counters.setdefault(name, 0)
        self.counters[name] += 1

    def split_markdown(self, text, token_limit=None):
        """Split a markdown text to fit the given token limit."""
        return self.split_text(text, token_limit=token_limit,
                               separators=(
                                   r"()(^# .*$)",
                                   r"()(^## .*$)",
                                   r"()(^### .*$)",
                                   r"()(^#### .*$)",
                                   r"(\n(?:\s*\n)+)",
                                   r"(\n+)",
                                   r"(\s+)"))

    def split_text(self, text, token_limit=None, separators=(r"(\n(?:\s*\n)+)()", r"(\n+)()", r"(\s+)()")):
        """Split a text to fit the given token limit."""
        if token_limit is None:
            token_limit = self.api.max_token_count()

        # Split text into parts that are each short enough to fit the token limit.
        short_parts = []
        for part in split_separator(text, separators[0]):
            if self.api.token_count(part) > token_limit:
                short_parts.extend(self.split_text(part, token_limit, separators[1:]))
            else:
                short_parts.append(part)

        # Combine short parts into longer ones that still fit the token limit.
        parts = []
        for part in short_parts:
            if parts and self.api.token_count(parts[-1] + part) <= token_limit:
                parts[-1] += part
            else:
                parts.append(part)
        return parts

    def summarize(self, text, token_limit=None, prompt="Summarize:", max_iterations=10):
        """Summarize a text to fit the given token limit."""
        max_tokens = self.api.max_token_count() - self.api.token_count(prompt)
        if token_limit is None:
            token_limit = max_tokens
        else:
            token_limit = min(token_limit, max_tokens)
        for _ in range(max_iterations):
            if self.api.token_count(text) <= token_limit:
                break
            text = "\n\n".join(
                self.ask(f"{prompt} {part}")
                for part in self.split_text(text, token_limit=token_limit))
        return text
