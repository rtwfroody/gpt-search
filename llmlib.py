import os
import textwrap

import openai
from diskcache import Cache
import appdirs

class Api:
    def ask(self, prompt):
        raise NotImplementedError

class Openai(Api):
    def __init__(self, model="gpt-3.5-turbo", verbose=False, api_key=None):
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.verbose = verbose

    def ask(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0]['message']['content']
        return result

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
        os.makedirs(log_dir, exist_ok=True)
        self.log = open(log_path, "a")

    def ask(self, prompt):
        self.log.write("\n".join(textwrap.wrap(f"Prompt: {prompt}", subsequent_indent="    ")))
        self.log.write("\n")

        assert len(prompt) > 25

        cache_key = ("ask", repr(self.api), prompt)
        result = self.cache.get(cache_key)
        self.increment_counter(f"ask-{self.api!r}")

        if result:
            self.increment_counter(f"ask-{self.api!r}-hit")
            self.log.write("    (cached)\n")
        else:
            self.increment_counter(f"ask-{self.api!r}-miss")

            result = self.api.ask(prompt)

        self.log.write("\n".join(textwrap.wrap(f"Response: {result}", subsequent_indent="    ")))
        self.log.write("\n")

        self.cache[cache_key] = result

        return result

    def increment_counter(self, name):
        self.counters.setdefault(name, 0)
        self.counters[name] += 1
