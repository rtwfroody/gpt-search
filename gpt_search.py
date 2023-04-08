#!/bin/env python3

"""Combine GPT with DuckDuckGo to answer questions."""

import argparse
import datetime
import json
import re
import sys
import textwrap

from bs4 import BeautifulSoup
from diskcache import Cache
from duckduckgo_search import ddg
from markdownify import MarkdownConverter
import appdirs
import requests

import llmlib

max_token_count = {
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4097
}

def simplify_html(html):
    """Convert HTML to markdown, removing some tags and links."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove unwanted tags
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    # Remove links. They're not helpful.
    for tag in soup.find_all("a"):
        del tag["href"]
    for tag in soup.find_all("img"):
        del tag["src"]
    soup.smooth()

    # Turn HTML into markdown, which is concise but will attempt to
    # preserve at least some formatting
    text = MarkdownConverter().convert_soup(soup)
    text = re.sub(r"\n(\s*\n)+", "\n\n", text)
    return text

def extract_title(html):
    """Extract the title from an HTML document."""
    soup = BeautifulSoup(html, 'html.parser')
    return soup.title.string

class GptSearch:
    """Combine GPT with DuckDuckGo to answer questions."""
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.verbose = False

        self.cache = Cache(appdirs.user_cache_dir("gpt_search"))

        self.llm = None

    def fetch(self, url):
        """Fetch a URL, caching the result."""
        key = ("fetch", url)
        if key in self.cache:
            if self.verbose:
                print("Cache hit for", key)
            return self.cache[key]

        try:
            # Fetch the URL
            response = requests.get(url, timeout=10)

            # Check if the request was successful
            if response.status_code != 200:
                print(f"Error fetching {url}: {response.status_code}")
                return None

            self.cache[key] = response.content
            return response.content

        except requests.RequestException as exception:
            print(f"Error fetching {url}: {exception}")
            return None

    def ddg_search(self, topic):
        """Search DuckDuckGo for a topic, caching the result."""
        key = ("ddg_search", topic)
        if key in self.cache:
            if self.verbose:
                print("Cache hit for", key)
            return self.cache[key]

        if self.verbose:
            print("Search DDG for:", topic)
        result = ddg(topic)

        self.cache[key] = result

        return result

    def ddg_top_hit(self, topic, skip=()):
        """Search DuckDuckGo for a topic, and return the top hit."""
        results = self.ddg_search(topic)
        for result in results:
            if result['href'] in skip:
                continue
            if self.verbose:
                print("  Fetching", result['href'])
            html = self.fetch(result['href'])
            if html:
                title = extract_title(html)
                content = simplify_html(html)
                if content:
                    return_value = (result['href'], str(title), content)
                    return return_value
        return (None, None, None)

    def fetch_sources(self, search_prompt):
        """Fetch sources for a question."""
        search_text = self.llm.ask(search_prompt)
        searches = json.loads(search_text)
        background_text = ""
        sources = []
        for search in searches:
            source, title, content = self.ddg_top_hit(search,
                                                      skip=[source for source, _ in sources])
            if not source:
                continue
            background_text += f"# {search}\n\n{content}\n\n"
            sources.append((source, title))
        return background_text, sources

    def main(self):
        """Main function that parses arguments etc."""
        parser = argparse.ArgumentParser(
            description="""Combine GPT with DuckDuckGo to answer questions.
            
            Beware that this will perform up to 5 GPT queries.""")
        parser.add_argument("--4", "-4", help="Use GPT4 (slower, costs more money)",
                            dest='gpt4', action="store_true")
        parser.add_argument("--verbose", "-v", help="Verbose output", action="store_true")
        parser.add_argument("question", help="What do you want to ask?")
        args = parser.parse_args()

        if args.gpt4:
            self.model = "gpt-4"
        else:
            self.model = "gpt-3.5-turbo"
        self.verbose = args.verbose

        self.llm = llmlib.Llm(llmlib.Openai(self.model), verbose=self.verbose)

        today_prompt = f"Today is {datetime.date.today().strftime('%a, %b %e, %Y')}."
        search_prompt = ("# Background\n\n"
                        f"{today_prompt}\n\n"
                        f"Prepare for this prompt: {args.question}\n\n"
                        "# Prompt\n\n"
                        "What 3 Internet search topics would help you answer this "
                        "question? Answer in a JSON list only.")
        background_text, sources = self.fetch_sources(search_prompt)

        background_text = self.llm.summarize(background_text,
                prompt=f"{today_prompt}\n\n"
                "You provide helpful and complete answers.\n\n"
                f"Make a list of facts that would help with: {args.question}\n\n")

        answer = self.llm.ask("\n\n".join([
            "# Background",
            background_text,
            today_prompt,
            "You provide helpful and complete answers.",
            "# Prompt",
            f"{args.question}"]))
        paragraphs = answer.splitlines()
        wrapped_paragraphs = [textwrap.wrap(p) for p in paragraphs]
        print("\n".join("\n".join(p) for p in wrapped_paragraphs))
        print(f"({self.llm.counter_string()})")
        print()
        print("Sources:")
        for source, title in sources:
            print(f"* [{title}]({source})")

if __name__ == "__main__":
    gptSearch = GptSearch()
    sys.exit(gptSearch.main())
