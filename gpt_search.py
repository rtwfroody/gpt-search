#!/bin/env python3

from pprint import pprint
import argparse
import datetime
import json
import textwrap
import requests
import sys
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from markdownify import MarkdownConverter
from diskcache import Cache
import re
import appdirs
import llmlib

from duckduckgo_search import ddg
from bs4 import BeautifulSoup

max_token_count = {
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4097
}

def simplify_html(html):
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

class GptSearch(object):
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.verbose = False

        self.query_count = 0

        self.cache = Cache(appdirs.user_cache_dir("gpt_search"))

    def fetch(self, url):
        key = ("fetch", url)
        if key in self.cache:
            print("Cache hit for", key)
            return self.cache[key]

        try:
            # Fetch the URL
            response = requests.get(url, timeout=10)

            # Check if the request was successful
            if response.status_code == 200:
                self.cache[key] = response.content
                return response.content
            else:
                print(f"Error fetching {url}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_title(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return soup.title.string

    def ask(self, prompt):
        key = ("ask", self.model, prompt)
        if key in self.cache:
            print("Cache hit for", key)
            return self.cache[key]

        if self.verbose:
            print("\n".join(textwrap.wrap(
                f"Prompt: {prompt}", subsequent_indent="    ", replace_whitespace=False)))
        self.query_count += 1
        result = self.chat([HumanMessage(content=prompt)]).content
        if self.verbose:
            print("\n".join(
                textwrap.wrap(f"Response: {result}",
                              subsequent_indent="    ", replace_whitespace=False)))

        self.cache[key] = result

        return result

    @staticmethod
    def background_text(background):
        return "\n\n".join(
            f"{subject}:\n{contents}" for subject, contents in background.items()
        )

    def token_count(self, text):
        return self.chat.get_num_tokens(text)

    def shorten(self, background):
        for subject in background:
            if not background[subject]:
                continue
            while self.token_count(background[subject]) > max_token_count[self.model]:
                background[subject] = background[subject][:int(len(background[subject]) * .9)]
        return background

    def summarize(self, background):
        # TODO: Use langchain
        for subject in background:
            if not background[subject]:
                continue
            # Leave 500 tokens for the prompt
            if (self.token_count(self.background_text(background)) <=
                    max_token_count[self.model] - 500):
                return background
            background[subject] = self.llm.ask(
                f"Concisely summarize the facts about {subject}:\n" + background[subject])

        return background

    def ddg_search(self, topic):
        key = ("ddg_search", topic)
        if key in self.cache:
            print("Cache hit for", key)
            return self.cache[key]

        if self.verbose:
            print("Search DDG for:", topic)
        result = ddg(topic)

        self.cache[key] = result

        return result

    def ddg_top_hit(self, topic, skip=[]):
        results = self.ddg_search(topic)
        for result in results:
            if result['href'] in skip:
                continue
            if self.verbose:
                print("  Fetching", result['href'])
            html = self.fetch(result['href'])
            if html:
                title = self.extract_title(html)
                content = simplify_html(html)
                if content:
                    return_value = (result['href'], str(title), content)
                    return return_value

    def main(self):
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

        self.chat = ChatOpenAI(model_name=self.model)
        self.llm = llmlib.Llm(llmlib.Openai(self.model), verbose=self.verbose)

        today_prompt = f"Today is {datetime.date.today().strftime('%A, %B %d, %Y')}."
        search_prompt = (f"{today_prompt}\n\n"
                        f"I want to know: {args.question}\n\n"
                        "What 3 search topics would help you answer this "
                        "question? Answer in a JSON list only.")
        search_text = self.llm.ask(search_prompt)
        searches = json.loads(search_text)
        background = {}
        sources = []
        for search in searches:
            source, title, content = self.ddg_top_hit(search,
                                                      skip=[source for source, _ in sources])
            background[search] = content
            sources.append((source, title))

        if args.verbose:
            pprint(("fetched:", {search : len(content) for search, content in background.items()}))
        background = self.summarize(background)
        if args.verbose:
            pprint(("summarized:",
                    {search : len(content) for search, content in background.items()}))
        print(
            self.llm.ask(f"{self.background_text(background)}\n\n{today_prompt}\n\n{args.question}")
        )
        print(f"({self.model}, {self.query_count} queries)")
        print()
        print("Sources:")
        for source, title in sources:
            print(f"* [{title}]({source})")

if __name__ == "__main__":
    gptSearch = GptSearch()
    sys.exit(gptSearch.main())
