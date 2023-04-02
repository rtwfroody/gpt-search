#!/bin/env python3

from bs4 import BeautifulSoup
from duckduckgo_search import ddg
from joblib import Memory
from pprint import pprint
import argparse
import datetime
import json
import os
import requests
import sys
import tiktoken
import textwrap
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

cachedir = os.path.join(os.path.dirname(__file__), ".cache")
memory = Memory(cachedir, verbose=0)

max_token_count = {
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4097
}

@memory.cache
def fetch_url_and_extract_info(url):
    try:
        # Fetch the URL
        response = requests.get(url, timeout=10)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract the text from the HTML
            text = ' '.join(soup.stripped_strings)
            title = soup.title.string

            return (title, text)
        else:
            print(f"Error fetching {url}: {response.status_code}")
            return (None, None)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return (None, None)

class GptSearch(object):
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.verbose = False

        self.query_count = 0

        self.ask = memory.cache(self.ask)
        self.ddg_top_hit = memory.cache(self.ddg_top_hit)

    def ask(self, prompt):
        if self.verbose:
            print("\n".join(textwrap.wrap(f"Prompt: {prompt}", subsequent_indent="    ", replace_whitespace=False)))
        self.query_count += 1
        result = self.chat([HumanMessage(content=prompt)]).content
        if self.verbose:
            print("\n".join(textwrap.wrap(f"Response: {result}", subsequent_indent="    ", replace_whitespace=False)))
        return result

    @staticmethod
    def background_text(background):
        return "\n\n".join(
            f"{subject}:\n{contents}" for subject, contents in background.items()
        )

    def token_count(self, text):
        # TODO: Use langchain
        encoding = tiktoken.encoding_for_model(self.model)
        # tiktoken seems to be undercounting tokens compared to the API
        return len(encoding.encode(text)) * 2

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
            if self.token_count(self.background_text(background)) <= max_token_count[self.model]:
                return background
            background[subject] = self.ask(
                f"Concisely summarize the facts about {subject}:\n" + background[subject])

        return background

    def ddg_top_hit(self, topic):
        if self.verbose:
            print("Search DDG for:", topic)
        results = ddg(topic)
        for result in results:
            if self.verbose:
                print("  Fetching", result['href'])
            (title, content) = fetch_url_and_extract_info(result['href'])
            if content:
                return (result['href'], title, content)

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

        today_prompt = f"Today is {datetime.date.today().strftime('%A, %B %d, %Y')}."
        search_prompt = (f"{today_prompt}\n\n"
                        f"I want to know: {args.question}\n\n"
                        "What 3 search topics would help you answer this "
                        "question? Answer in a JSON list only.")
        search_text = self.ask(search_prompt)
        searches = json.loads(search_text)
        background = {}
        sources = []
        for search in searches:
            source, title, content = self.ddg_top_hit(search)
            background[search] = content
            sources.append((source, title))

        if args.verbose:
            pprint(("fetched:", {search : len(content) for search, content in background.items()}))
        background = self.shorten(background)
        if args.verbose:
            pprint(("shortened:", {search : len(content) for search, content in background.items()}))
        background = self.summarize(background)
        if args.verbose:
            pprint(("summarized:", {search : len(content) for search, content in background.items()}))
        print(
            self.ask(f"{self.background_text(background)}\n\n{today_prompt}\n\n{args.question}")
        )
        print(f"({self.model}, {self.query_count} queries)")
        print()
        print("Sources:")
        for source, title in sources:
            print(f"* [{title}]({source})")

if __name__ == "__main__":
    gptSearch = GptSearch()
    sys.exit(gptSearch.main())
