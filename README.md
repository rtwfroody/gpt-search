# GPT Search

This program combines the power of GPT with DuckDuckGo search results to answer
questions. It fetches relevant information from the web, shortens and summarizes
it, and then uses GPT to generate a response to the user's question.

## Requirements

- Python 3
- OpenAI API Key
- BeautifulSoup
- requests
- tiktoken

## Installation

To install the required packages, run:

```bash
pip install duckduckgo-search openai beautifulsoup4 requests joblib tiktoken
```

## Usage

To run the program, execute the following command:

```bash
./gpt_search.py [options] <question>
```

Options:

- `--4` or `-4`: Use GPT-4 model (slower, costs more money).
- `--verbose` or `-v`: Enable verbose output.

Replace `<question>` with the question you want to ask.

## Example

```bash
./gpt_search.py --verbose "What is the process of photosynthesis?"
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key.

## Notes

Please be aware that this program can perform up to 5 GPT queries, which may have associated costs.

## Examples

### gpt-3.5-turbo

```
$ ./gpt_search.py "News about ChatGPT."
ChatGPT, the AI language model developed by OpenAI, has been making waves in the tech world with its advanced capabilities. As previously reported, ChatGPT released experimental AI plugins on March 23, 2023, which allowed the AI to access portions of the internet and provided added functionality for its users.

Recently, OpenAI announced its most advanced language model yet, GPT-4, which is currently only available in the ChatGPT Plus paid subscription and as an API for developers. Companies such as Duolingo, Be My Eyes, Stripe, and Khan Academy have already integrated GPT-4 into their applications and services.

One notable feature of GPT-4 is its ability to process up to 25,000 words of text from the user and receive images as a basis for interaction. It is also more advanced in creativity, visual input, and longer context, making it significantly safer to use than its predecessor.

However, not all experts are on board with the rapid development of GPT-4 and future models. Over 1,000 AI experts have signed an open letter calling on developers to slow down on the development of GPT-4.5 and GPT-5 due to potential large-scale risks.

Nonetheless, developers claim that GPT-5 will complete its training this year and could bring an AI revolution with it. As for ChatGPT, users can expect even more advancements and features to make their interactions with the AI even more seamless and beneficial.
(gpt-3.5-turbo)

Sources:
* [ChatGPT — Release Notes | OpenAI Help Center](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)
* [What happens when ChatGPT lies about real people? - The Washington Post](https://www.washingtonpost.com/technology/2023/04/05/chatgpt-lies/)
* [GPT-4: how to use, new features, availability, and more | Digital Trends](https://www.digitaltrends.com/computing/chatgpt-4-everything-we-know-so-far/)
```

### gpt-4

```
tnewsome@compy-linux:~/projects/gpt-search$ ./gpt_search.py -4 "News about ChatGPT."
Title: The Latest Developments and Applications in the ChatGPT Space

Recent news surrounding ChatGPT, OpenAI's Al-powered chatbot, reveals its growing presence and applications across various industries. With a rapidly expanding user base, numerous startups backed by Y Combinator and other investors are building on ChatGPT's capabilities, exploring multiple use cases and introducing innovative ideas, including:

1. Yuma: Focused on Shopify merchants, Yuma provides an AI-driven solution for customer support by integrating with help desk software. Utilizing language models akin to ChatGPT, it suggests appropriate replies based on users' historical data, streamlining the customer support process and reducing response time.

2. Baselit: Leaning on OpenAI's GPT-3, Baselit enables businesses to engage in chatbot-style analytics. The platform permits users to perform database queries using plain English, rendering the process more accessible and efficient. By connecting to various databases, users can obtain insights without the need for specialized coding skills.

3. Lasso: An interesting blend of RPA (Robotic Process Automation) and ChatGPT-like technology, Lasso allows users to automate processes using natural language. By simplifying the setup, it offers cost-effective solutions in comparison to well-established RPA solutions while accelerating the automation of various repetitive tasks.

4. BerriAI: Built for developers, BerriAI assists in creating ChatGPT apps for organization data through diverse data connectors. It enables organizations to spin up multiple instances and share prototypes with different configurations, offering benefits such as an enhanced employee experience and streamlined customer support.

While these startups encounter stiff competition from established companies, such as UiPath, Automation Anywhere, and Borealis AI, among others, their growth attests to the burgeoning interest in ChatGPT applications. As more enterprises invest in workflow automation tools and integrate them into their existing infrastructures, startups that leverage the power of ChatGPT and its enhanced language capabilities are poised for success.

In conclusion, the ChatGPT landscape continues to evolve and expand, driven by OpenAI's advancements and increased adoption by startups and businesses alike. From improved customer support and analytics to robotic automation, ChatGPT applications have the potential to transform multiple sectors, redefine user experience, and revolutionize several aspects of business operations.
(gpt-4)

Sources:
* [ChatGPT — Release Notes | OpenAI Help Center](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)
* [GPT-4: how to use, new features, availability, and more | Digital Trends](https://www.digitaltrends.com/computing/chatgpt-4-everything-we-know-so-far/)
* [These Y Combinator-backed startups are trying to build 'ChatGPT for X' | TechCrunch](https://techcrunch.com/2023/04/04/these-y-combinator-startups-are-trying-to-build-chatgpt-for-x/)
```