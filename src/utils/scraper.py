import logging

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from path import DATA_FOLDER, NOVELS_FOLDER

# Log INFO level messages in console
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

OK = 200  # Status code for a good HTTPS response


def extract_text_from_ptag(p_tag: Tag) -> str:
    """
    Extract text from the given <p>.

    Parameters
    ----------
    tag : Tag
        The <p> to extract text from

    Returns
    -------
    str
        The text of the given <p>, stripping whitespaces at both ends.
        In all other cases, an empty string is returned.
    """
    try:
        return p_tag.contents[0].strip()
    except (IndexError, AttributeError):
        return ""


def scrape_novel(
    author_name: str,
    novel_name: str,
    novel_link: str,
    first_page: int,
    last_page: int,
):
    """
    Scrape a novel from the given link.

    This function scrapes texts from a sequence of web pages displaying a novel,
    removes unwanted elements like website branding in the process,
    and saves results to a .txt file in an appropriate directory.

    Parameters
    ----------
    author_name : str
        Name of the novel's author, used for organizing the output directory structure
    novel_name : str
        Title of the novel, used for the output filename
    novel_link : str
        URL of the novel's page, should contain a page number that can be modified
    first_page : int
        The starting page number to scrape
    last_page : int
        The ending page number to scrape (inclusive)
    """
    pages = []
    for page_number in range(first_page, last_page + 1):
        # Update page number in the URL
        base_url, _ = novel_link.rsplit("_", 1)
        new_url = f"{base_url}_{page_number}.html"

        # Read the page in "utf-8", which should handle chinese characters
        response = requests.get(new_url)
        response.encoding = "utf-8"

        filename = f"{author_name}_{novel_name}.txt"
        if response.status_code != OK:
            LOGGER.error(f"Scrape failed for {filename}: HTTPS request failed on page {page_number}")
            return

        html = BeautifulSoup(response.text, "html.parser")
        article = html.find("article", class_="article-content")
        if not article:
            LOGGER.error(f"Scrape failed for {filename}: couldn't find article on page {page_number}")
            return

        # Lines are stored in p-tags
        lines = [extract_text_from_ptag(p_tag) for p_tag in article.find_all("p")]

        # Skip unwanted elements before chapter 1
        #   1. Title and author
        #   2. 文案 (short summary)
        #   3. 标签 (labels)
        #   4. search keywords
        #   5. editor comments
        if page_number == first_page:
            for i, line in enumerate(lines):
                if line.startswith("第1章") or line.startswith("第一章"):
                    lines = lines[i:]
                    break
            else:
                LOGGER.error(f"Scrape failed for {filename}: couldn't find chapter 1")
                return

        # Also skip the last <p> (website branding)
        lines.pop()

        # Join lines into a page
        page = "\n".join(lines)
        pages.append(page)
    else:
        novel_path = NOVELS_FOLDER / filename
        novel_text = "\n".join(pages)
        novel_size = len(novel_text)

        with open(novel_path, mode="w", encoding="utf-8") as f:
            f.write(novel_text)

        LOGGER.info(f"Success: {novel_size} characters has been scraped into {filename}")


def main():
    metadata = pd.read_csv(DATA_FOLDER / "metadata.csv")
    for _, row in metadata.iterrows():
        scrape_novel(**row)


if __name__ == "__main__":
    main()
