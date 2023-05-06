import aiohttp
from bs4 import BeautifulSoup


async def scrape_page(url: str) -> str:
    # Asynchronously send an HTTP request to the URL.
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise aiohttp.ClientError(f"An error occurred: {response.status}")
            html = await response.text()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # List of element IDs or class names to remove
    elements_to_remove = [
        "header",
        "footer",
        "sidebar",
        "nav",
        "menu",
        "ad",
        "advertisement",
        "cookie-banner",
        "popup",
        "social",
        "breadcrumb",
        "pagination",
        "comment",
        "comments",
    ]

    # Remove unwanted elements by ID or class name
    for element in elements_to_remove:
        for e in soup.find_all(id=element) + soup.find_all(class_=element):
            e.decompose()

    # Extract text from the remaining HTML tags
    text = " ".join(soup.stripped_strings)

    return text
