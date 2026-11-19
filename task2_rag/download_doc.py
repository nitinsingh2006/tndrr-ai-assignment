"""
Download the source PDF document for the RAG pipeline.
Downloads: "A Comprehensive Overview of Large Language Models" (arXiv 2307.06435)
"""
import urllib.request
import os
import sys


def download_pdf(url: str, dest: str) -> bool:
    """Download a PDF from a URL to a local destination path.

    Args:
        url: Source URL of the PDF.
        dest: Local file path to save the PDF.

    Returns:
        True if download succeeded, False otherwise.
    """
    os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else ".", exist_ok=True)

    if os.path.exists(dest):
        size = os.path.getsize(dest) / 1e6
        print(f"Already exists: {dest} ({size:.1f} MB)")
        return True

    print(f"Downloading {url} ...")
    try:
        # Set a user-agent to avoid 403 on some servers
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-Pipeline/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=120) as response, open(dest, "wb") as out:
            out.write(response.read())

        size = os.path.getsize(dest) / 1e6
        print(f"Downloaded {size:.1f} MB to {dest}")
        return True

    except Exception as exc:
        print(f"ERROR downloading {url}: {exc}", file=sys.stderr)
        # Clean up partial file
        if os.path.exists(dest):
            os.remove(dest)
        return False


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    url = "https://arxiv.org/pdf/2307.06435"
    dest = "data/survey_llm.pdf"

    success = download_pdf(url, dest)
    if not success:
        # Fallback: try the HTML version redirected PDF
        fallback_url = "https://arxiv.org/pdf/2307.06435v2"
        print(f"Trying fallback URL: {fallback_url}")
        success = download_pdf(fallback_url, dest)

    if success:
        print("PDF ready for processing.")
    else:
        print("Download failed. Place a PDF manually at data/survey_llm.pdf")
        sys.exit(1)
