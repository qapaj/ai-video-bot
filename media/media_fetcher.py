"""
media/media_fetcher.py
Downloads relevant images and videos from free sources:
  Pexels, Pixabay, Wikimedia Commons
Falls back to generated gradient cards if APIs fail.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests

from utils.helpers import download_file, file_ok
from utils.logger import get_logger

log = get_logger("media.fetcher")

PEXELS_KEY  = os.environ.get("PEXELS_API_KEY", "")
PIXABAY_KEY = os.environ.get("PIXABAY_API_KEY", "")
MAX_VIDEOS  = 5
MAX_IMAGES  = 15
TIMEOUT     = 25

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Arabic → English keyword map for better API results
KEYWORD_MAP: Dict[str, List[str]] = {
    "الذكاء الاصطناعي": ["artificial intelligence", "AI technology", "machine learning"],
    "الفضاء":           ["space", "galaxy", "cosmos"],
    "التكنولوجيا":      ["technology", "innovation", "digital"],
    "الطبيعة":          ["nature", "landscape", "environment"],
    "المدينة":          ["city", "urban", "architecture"],
    "العلوم":           ["science", "research", "laboratory"],
    "الصحة":            ["health", "medicine", "wellness"],
    "التاريخ":          ["history", "ancient", "civilization"],
    "الاقتصاد":         ["economy", "business", "finance"],
    "التعليم":          ["education", "learning", "school"],
}


class MediaFetcher:
    """
    Fetches relevant media assets for a given topic.
    """

    def fetch(self, topic: str, output_dir: str,
              language: str = "ar") -> dict:
        """
        Download images and videos related to topic.
        Returns {"videos": [...], "images": [...], "total": int}
        """
        os.makedirs(output_dir, exist_ok=True)
        keywords = self._keywords(topic, language)
        kw1      = keywords[0]
        kw2      = keywords[1] if len(keywords) > 1 else kw1

        log.info("Fetching media: %s → %s", topic, keywords[:2])

        # Collect URLs
        all_items: List[dict] = []
        all_items += self._pexels_videos(kw1, 3)
        all_items += self._pixabay_videos(kw1, 2)
        all_items += self._pexels_photos(kw1, 8)
        all_items += self._pexels_photos(kw2, 4)
        all_items += self._pixabay_images(kw1, 8)
        all_items += self._wikimedia_images(kw1, 4)

        # Deduplicate
        seen, unique = set(), []
        for item in all_items:
            if item["url"] not in seen:
                seen.add(item["url"])
                unique.append(item)

        videos = [m for m in unique if m["type"] == "video"][:MAX_VIDEOS]
        images = [m for m in unique if m["type"] == "image"][:MAX_IMAGES]
        log.info("URLs: %d videos, %d images", len(videos), len(images))

        # Download in parallel
        dl_vids, dl_imgs = [], []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {}
            for i, item in enumerate(videos):
                futures[ex.submit(self._download, item, output_dir, i)] = "v"
            for i, item in enumerate(images):
                futures[ex.submit(self._download, item, output_dir, i + 100)] = "i"
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    (dl_vids if res["type"] == "video" else dl_imgs).append(res)

        log.info("Downloaded: %d videos, %d images", len(dl_vids), len(dl_imgs))
        return {
            "videos": dl_vids,
            "images": dl_imgs,
            "total":  len(dl_vids) + len(dl_imgs),
        }

    # ── Sources ────────────────────────────────────────────────────────────────

    def _pexels_videos(self, q: str, n: int) -> List[dict]:
        if not PEXELS_KEY: return []
        try:
            r = requests.get("https://api.pexels.com/videos/search",
                headers={"Authorization": PEXELS_KEY},
                params={"query": q, "per_page": n, "orientation": "portrait"},
                timeout=15)
            items = []
            for v in r.json().get("videos", []):
                for f in sorted(v.get("video_files", []),
                                key=lambda x: x.get("height", 0), reverse=True):
                    if f.get("quality") in ("hd", "sd"):
                        items.append({"url": f["link"], "type": "video", "source": "pexels"})
                        break
            return items
        except Exception: return []

    def _pixabay_videos(self, q: str, n: int) -> List[dict]:
        if not PIXABAY_KEY: return []
        try:
            r = requests.get("https://pixabay.com/api/videos/",
                params={"key": PIXABAY_KEY, "q": q, "per_page": n},
                timeout=15)
            items = []
            for v in r.json().get("hits", []):
                vids = v.get("videos", {})
                for qual in ("large", "medium", "small"):
                    if qual in vids:
                        items.append({"url": vids[qual]["url"], "type": "video",
                                      "source": "pixabay"})
                        break
            return items
        except Exception: return []

    def _pexels_photos(self, q: str, n: int) -> List[dict]:
        if not PEXELS_KEY: return []
        try:
            r = requests.get("https://api.pexels.com/v1/search",
                headers={"Authorization": PEXELS_KEY},
                params={"query": q, "per_page": n, "orientation": "portrait"},
                timeout=15)
            items = []
            for p in r.json().get("photos", []):
                url = p.get("src", {}).get("portrait") or p.get("src", {}).get("large")
                if url: items.append({"url": url, "type": "image", "source": "pexels"})
            return items
        except Exception: return []

    def _pixabay_images(self, q: str, n: int) -> List[dict]:
        if not PIXABAY_KEY: return []
        try:
            r = requests.get("https://pixabay.com/api/",
                params={"key": PIXABAY_KEY, "q": q, "per_page": n,
                        "image_type": "photo", "orientation": "vertical",
                        "safesearch": "true"},
                timeout=15)
            items = []
            for img in r.json().get("hits", []):
                url = img.get("largeImageURL") or img.get("webformatURL")
                if url: items.append({"url": url, "type": "image", "source": "pixabay"})
            return items
        except Exception: return []

    def _wikimedia_images(self, q: str, n: int) -> List[dict]:
        try:
            r = requests.get("https://commons.wikimedia.org/w/api.php",
                params={"action": "query", "list": "search",
                        "srsearch": f"{q} filetype:bitmap",
                        "srnamespace": 6, "srlimit": n, "format": "json"},
                headers={"User-Agent": UA}, timeout=15)
            items = []
            for res in r.json().get("query", {}).get("search", []):
                ir = requests.get("https://commons.wikimedia.org/w/api.php",
                    params={"action": "query", "titles": res["title"],
                            "prop": "imageinfo", "iiprop": "url",
                            "iiurlwidth": 1200, "format": "json"},
                    headers={"User-Agent": UA}, timeout=10)
                for page in ir.json().get("query", {}).get("pages", {}).values():
                    info = page.get("imageinfo", [{}])[0]
                    url  = info.get("thumburl") or info.get("url", "")
                    if url and url.lower().endswith((".jpg", ".jpeg", ".png")):
                        items.append({"url": url, "type": "image", "source": "wikimedia"})
                        break
            return items
        except Exception: return []

    # ── Download ───────────────────────────────────────────────────────────────

    def _download(self, item: dict, output_dir: str, index: int) -> Optional[dict]:
        url  = item["url"]
        ext  = ".mp4" if item["type"] == "video" else ".jpg"
        for valid in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov"):
            if url.split("?")[0].lower().endswith(valid):
                ext = valid; break
        path = os.path.join(output_dir,
                            f"{item['type']}_{index:03d}_{item['source']}{ext}")
        if file_ok(path):
            return {**item, "local_path": path}

        headers = {"User-Agent": UA}
        if download_file(url, path, timeout=TIMEOUT, headers=headers):
            log.debug("✓ %s (%dKB)", os.path.basename(path),
                      os.path.getsize(path) // 1024)
            return {**item, "local_path": path}
        return None

    # ── Keyword translation ────────────────────────────────────────────────────

    def _keywords(self, topic: str, language: str) -> List[str]:
        for ar_key, en_list in KEYWORD_MAP.items():
            if ar_key in topic:
                return en_list
        return [topic, topic + " background", topic + " concept"]
