"""
Game Knowledge Search Module

Provides web search capabilities for looking up game-specific information
during video analysis. Enables the model to understand game mechanics,
characters, items, and lore by searching the internet.

Features:
- DuckDuckGo search (free, no API key needed)
- Gaming wiki integration (Fandom, IGN, GameFAQs)
- Result caching to avoid redundant searches
- Game context detection from video content
- Tool interface for LLM function calling
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    snippet: str
    source: str  # e.g., "fandom", "ign", "wikipedia", "general"
    relevance_score: float = 1.0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "relevance_score": self.relevance_score,
        }

    def format_for_context(self) -> str:
        """Format for inclusion in LLM context."""
        return f"[{self.source.upper()}] {self.title}\n{self.snippet}"


@dataclass
class GameContext:
    """Detected game context from video analysis."""
    game_name: Optional[str] = None
    game_genre: Optional[str] = None
    detected_characters: list[str] = field(default_factory=list)
    detected_items: list[str] = field(default_factory=list)
    detected_locations: list[str] = field(default_factory=list)
    detected_mechanics: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "game_name": self.game_name,
            "game_genre": self.game_genre,
            "detected_characters": self.detected_characters,
            "detected_items": self.detected_items,
            "detected_locations": self.detected_locations,
            "detected_mechanics": self.detected_mechanics,
            "confidence": self.confidence,
        }


class SearchCache:
    """Caches search results to avoid redundant API calls."""

    def __init__(self, cache_dir: str = "data/search_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self._memory_cache: dict[str, tuple[float, list[SearchResult]]] = {}

    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[list[SearchResult]]:
        """Get cached results if available and not expired."""
        cache_key = self._get_cache_key(query)

        # Check memory cache first
        if cache_key in self._memory_cache:
            timestamp, results = self._memory_cache[cache_key]
            if time.time() - timestamp < self.ttl_seconds:
                return results

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                if time.time() - data["timestamp"] < self.ttl_seconds:
                    results = [SearchResult(**r) for r in data["results"]]
                    self._memory_cache[cache_key] = (data["timestamp"], results)
                    return results
            except Exception:
                pass

        return None

    def set(self, query: str, results: list[SearchResult]) -> None:
        """Cache search results."""
        cache_key = self._get_cache_key(query)
        timestamp = time.time()

        # Memory cache
        self._memory_cache[cache_key] = (timestamp, results)

        # Disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "query": query,
                    "timestamp": timestamp,
                    "results": [r.to_dict() for r in results],
                }, f)
        except Exception as e:
            logger.warning(f"Failed to cache search results: {e}")


class GameKnowledgeSearcher:
    """
    Web search tool for game knowledge lookup.

    Provides the model with ability to search for:
    - Game mechanics and systems
    - Character information and lore
    - Item/weapon/ability details
    - Location and map information
    - Game-specific terminology

    Example:
        >>> searcher = GameKnowledgeSearcher()
        >>> results = searcher.search("Elden Ring Margit the Fell Omen boss guide")
        >>> for r in results:
        ...     print(f"{r.title}: {r.snippet[:100]}...")

        >>> # With game context
        >>> searcher.set_game_context("Dark Souls 3")
        >>> results = searcher.search("Pontiff Sulyvahn")  # Auto-adds game context
    """

    # Gaming-focused domains for prioritized results
    GAMING_DOMAINS = [
        "fandom.com",
        "fextralife.com",
        "ign.com",
        "gamespot.com",
        "polygon.com",
        "kotaku.com",
        "pcgamer.com",
        "eurogamer.net",
        "gamefaqs.gamespot.com",
        "reddit.com/r/",
        "steampowered.com",
        "playstation.com",
        "xbox.com",
        "nintendo.com",
    ]

    def __init__(
        self,
        cache_dir: str = "data/search_cache",
        max_results: int = 5,
        timeout: int = 10,
    ):
        self.cache = SearchCache(cache_dir)
        self.max_results = max_results
        self.timeout = timeout
        self.game_context: Optional[GameContext] = None

        # Track search history for context
        self._search_history: list[str] = []

        logger.info("GameKnowledgeSearcher initialized")

    def set_game_context(self, game_name: str, genre: Optional[str] = None) -> None:
        """Set the current game context for more relevant searches."""
        self.game_context = GameContext(
            game_name=game_name,
            game_genre=genre,
            confidence=1.0,
        )
        logger.info(f"Game context set: {game_name}")

    def clear_game_context(self) -> None:
        """Clear the current game context."""
        self.game_context = None

    def search(
        self,
        query: str,
        include_game_context: bool = True,
        search_type: str = "general",
    ) -> list[SearchResult]:
        """
        Search for game-related information.

        Args:
            query: Search query
            include_game_context: Whether to prepend game name to query
            search_type: Type of search - "general", "wiki", "guide", "lore"

        Returns:
            List of SearchResult objects
        """
        # Build full query with game context
        full_query = query
        if include_game_context and self.game_context and self.game_context.game_name:
            if self.game_context.game_name.lower() not in query.lower():
                full_query = f"{self.game_context.game_name} {query}"

        # Add search type modifiers
        if search_type == "wiki":
            full_query = f"{full_query} wiki"
        elif search_type == "guide":
            full_query = f"{full_query} guide walkthrough"
        elif search_type == "lore":
            full_query = f"{full_query} lore story explained"

        # Check cache
        cached = self.cache.get(full_query)
        if cached:
            logger.info(f"Cache hit for: {full_query}")
            return cached

        # Perform search
        results = self._perform_search(full_query)

        # Cache results
        if results:
            self.cache.set(full_query, results)
            self._search_history.append(full_query)

        return results

    def _perform_search(self, query: str) -> list[SearchResult]:
        """Perform the actual web search using DuckDuckGo."""
        try:
            # Try DuckDuckGo search
            return self._search_duckduckgo(query)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            try:
                # Fallback to alternative method
                return self._search_duckduckgo_html(query)
            except Exception as e2:
                logger.error(f"All search methods failed: {e2}")
                return []

    def _search_duckduckgo(self, query: str) -> list[SearchResult]:
        """Search using duckduckgo-search library."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search not installed. Install with: pip install duckduckgo-search")
            raise

        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(
                query,
                max_results=self.max_results * 2,  # Get extra to filter
                safesearch="off",
            ))

        for item in search_results[:self.max_results * 2]:
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("href", ""),
                snippet=item.get("body", ""),
                source=self._identify_source(item.get("href", "")),
            )
            # Boost gaming domain results
            if any(domain in result.url for domain in self.GAMING_DOMAINS):
                result.relevance_score = 1.5
            results.append(result)

        # Sort by relevance and return top results
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:self.max_results]

    def _search_duckduckgo_html(self, query: str) -> list[SearchResult]:
        """Fallback: Search DuckDuckGo via HTML scraping."""
        import urllib.request
        from html.parser import HTMLParser

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current_result = {}
                self.in_result = False
                self.in_title = False
                self.in_snippet = False

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                    self.in_result = True
                    self.in_title = True
                    self.current_result = {"url": attrs_dict.get("href", "")}
                elif tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                    self.in_snippet = True

            def handle_endtag(self, tag):
                if tag == "a" and self.in_title:
                    self.in_title = False
                elif tag == "a" and self.in_snippet:
                    self.in_snippet = False
                    if self.current_result:
                        self.results.append(self.current_result)
                        self.current_result = {}

            def handle_data(self, data):
                if self.in_title:
                    self.current_result["title"] = data.strip()
                elif self.in_snippet:
                    self.current_result["snippet"] = data.strip()

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; GameplayVisionLLM/1.0)"}

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            html = response.read().decode("utf-8")

        parser = DDGParser()
        parser.feed(html)

        results = []
        for item in parser.results[:self.max_results]:
            if item.get("title") and item.get("url"):
                results.append(SearchResult(
                    title=item["title"],
                    url=item["url"],
                    snippet=item.get("snippet", ""),
                    source=self._identify_source(item["url"]),
                ))

        return results

    def _identify_source(self, url: str) -> str:
        """Identify the source type from URL."""
        url_lower = url.lower()

        if "fandom.com" in url_lower or "wikia.com" in url_lower:
            return "fandom"
        elif "fextralife.com" in url_lower:
            return "fextralife"
        elif "wikipedia.org" in url_lower:
            return "wikipedia"
        elif "ign.com" in url_lower:
            return "ign"
        elif "gamespot.com" in url_lower or "gamefaqs" in url_lower:
            return "gamespot"
        elif "reddit.com" in url_lower:
            return "reddit"
        elif "youtube.com" in url_lower:
            return "youtube"
        elif "steam" in url_lower:
            return "steam"
        else:
            return "web"

    def search_character(self, character_name: str) -> list[SearchResult]:
        """Search for character information."""
        return self.search(f"{character_name} character", search_type="wiki")

    def search_item(self, item_name: str) -> list[SearchResult]:
        """Search for item/weapon/equipment information."""
        return self.search(f"{item_name} item stats location", search_type="wiki")

    def search_boss(self, boss_name: str) -> list[SearchResult]:
        """Search for boss fight information."""
        return self.search(f"{boss_name} boss fight guide strategy", search_type="guide")

    def search_mechanic(self, mechanic: str) -> list[SearchResult]:
        """Search for game mechanic explanation."""
        return self.search(f"{mechanic} how it works explained", search_type="guide")

    def search_location(self, location: str) -> list[SearchResult]:
        """Search for location/area information."""
        return self.search(f"{location} area map guide", search_type="wiki")

    def search_lore(self, topic: str) -> list[SearchResult]:
        """Search for lore and story information."""
        return self.search(topic, search_type="lore")

    def format_results_for_llm(
        self,
        results: list[SearchResult],
        max_chars: int = 2000,
    ) -> str:
        """Format search results for inclusion in LLM prompt."""
        if not results:
            return "No search results found."

        lines = ["## Web Search Results\n"]
        char_count = 0

        for i, result in enumerate(results, 1):
            entry = f"{i}. [{result.source.upper()}] **{result.title}**\n   {result.snippet}\n"

            if char_count + len(entry) > max_chars:
                lines.append("... (more results truncated)")
                break

            lines.append(entry)
            char_count += len(entry)

        return "\n".join(lines)

    def get_tool_definition(self) -> dict:
        """
        Get the tool definition for LLM function calling.

        Returns a schema that can be used with OpenAI-style function calling
        or similar interfaces.
        """
        return {
            "type": "function",
            "function": {
                "name": "search_game_knowledge",
                "description": "Search the internet for game-specific information like characters, items, mechanics, boss strategies, or lore. Use this when you need to understand something about the game being played.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. Be specific about what you're looking for.",
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["general", "wiki", "guide", "lore"],
                            "description": "Type of search: 'wiki' for factual info, 'guide' for strategies, 'lore' for story/background.",
                            "default": "general",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute_tool_call(self, query: str, search_type: str = "general") -> str:
        """
        Execute a tool call from the LLM.

        This is the interface for when the model decides to search.

        Returns:
            Formatted search results as a string
        """
        logger.info(f"Tool call: search_game_knowledge(query='{query}', type='{search_type}')")
        results = self.search(query, search_type=search_type)
        return self.format_results_for_llm(results)


class GameDetector:
    """
    Detects which game is being played from video content.

    Uses OCR text, visual elements, and audio cues to identify the game.
    """

    # Common game title patterns in UI/menus
    GAME_PATTERNS = [
        # Soulslike games
        (r"elden\s*ring", "Elden Ring", "action-rpg"),
        (r"dark\s*souls\s*(iii|3)", "Dark Souls 3", "action-rpg"),
        (r"dark\s*souls\s*(ii|2)", "Dark Souls 2", "action-rpg"),
        (r"dark\s*souls", "Dark Souls", "action-rpg"),
        (r"bloodborne", "Bloodborne", "action-rpg"),
        (r"sekiro", "Sekiro: Shadows Die Twice", "action-rpg"),

        # Popular games
        (r"minecraft", "Minecraft", "sandbox"),
        (r"fortnite", "Fortnite", "battle-royale"),
        (r"valorant", "Valorant", "fps"),
        (r"league\s*of\s*legends|lol", "League of Legends", "moba"),
        (r"counter[\-\s]?strike|cs:?go|cs2", "Counter-Strike", "fps"),
        (r"overwatch", "Overwatch", "fps"),
        (r"apex\s*legends", "Apex Legends", "battle-royale"),
        (r"call\s*of\s*duty|cod|warzone", "Call of Duty", "fps"),
        (r"gta|grand\s*theft\s*auto", "Grand Theft Auto", "action"),
        (r"zelda|breath\s*of\s*the\s*wild|tears\s*of\s*the\s*kingdom", "The Legend of Zelda", "action-adventure"),
        (r"pokemon|pok[eé]mon", "Pokemon", "rpg"),
        (r"fifa|ea\s*sports\s*fc", "EA Sports FC", "sports"),
        (r"rocket\s*league", "Rocket League", "sports"),
        (r"cyberpunk\s*2077", "Cyberpunk 2077", "action-rpg"),
        (r"the\s*witcher\s*3|witcher\s*3", "The Witcher 3", "action-rpg"),
        (r"baldur'?s\s*gate\s*3|bg3", "Baldur's Gate 3", "rpg"),
        (r"diablo\s*(iv|4)", "Diablo 4", "action-rpg"),
        (r"world\s*of\s*warcraft|wow", "World of Warcraft", "mmorpg"),
        (r"final\s*fantasy\s*(xiv|14)", "Final Fantasy XIV", "mmorpg"),
        (r"destiny\s*2", "Destiny 2", "fps"),
        (r"hades\s*(ii|2)?", "Hades", "roguelike"),
        (r"hollow\s*knight", "Hollow Knight", "metroidvania"),
        (r"celeste", "Celeste", "platformer"),
        (r"stardew\s*valley", "Stardew Valley", "simulation"),
        (r"terraria", "Terraria", "sandbox"),
        (r"among\s*us", "Among Us", "social-deduction"),
    ]

    def __init__(self):
        self.detected_context: Optional[GameContext] = None
        self._text_evidence: list[str] = []

    def detect_from_ocr(self, ocr_results: list[dict]) -> Optional[GameContext]:
        """
        Detect game from OCR text extracted from video.

        Args:
            ocr_results: List of OCR detections with 'text' field

        Returns:
            GameContext if detected, None otherwise
        """
        all_text = " ".join(
            r.get("text", "") for r in ocr_results
        ).lower()

        for pattern, game_name, genre in self.GAME_PATTERNS:
            if re.search(pattern, all_text, re.IGNORECASE):
                self.detected_context = GameContext(
                    game_name=game_name,
                    game_genre=genre,
                    confidence=0.8,
                )
                logger.info(f"Detected game from OCR: {game_name}")
                return self.detected_context

        return None

    def detect_from_speech(self, speech_results: list[dict]) -> Optional[GameContext]:
        """
        Detect game from speech transcription.

        Args:
            speech_results: List of speech segments with 'text' field

        Returns:
            GameContext if detected, None otherwise
        """
        all_text = " ".join(
            r.get("text", "") for r in speech_results
        ).lower()

        for pattern, game_name, genre in self.GAME_PATTERNS:
            if re.search(pattern, all_text, re.IGNORECASE):
                self.detected_context = GameContext(
                    game_name=game_name,
                    game_genre=genre,
                    confidence=0.6,  # Lower confidence from speech
                )
                logger.info(f"Detected game from speech: {game_name}")
                return self.detected_context

        return None

    def add_text_evidence(self, text: str) -> None:
        """Add text evidence for game detection."""
        self._text_evidence.append(text.lower())

    def get_best_guess(self) -> Optional[GameContext]:
        """Get the best game guess from all collected evidence."""
        if self.detected_context:
            return self.detected_context

        all_evidence = " ".join(self._text_evidence)

        for pattern, game_name, genre in self.GAME_PATTERNS:
            if re.search(pattern, all_evidence, re.IGNORECASE):
                self.detected_context = GameContext(
                    game_name=game_name,
                    game_genre=genre,
                    confidence=0.5,
                )
                return self.detected_context

        return None


# =============================================================================
# Integration with Reasoning Core
# =============================================================================

def create_search_enhanced_system_prompt(
    base_prompt: str,
    game_context: Optional[GameContext] = None,
) -> str:
    """
    Enhance the system prompt with search capabilities description.

    Args:
        base_prompt: The original system prompt
        game_context: Optional detected game context

    Returns:
        Enhanced system prompt
    """
    search_instructions = """

## Web Search Capability

You have access to web search to look up game-specific information. When analyzing gameplay:

1. If you encounter an unfamiliar character, enemy, item, or mechanic, you can search for it
2. Use search to provide accurate information about boss strategies, item locations, or game lore
3. When the user asks about specific game elements, search to give detailed answers

To search, include a search request in your reasoning like:
[SEARCH: "query here"]

The system will perform the search and provide results for your analysis.
"""

    game_info = ""
    if game_context and game_context.game_name:
        game_info = f"\n\n## Current Game Context\nThe video appears to be from: **{game_context.game_name}**"
        if game_context.game_genre:
            game_info += f" (Genre: {game_context.game_genre})"

    return base_prompt + search_instructions + game_info
