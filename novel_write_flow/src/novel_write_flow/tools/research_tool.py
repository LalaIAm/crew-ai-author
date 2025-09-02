from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json
import time
from urllib.parse import urlparse

from crewai.tools import BaseTool, ScrapeWebsiteTool, SerperApiTool
from pydantic import BaseModel, Field, HttpUrl, validator


class ResearchType(str, Enum):
    """Enumeration of supported research types."""
    GENERAL = "general"
    GENRE_ANALYSIS = "genre_analysis"
    MARKET_RESEARCH = "market_research"
    COMPETITOR_RESEARCH = "competitor_research"
    TREND_ANALYSIS = "trend_analysis"
    DATA_COLLECTION = "data_collection"


class SearchEngine(str, Enum):
    """Enumeration of supported search engines."""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"


class ResearchToolInput(BaseModel):
    """
    Input schema for ResearchTool.

    This tool supports flexible research queries across multiple dimensions including
    topic analysis, market research, and competitive intelligence gathering.
    """

    query: str = Field(
        ...,
        description="Main research query or topic to investigate"
    )

    research_type: ResearchType = Field(
        ResearchType.GENERAL,
        description="Type of research to perform (general, genre_analysis, market_research, competitor_research, trend_analysis, data_collection)"
    )

    sources_limit: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of sources to collect (1-50)"
    )

    include_insights: bool = Field(
        True,
        description="Whether to include AI-generated insights in the output"
    )

    include_trends: bool = Field(
        True,
        description="Whether to analyze and include trend information"
    )

    include_citations: bool = Field(
        True,
        description="Whether to include source citations in the output"
    )

    search_engine: SearchEngine = Field(
        SearchEngine.GOOGLE,
        description="Preferred search engine to use"
    )

    urls: Optional[List[HttpUrl]] = Field(
        None,
        description="Specific URLs to scrape for additional data"
    )

    time_range: Optional[str] = Field(
        None,
        description="Time range for searches (e.g., 'past_week', 'past_month', 'past_year')"
    )

    domain_filter: Optional[str] = Field(
        None,
        description="Restrict search results to specific domain (e.g., 'example.com')"
    )

    keywords: Optional[List[str]] = Field(
        None,
        description="Additional keywords to enhance search results"
    )

    @validator('urls', pre=True, each_item=True)
    def validate_urls(cls, v):
        """Convert string URLs to HttpUrl objects."""
        if isinstance(v, str):
            from pydantic import parse_obj_as
            return parse_obj_as(HttpUrl, v)
        return v


class ResearchResult(BaseModel):
    """Structured output for research results."""

    query: str
    research_type: ResearchType
    search_results: List[Dict[str, Any]]
    scraped_content: Optional[List[Dict[str, Any]]] = None
    insights: Optional[List[str]] = None
    trends: Optional[List[Dict[str, Any]]] = None
    citations: Optional[List[str]] = None
    competitors: Optional[List[Dict[str, Any]]] = None
    market_data: Optional[Dict[str, Any]] = None


class ResearchTool(BaseTool):
    """
    Advanced Research Tool for comprehensive data collection and analysis.

    This tool provides a versatile research interface that combines web scraping,
    search capabilities, and analytical processing to gather insights across multiple
    domains. It's designed to support various research types including genre analysis,
    market research, competitor analysis, and trend identification.

    The tool supports multiple search engines (Google, Bing, DuckDuckGo) and
    integrates with CrewAI's built-in scraping capabilities for comprehensive
    information collection.

    Features:
    - Multi-engine search integration
    - Web scraping capabilities
    - Trend analysis and pattern recognition
    - Competitor intelligence gathering
    - Market research data collection
    - Structured output with citations
    - Rate limiting and error handling
    - Support for genre-specific analysis

    Usage Examples:

    # Basic research query
    tool_input = ResearchToolInput(
        query="modern science fiction tropes",
        research_type=ResearchType.GENRE_ANALYSIS
    )

    # Market research with specific keywords
    tool_input = ResearchToolInput(
        query="ebook publishing trends",
        research_type=ResearchType.MARKET_RESEARCH,
        keywords=["self-publishing", "AI writing", "market share"],
        time_range="past_year"
    )

    # Competitor analysis
    tool_input = ResearchToolInput(
        query="fiction writing software",
        research_type=ResearchType.COMPETITOR_RESEARCH,
        sources_limit=20
    )

    # Trend analysis with specific domains
    tool_input = ResearchTypeInput(
        query="emerging genres in literature",
        research_type=ResearchType.TREND_ANALYSIS,
        domain_filter=".edu"
    )

    Attributes:
        name: Name of the tool
        description: Clear description for agent usage
        args_schema: Pydantic input model schema
        max_retries: Maximum number of retry attempts for failed requests
        retry_delay: Delay between retry attempts in seconds
        rate_limit_delay: Minimum delay between API calls to prevent rate limiting
    """

    name: str = "Advanced Research Tool"
    description: str = (
        "A comprehensive research tool that combines web search, scraping, "
        "and analytical capabilities to provide structured insights, trends, "
        "and data across multiple research domains. Supports genre analysis, "
        "market research, competitor intelligence, and trend identification."
    )
    args_schema: Any = ResearchToolInput

    # Configuration options
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 1.0

    def _run(
        self,
        query: str,
        research_type: ResearchType = ResearchType.GENERAL,
        sources_limit: int = 10,
        include_insights: bool = True,
        include_trends: bool = True,
        include_citations: bool = True,
        search_engine: SearchEngine = SearchEngine.GOOGLE,
        urls: Optional[List[HttpUrl]] = None,
        time_range: Optional[str] = None,
        domain_filter: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> str:
        """
        Execute research query with comprehensive analysis.

        Args:
            query: Main research query or topic
            research_type: Type of research to perform
            sources_limit: Maximum number of sources to collect
            include_insights: Whether to include AI-generated insights
            include_trends: Whether to include trend analysis
            include_citations: Whether to include source citations
            search_engine: Preferred search engine
            urls: Specific URLs to scrape
            time_range: Time range for searches
            domain_filter: Domain filter for searches
            keywords: Additional keywords for search enhancement

        Returns:
            JSON-formatted string containing research results
        """
        try:
            # Prepare enhanced query with keywords
            enhanced_query = self._enhance_query(query, keywords)

            # Perform search based on research type
            search_results = self._perform_search(
                enhanced_query, search_engine, sources_limit,
                time_range, domain_filter
            )

            # Scraping additional content if URLs provided
            scraped_content = None
            if urls:
                scraped_content = self._scrape_urls(urls)

            # Perform type-specific analysis
            analysis_result = self._analyze_by_type(
                research_type, search_results, scraped_content
            )

            # Generate structured output
            result = ResearchResult(
                query=query,
                research_type=research_type,
                search_results=search_results,
                scraped_content=scraped_content,
                insights=self._generate_insights(analysis_result) if include_insights else None,
                trends=self._analyze_trends(enhanced_query, research_type) if include_trends else None,
                citations=self._extract_citations(search_results) if include_citations else None,
                competitors=self._identify_competitors(analysis_result) if research_type == ResearchType.COMPETITOR_RESEARCH else None,
                market_data=self._collect_market_data(analysis_result) if research_type == ResearchType.MARKET_RESEARCH else None
            )

            return result.json(indent=2)

        except Exception as e:
            return json.dumps({
                "error": f"Research failed: {str(e)}",
                "query": query,
                "research_type": research_type.value
            }, indent=2)

    def _enhance_query(self, base_query: str, keywords: Optional[List[str]] = None) -> str:
        """
        Enhance the base query with additional keywords for better search results.

        Args:
            base_query: Original search query
            keywords: List of additional keywords

        Returns:
            Enhanced query string
        """
        if not keywords:
            return base_query

        # Combine base query with keywords
        enhanced_parts = [base_query] + keywords
        return " ".join(enhanced_parts)

    def _perform_search(
        self,
        query: str,
        search_engine: SearchEngine,
        limit: int,
        time_range: Optional[str] = None,
        domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search using specified search engine with rate limiting.

        Args:
            query: Search query
            search_engine: Search engine to use
            limit: Maximum number of results
            time_range: Time range filter
            domain_filter: Domain filter

        Returns:
            List of search result dictionaries
        """
        try:
            # Rate limiting delay
            time.sleep(self.rate_limit_delay)

            # Initialize search tool
            search_tool = SerperApiTool()

            # Prepare search parameters
            search_params = {
                "search_query": query,
                "num_results": limit
            }

            if time_range:
                search_params["time_range"] = time_range

            if domain_filter:
                search_params["site"] = domain_filter

            # Execute search with retries
            for attempt in range(self.max_retries):
                try:
                    results = search_tool._run(**search_params)
                    return self._parse_search_results(results, limit)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    time.sleep(self.retry_delay * (attempt + 1))

        except Exception as e:
            raise Exception(f"Search failed after {self.max_retries} attempts: {str(e)}")

    def _parse_search_results(self, raw_results: Any, limit: int) -> List[Dict[str, Any]]:
        """
        Parse and standardize search results.

        Args:
            raw_results: Raw results from search API
            limit: Maximum number of results to return

        Returns:
            List of standardized search result dictionaries
        """
        results = []

        # Handle different result formats
        if isinstance(raw_results, str):
            try:
                raw_results = json.loads(raw_results)
            except json.JSONDecodeError:
                # If not JSON, treat as error message
                return [{"error": raw_results, "url": "", "title": "", "snippet": ""}]

        # Extract search results
        search_items = []
        if isinstance(raw_results, dict):
            # Handle Google/Serper API format
            if "organic_results" in raw_results:
                search_items = raw_results["organic_results"][:limit]
            elif "results" in raw_results:
                search_items = raw_results["results"][:limit]

        # Standardize format
        for item in search_items:
            if isinstance(item, dict):
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", item.get("url", "")),
                    "snippet": item.get("snippet", ""),
                    "display_link": item.get("display_link", ""),
                    "rank": item.get("position", len(results) + 1)
                }
                results.append(result)

        return results

    def _scrape_urls(self, urls: List[HttpUrl]) -> List[Dict[str, Any]]:
        """
        Scrape content from specified URLs with error handling.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of scraped content dictionaries
        """
        scraped_data = []
        scraper = ScrapeWebsiteTool()

        for url in urls:
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)

                # Execute scraping with retries
                for attempt in range(self.max_retries):
                    try:
                        content = scraper._run(website_url=str(url))

                        # Validate and clean content
                        if content and len(content.strip()) > 100:  # Minimum content threshold
                            scraped_data.append({
                                "url": str(url),
                                "content": content[:5000],  # Limit content size
                                "domain": urlparse(str(url)).netloc,
                                "status": "success"
                            })
                            break
                        else:
                            raise ValueError("Empty or insufficient content")

                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            scraped_data.append({
                                "url": str(url),
                                "content": "",
                                "domain": urlparse(str(url)).netloc,
                                "status": "error",
                                "error": str(e)
                            })
                        time.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                scraped_data.append({
                    "url": str(url),
                    "content": "",
                    "domain": urlparse(str(url)).netloc,
                    "status": "error",
                    "error": str(e)
                })

        return scraped_data

    def _analyze_by_type(
        self,
        research_type: ResearchType,
        search_results: List[Dict[str, Any]],
        scraped_content: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Perform type-specific analysis of research data.

        Args:
            research_type: Type of research being performed
            search_results: Search results data
            scraped_content: Optional scraped content

        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            "research_type": research_type.value,
            "total_sources": len(search_results),
            "data_sources": search_results
        }

        if scraped_content:
            analysis["scraped_sources"] = len(scraped_content)
            analysis["scraped_data"] = scraped_content

        # Perform type-specific processing
        if research_type == ResearchType.GENRE_ANALYSIS:
            analysis.update(self._analyze_genre_patterns(search_results, scraped_content))
        elif research_type == ResearchType.MARKET_RESEARCH:
            analysis.update(self._analyze_market_trends(search_results, scraped_content))
        elif research_type == ResearchType.COMPETITOR_RESEARCH:
            analysis.update(self._analyze_competitors(search_results, scraped_content))
        elif research_type == ResearchType.TREND_ANALYSIS:
            analysis.update(self._analyze_temporal_trends(search_results, scraped_content))

        return analysis

    def _analyze_genre_patterns(
        self,
        search_results: List[Dict[str, Any]],
        scraped_content: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze patterns specific to genre research.

        Args:
            search_results: Search results data
            scraped_content: Optional scraped content

        Returns:
            Genre analysis dictionary
        """
        return {
            "genre_keywords": self._extract_keywords(search_results),
            "genre_patterns": self._identify_patterns(search_results, "genre"),
            "genre_evolution": self._analyze_evolution(search_results)
        }

    def _analyze_market_trends(
        self,
        search_results: List[Dict[str, Any]],
        scraped_content: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze market-related patterns and trends.

        Args:
            search_results: Search results data
            scraped_content: Optional scraped content

        Returns:
            Market analysis dictionary
        """
        return {
            "market_keywords": self._extract_keywords(search_results),
            "market_indicators": self._identify_indicators(search_results, "market"),
            "market_size": self._estimate_market_size(search_results),
            "growth_trends": self._analyze_growth_trends(search_results)
        }

    def _analyze_competitors(
        self,
        search_results: List[Dict[str, Any]],
        scraped_content: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze competitor landscape and positioning.

        Args:
            search_results: Search results data
            scraped_content: Optional scraped content

        Returns:
            Competitor analysis dictionary
        """
        return {
            "competitor_keywords": self._extract_keywords(search_results),
            "competitor_intensity": self._measure_competition(search_results),
            "competition_sources": self._identify_competition_sources(search_results)
        }

    def _analyze_temporal_trends(
        self,
        search_results: List[Dict[str, Any]],
        scraped_content: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze temporal trends and pattern evolution.

        Args:
            search_results: Search results data
            scraped_content: Optional scraped content

        Returns:
            Trend analysis dictionary
        """
        return {
            "trend_patterns": self._identify_patterns(search_results, "trend"),
            "temporal_distribution": self._analyze_time_distribution(search_results),
            "trend_velocity": self._calculate_trend_velocity(search_results)
        }

    def _extract_keywords(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract relevant keywords from search results.

        Args:
            search_results: Search results data

        Returns:
            List of extracted keywords
        """
        keywords = set()
        for result in search_results:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()

            # Simple keyword extraction (could be enhanced with NLP)
            words = title.split() + snippet.split()
            for word in words:
                word = word.strip(".,!?;:")
                if len(word) > 3:  # Only longer words
                    keywords.add(word)

        return list(keywords)[:20]  # Limit to top 20 keywords

    def _identify_patterns(
        self,
        search_results: List[Dict[str, Any]],
        pattern_type: str
    ) -> List[str]:
        """
        Identify patterns in research data based on type.

        Args:
            search_results: Search results data
            pattern_type: Type of patterns to identify

        Returns:
            List of identified patterns
        """
        patterns = []

        # Basic pattern identification based on content analysis
        for result in search_results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")

            if pattern_type == "genre":
                # Look for genre-related terms
                if any(word in (title + snippet).lower() for word in
                      ["genre", "literary", "style", "themes", "trope"]):
                    patterns.append(f"Genre pattern identified in: {title}")
            elif pattern_type == "trend":
                # Look for trend-related terms
                if any(word in (title + snippet).lower() for word in
                      ["trend", "emerging", "new", "popular", "rising"]):
                    patterns.append(f"Trend pattern identified in: {title}")

        return patterns[:10]  # Limit to top 10 patterns

    def _analyze_evolution(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze evolution of a topic based on available data.

        Args:
            search_results: Search results data

        Returns:
            Evolution analysis dictionary
        """
        return {
            "evolution_score": len(search_results) * 0.1,  # Placeholder calculation
            "evolution_factors": ["publication_frequency", "topic_maturity"],
            "evolution_stage": "maturing" if len(search_results) > 5 else "emerging"
        }

    def _measure_competition(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Measure competitive intensity from search results.

        Args:
            search_results: Search results data

        Returns:
            Competition level string
        """
        source_count = len(search_results)
        if source_count > 15:
            return "high_competition"
        elif source_count > 7:
            return "moderate_competition"
        else:
            return "low_competition"

    def _identify_competition_sources(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        Identify sources of competition from search results.

        Args:
            search_results: Search results data

        Returns:
            List of competition sources
        """
        competitors = []
        for result in search_results:
            url = result.get("url", "")
            if url and any(domain in url.lower() for domain in
                  ["competitor", "company", "product", "service"]):
                competitors.append(url)
        return competitors[:5]

    def _analyze_time_distribution(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze temporal distribution of results.

        Args:
            search_results: Search results data

        Returns:
            Time distribution analysis
        """
        return {
            "distribution_pattern": "recent_focus" if len(search_results) > 0 else "no_data",
            "frequency_score": len(search_results) * 0.2
        }

    def _calculate_trend_velocity(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate velocity/rate of change for trends.

        Args:
            search_results: Search results data

        Returns:
            Trend velocity assessment
        """
        velocity = len(search_results) * 0.3  # Placeholder calculation
        return {
            "velocity_score": velocity,
            "velocity_level": "rapid" if velocity > 3.0 else "moderate" if velocity > 1.5 else "slow"
        }

    def _identify_indicators(self, search_results: List[Dict[str, Any]], indicator_type: str) -> List[str]:
        """
        Identify key indicators based on type.

        Args:
            search_results: Search results data
            indicator_type: Type of indicators to identify

        Returns:
            List of market indicators
        """
        indicators = []
        for result in search_results:
            snippet = result.get("snippet", "").lower()
            if indicator_type == "market":
                if any(word in snippet for word in ["sales", "revenue", "market", "demand", "supply"]):
                    indicators.append(f"Market indicator: {snippet[:100]}...")
        return indicators[:5]

    def _estimate_market_size(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate market size from available data.

        Args:
            search_results: Search results data

        Returns:
            Market size estimation
        """
        # Placeholder calculation based on result count and quality
        source_count = len(search_results)
        return {
            "estimated_size": source_count * 100000,  # Rough estimate
            "confidence_level": "medium" if source_count > 5 else "low",
            "unit": "estimated_users_or_revenue"
        }

    def _analyze_growth_trends(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze growth trends from search results.

        Args:
            search_results: Search results data

        Returns:
            List of growth trend dictionaries
        """
        trends = []
        if len(search_results) > 0:
            trends.append({
                "trend_type": "search_interest_growth",
                "direction": "increasing" if len(search_results) > 10 else "stable",
                "magnitude": len(search_results) * 0.1
            })
        return trends

    def _generate_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generate AI-powered insights from analysis results.

        Args:
            analysis_result: Results from type-specific analysis

        Returns:
            List of insightful observations
        """
        insights = []

        research_type = analysis_result.get("research_type", "")
        total_sources = analysis_result.get("total_sources", 0)

        # Basic insight generation
        insights.append(f"Research completed with {total_sources} sources for {research_type} analysis.")

        if total_sources > 10:
            insights.append("High volume of sources suggests strong interest in this topic.")
        elif total_sources < 3:
            insights.append("Limited sources available - topic may be niche or emerging.")

        # Type-specific insights
        if research_type == ResearchType.GENRE_ANALYSIS.value:
            genre_keywords = analysis_result.get("genre_keywords", [])
            insights.append(f"Key genre elements identified: {', '.join(genre_keywords[:5])}")
        elif research_type == ResearchType.MARKET_RESEARCH.value:
            market_size = analysis_result.get("market_size", {})
            estimated_size = market_size.get("estimated_size", 0)
            insights.append(f"Estimated market size: {estimated_size:,.0f} units")

        return insights[:5]  # Limit to top 5 insights

    def _analyze_trends(self, query: str, research_type: ResearchType) -> List[Dict[str, Any]]:
        """
        Analyze trends related to the research query.

        Args:
            query: Search query
            research_type: Type of research

        Returns:
            List of trend dictionaries
        """
        trends = []

        # Simulate trend analysis (would integrate with trend APIs in production)
        trends.append({
            "trend_name": f"{query}_interest",
            "trend_value": 75.5,
            "trend_direction": "increasing",
            "time_period": "past_year",
            "confidence": 0.85
        })

        return trends

    def _extract_citations(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract and format citations from search results.

        Args:
            search_results: Search results data

        Returns:
            List of formatted citations
        """
        citations = []
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            if title and url:
                # APA-style citation format
                citation = f"[{i}] {title}. Retrieved from {url}"
                citations.append(citation)

        return citations

    def _identify_competitors(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify and analyze competitors from research data.

        Args:
            analysis_result: Analysis results

        Returns:
            List of competitor dictionaries
        """
        competitors = []
        competition_sources = analysis_result.get("competition_sources", [])

        for source in competition_sources:
            competitors.append({
                "name": source.split('/')[-1].replace('.com', '').replace('www.', ''),
                "url": source,
                "threat_level": "medium",
                "market_presence": "established"
            })

        return competitors

    def _collect_market_data(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect and summarize market data from research results.

        Args:
            analysis_result: Analysis results

        Returns:
            Market data dictionary
        """
        market_size = analysis_result.get("market_size", {})
        growth_trends = analysis_result.get("growth_trends", [])

        return {
            "total_market_size": market_size.get("estimated_size", 0),
            "growth_rate": 0.05,  # Placeholder
            "key_players": len(growth_trends),
            "emerging_opportunities": ["digital_publishing", "AI_assistance"],
            "market_maturity": "maturing"
        }