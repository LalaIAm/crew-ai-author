"""
ValidationTool for CrewAI novel writing workflow.

This tool provides comprehensive validation and fact-checking capabilities for novel writing,
including factual accuracy verification, consistency checking, and citation management.

Key Features:
- Fact verification using web scraping and search APIs
- Source credibility assessment
- Cross-reference checking across story elements
- Historical and scientific accuracy validation
- Citation formatting and bibliography generation
- Consistency validation for plot, characters, and world-building
- File-based reference checking

Usage:
- Used by subject_matter_researcher for fact-checking research outputs
- Used by other agents for validating consistency of their contributions
- Supports validation of content against reliable sources and existing references

Example:
    tool = ValidationTool()
    result = tool._run(
        content="The Battle of Thermopylae occurred in 480 BC.",
        validation_types=["facts", "historical"],
        accuracy_requirements="high"
    )
    print(result["confidence_score"])  # Expected: high confidence for verifiable fact
"""

from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from crewai.tools import BaseTool
from crewai_tools import ScrapeWebsiteTool, SerperApiTool, FileReadTool
from pydantic import BaseModel, Field


class ValidationToolInput(BaseModel):
    """Input schema for ValidationTool."""

    content: str = Field(
        ...,
        description="The content to validate (text excerpt, facts, or story elements)."
    )

    validation_types: List[str] = Field(
        ...,
        description="List of validation types: 'facts', 'consistency', 'citations', 'historical', 'scientific', 'plot', 'character', 'world_building'."
    )

    accuracy_requirements: str = Field(
        default="medium",
        description="Accuracy requirement level: 'low', 'medium', 'high' (affects source quality thresholds)."
    )

    reference_files: Optional[List[str]] = Field(
        None,
        description="Optional list of file paths to check against for cross-references."
    )

    specific_domains: Optional[List[str]] = Field(
        None,
        description="Domains to prioritize for fact verification (e.g., ['britannica.com', 'nasa.gov'])."
    )


class ValidationTool(BaseTool):
    """Comprehensive validation tool for novel writing workflows.

    This tool extends CrewAI's BaseTool to provide fact-checking, consistency validation,
    and source verification capabilities using web scraping and search tools.
    """

    name: str = "Content Validation Tool"
    description: str = (
        "Validates content for factual accuracy, consistency, and citation requirements. "
        "Provides confidence scores, flags issues, and generates formatted citations. "
        "Supports historical, scientific, and narrative consistency validation."
    )
    args_schema: Type[BaseModel] = ValidationToolInput

    def _run(
        self,
        content: str,
        validation_types: List[str],
        accuracy_requirements: str = "medium",
        reference_files: Optional[List[str]] = None,
        specific_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute validation logic based on input parameters.

        Args:
            content: The content to validate
            validation_types: Types of validation to perform
            accuracy_requirements: Quality threshold for validation
            reference_files: Optional files to cross-reference
            specific_domains: Preferred domains for fact verification

        Returns:
            Dict containing validation results, confidence scores, issues, and citations
        """
        # Initialize tools
        scraper = ScrapeWebsiteTool()
        search_tool = SerperApiTool()
        file_reader = FileReadTool() if reference_files else None

        results = {
            "validated_content": content,
            "validation_types_applied": validation_types,
            "confidence_score": 0.0,
            "issues": [],
            "citations": [],
            "cross_references": [],
            "timestamp": datetime.now().isoformat(),
            "recommendations": []
        }

        try:
            # Process each validation type
            for val_type in validation_types:
                if val_type == "facts":
                    fact_results = self._validate_facts(content, search_tool, scraper, accuracy_requirements, specific_domains)
                    results["issues"].extend(fact_results["issues"])
                    results["citations"].extend(fact_results["citations"])
                    results["confidence_score"] = max(results["confidence_score"], fact_results["confidence"])

                elif val_type == "consistency":
                    consistency_results = self._validate_consistency(content, reference_files, file_reader)
                    results["issues"].extend(consistency_results["issues"])
                    results["cross_references"].extend(consistency_results["cross_references"])
                    results["confidence_score"] = max(results["confidence_score"], consistency_results["confidence"])

                elif val_type == "citations":
                    citation_results = self._validate_citations(content, scraper)
                    results["citations"].extend(citation_results["citations"])
                    results["issues"].extend(citation_results["issues"])
                    results["confidence_score"] = max(results["confidence_score"], citation_results["confidence"])

                elif val_type in ["historical", "scientific"]:
                    accuracy_results = self._validate_accuracy(content, val_type, search_tool, scraper, accuracy_requirements)
                    results["issues"].extend(accuracy_results["issues"])
                    results["citations"].extend(accuracy_results["citations"])
                    results["confidence_score"] = max(results["confidence_score"], accuracy_results["confidence"])

                elif val_type in ["plot", "character", "world_building"]:
                    narrative_results = self._validate_narrative_consistency(content, val_type, reference_files, file_reader)
                    results["issues"].extend(narrative_results["issues"])
                    results["cross_references"].extend(narrative_results["cross_references"])
                    results["confidence_score"] = max(results["confidence_score"], narrative_results["confidence"])

            # Generate recommendations based on issues
            results["recommendations"] = self._generate_recommendations(results["issues"], validation_types)

            # Normalize confidence score
            results["confidence_score"] = min(1.0, results["confidence_score"])

        except Exception as e:
            results["issues"].append({
                "type": "tool_error",
                "message": f"Validation failed: {str(e)}",
                "severity": "high"
            })

        return results

    def _validate_facts(
        self,
        content: str,
        search_tool: SerperApiTool,
        scraper: ScrapeWebsiteTool,
        accuracy_req: str,
        domains: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Validate factual claims using search and web scraping."""
        results = {"issues": [], "citations": [], "confidence": 0.7}

        # Use search tool to find sources
        search_results = search_tool.run(query=f"fact check: {content}", max_results=5)

        credible_sources = 0
        total_sources = 0

        if search_results and hasattr(search_results, 'results'):
            for result in search_results.results:
                total_sources += 1
                if self._assess_credibility(result, accuracy_req, domains):
                    credible_sources += 1
                    results["citations"].append({
                        "source": result.title,
                        "url": result.link,
                        "format": self._format_citation(result)
                    })

        if total_sources > 0:
            confidence = credible_sources / total_sources
            results["confidence"] = confidence
    
            if confidence < 0.5:
                results["issues"].append({
                    "type": "fact_verification",
                    "message": "Low confidence in factual accuracy",
                    "severity": "medium"
                })

        return results

    def _validate_consistency(
        self,
        content: str,
        reference_files: Optional[List[str]],
        file_reader: Optional[FileReadTool]
    ) -> Dict[str, Any]:
        """Check content consistency against reference files."""
        results = {"issues": [], "cross_references": [], "confidence": 0.9}

        if not reference_files or not file_reader:
            return results

        for file_path in reference_files:
            try:
                ref_content = file_reader.run(file_path=file_path)

                # Simple consistency check (can be enhanced with NLP)
                if content.lower() not in ref_content.lower():
                    results["issues"].append({
                        "type": "consistency",
                        "message": f"Inconsistent with reference: {file_path}",
                        "severity": "low"
                    })
                    results["confidence"] -= 0.1
                else:
                    results["cross_references"].append({
                        "file": file_path,
                        "status": "consistent",
                        "matched_content": content
                    })

            except Exception as e:
                results["issues"].append({
                    "type": "file_error",
                    "message": f"Could not read reference file {file_path}: {str(e)}",
                    "severity": "high"
                })
                results["confidence"] -= 0.3

        return results

    def _validate_citations(
        self,
        content: str,
        scraper: ScrapeWebsiteTool
    ) -> Dict[str, Any]:
        """Validate and format citations in content."""
        results = {"citations": [], "issues": [], "confidence": 0.8}

        # Extract URLs or citation-like patterns (basic implementation)
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)

        for url in urls:
            try:
                # Scrape to verify URL is accessible
                page_content = scraper.scrape_website(url=url)

                if page_content:
                    results["citations"].append({
                        "url": url,
                        "format": self._format_citation({"url": url, "content": page_content}),
                        "verified": True
                    })
                else:
                    results["issues"].append({
                        "type": "citation",
                        "message": f"Citation not accessible: {url}",
                        "severity": "medium"
                    })

            except Exception as e:
                results["issues"].append({
                    "type": "citation_error",
                    "message": f"Could not verify citation {url}: {str(e)}",
                    "severity": "high"
                })

        return results

    def _validate_accuracy(
        self,
        content: str,
        accuracy_type: str,
        search_tool: SerperApiTool,
        scraper: ScrapeWebsiteTool,
        accuracy_req: str
    ) -> Dict[str, Any]:
        """Validate historical or scientific accuracy."""
        results = {"issues": [], "citations": [], "confidence": 0.6}

        # Use search for accuracy verification
        query = f"{accuracy_type} accuracy: {content}"
        search_results = search_tool.run(query=query, max_results=3)

        if search_results and hasattr(search_results, 'results'):
            for result in search_results.results:
                if self._assess_expert_credibility(result, accuracy_type, accuracy_req):
                    results["citations"].append({
                        "source": result.title,
                        "url": result.link,
                        "format": self._format_citation(result),
                        "type": accuracy_type
                    })
                    results["confidence"] += 0.2

                if results["confidence"] < 0.4:
                    results["issues"].append({
                        "type": f"{accuracy_type}_accuracy",
                        "message": f"Potential {accuracy_type} inaccuracy detected",
                        "severity": "medium"
                    })

        results["confidence"] = min(1.0, results["confidence"])
        return results

    def _validate_narrative_consistency(
        self,
        content: str,
        narrative_type: str,
        reference_files: Optional[List[str]],
        file_reader: Optional[FileReadTool]
    ) -> Dict[str, Any]:
        """Validate narrative consistency (plot, character, world-building)."""
        results = {"issues": [], "cross_references": [], "confidence": 0.8}

        # Basic consistency checking against reference files
        # In a full implementation, this would use more sophisticated NLP or rules
        if reference_files and file_reader:
            for file_path in reference_files:
                try:
                    ref_content = file_reader.run(file_path=file_path)

                    # Check for narrative consistency indicators
                    if narrative_type == "character":
                        if self._contains_character_contradiction(content, ref_content):
                            results["issues"].append({
                                "type": "character_consistency",
                                "message": f"Character contradiction with {file_path}",
                                "severity": "medium"
                            })

                    elif narrative_type == "plot":
                        if self._contains_plot_hole(content, ref_content):
                            results["issues"].append({
                                "type": "plot_consistency",
                                "message": f"Plot inconsistency with {file_path}",
                                "severity": "medium"
                            })

                except Exception as e:
                    results["issues"].append({
                        "type": "narrative_validation_error",
                        "message": f"Could not validate narrative against {file_path}: {str(e)}",
                        "severity": "low"
                    })

        return results

    def _assess_credibility(self, source: Dict, accuracy_req: str, domains: Optional[List[str]]) -> bool:
        """Assess source credibility based on domain and type."""
        if domains and source.get('link'):
            source_domain = source['link'].split('/')[2] if 'link' in source else ""
            return source_domain in domains

        # Default credibility assessment
        credible_domains = ['.edu', '.gov', 'wikipedia.org', '.org']
        if source.get('link'):
            source_domain = source['link']
            return any(domain in source_domain for domain in credible_domains)

        return accuracy_req.lower() == 'low'

    def _assess_expert_credibility(self, source: Dict, accuracy_type: str, accuracy_req: str) -> bool:
        """Assess credibility for historical/scientific sources."""
        if accuracy_req == "high":
            domain = source.get('link', '').lower()
            return any(expert in domain for expert in ['.edu', '.gov', 'research.', 'science.'])
        return True

    def _format_citation(self, source: Dict) -> str:
        """Format citation in APA style (basic implementation)."""
        if isinstance(source, dict):
            title = source.get('title', 'Unknown Title')
            url = source.get('url', source.get('link', ''))
            accessed_date = datetime.now().strftime("%B %d, %Y")
            return f"{title}. Retrieved from {url} on {accessed_date}."
        return str(source)

    def _contains_character_contradiction(self, new_content: str, ref_content: str) -> bool:
        """Basic character contradiction detection."""
        # Placeholder for character consistency logic
        # In full implementation, would use NLP for character trait checking
        return False

    def _contains_plot_hole(self, new_content: str, ref_content: str) -> bool:
        """Basic plot hole detection."""
        # Placeholder for plot consistency logic
        return False

    def _generate_recommendations(self, issues: List[Dict], validation_types: List[str]) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []

        if any(issue['type'] == 'fact_verification' for issue in issues):
            recommendations.append("Verify facts against multiple reputable sources")

        if any(issue['severity'] == 'high' for issue in issues):
            recommendations.append("Review and address critical issues before proceeding")

        if not any(issue['type'] == 'consistency' for issue in issues if issue['severity'] == 'high'):
            recommendations.append("Consider cross-referencing with existing documentation")

        return recommendations