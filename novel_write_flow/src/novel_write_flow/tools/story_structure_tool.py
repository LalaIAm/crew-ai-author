from typing import Any, Dict, List, Optional, Union, Type
from enum import Enum
import json
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, validator


class AnalysisType(str, Enum):
    """Enumeration of supported analysis types."""
    PLOT_STRUCTURE = "plot_structure"
    CHARACTER_DEVELOPMENT = "character_development"
    WORLDBUILDING = "worldbuilding"
    RELATIONSHIPS = "relationships"
    STORY_CONSISTENCY = "story_consistency"
    NARRATIVE_FLOW = "narrative_flow"


class PlotFramework(str, Enum):
    """Enumeration of supported plot frameworks."""
    THREE_ACT = "three_act"
    HERO_JOURNEY = "hero_journey"
    SAVE_THE_CAT = "save_the_cat"
    SEVEN_POINT = "seven_point"
    SNOWFLAKE = "snowflake"
    CUSTOM = "custom"


class OutputType(str, Enum):
    """Enumeration of output types."""
    OUTLINE = "outline"
    PROFILE = "profile"
    ANALYSIS = "analysis"
    TEMPLATE = "template"
    VALIDATION = "validation"


class StoryStructureInput(BaseModel):
    """
    Input schema for StoryStructureTool.

    This tool supports comprehensive story structure analysis and generation
    including plot frameworks, character development, world-building, and
    narrative consistency validation.
    """

    story_concept: str = Field(
        ...,
        description="Main story concept, premise, or plot summary to analyze or develop"
    )

    analysis_type: AnalysisType = Field(
        AnalysisType.PLOT_STRUCTURE,
        description="Type of analysis to perform (plot_structure, character_development, worldbuilding, relationships, story_consistency, narrative_flow)"
    )

    plot_framework: Optional[PlotFramework] = Field(
        None,
        description="Plot structure framework to use (three_act, hero_journey, save_the_cat, seven_point, snowflake, custom)"
    )

    output_type: OutputType = Field(
        OutputType.ANALYSIS,
        description="Desired output type (outline, profile, analysis, template, validation)"
    )

    character_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Character information for development analysis"
    )

    world_details: Optional[Dict[str, Any]] = Field(
        None,
        description="World-building elements and rules for consistency checking"
    )

    template_file: Optional[str] = Field(
        None,
        description="Path to template file to use for generation"
    )

    output_file: Optional[str] = Field(
        None,
        description="Path to save generated output"
    )

    include_validation: bool = Field(
        True,
        description="Whether to include consistency validation in output"
    )

    detailed_analysis: bool = Field(
        False,
        description="Whether to perform detailed analysis (takes longer but more comprehensive)"
    )

    @validator('character_details', 'world_details', pre=True, always=True)
    def validate_details(cls, v):
        """Ensure details are dictionaries when provided."""
        if v is None:
            return {}
        return v if isinstance(v, dict) else {}


class PlotStructureResult(BaseModel):
    """Structured output for plot structure analysis."""
    framework: str
    structure: Dict[str, Any]
    key_elements: List[Dict[str, str]]
    timeline: List[Dict[str, Any]]
    conflicts: List[str]
    turning_points: List[str]


class CharacterProfile(BaseModel):
    """Character development profile structure."""
    name: str
    archetype: str
    personality_traits: List[str]
    motivations: List[str]
    relationships: Dict[str, str]
    character_arc: List[Dict[str, str]]
    psychological_profile: Dict[str, Any]


class WorldBuildingTemplate(BaseModel):
    """World-building framework structure."""
    geographical: Dict[str, Any]
    societal: Dict[str, Any]
    magical_rules: Optional[Dict[str, Any]] = None
    historical_events: List[Dict[str, Any]]
    cultural_elements: Dict[str, str]
    consistency_check: Dict[str, Any]


class ValidationResult(BaseModel):
    """Story consistency validation results."""
    overall_score: float
    issues_found: List[Dict[str, str]]
    recommendations: List[str]
    consistency_report: Dict[str, Any]


class StoryStructureTool(BaseTool):
    """
    Comprehensive Story Structure Tool for narrative analysis and development.

    This multi-purpose tool provides advanced capabilities for story creation and analysis,
    supporting multiple agents in the novel writing process:
    - plot_architect: Uses PLOT_STRUCTURE and NARRATIVE_FLOW analysis
    - character_designer: Uses CHARACTER_DEVELOPMENT and RELATIONSHIPS analysis
    - world_builder: Uses WORLDBUILDING and STORY_CONSISTENCY analysis

    The tool integrates with FileReadTool and FileWriteTool for template access
    and output generation, providing seamless workflow integration.

    Features:
    - Multiple plot structure frameworks (Three-Act, Hero's Journey, Save the Cat, etc.)
    - Character development analysis with psychological profiling
    - World-building templates with consistency validation
    - Relationship mapping and social dynamics analysis
    - Narrative flow analysis and pacing suggestions
    - Comprehensive validation for story consistency
    - Template-based output generation
    - File I/O integration for templates and results

    Usage Examples:

    # Basic plot structure analysis
    tool_input = StoryStructureInput(
        story_concept="A young wizard discovers an ancient artifact that could change the world",
        analysis_type=AnalysisType.PLOT_STRUCTURE,
        plot_framework=PlotFramework.HERO_JOURNEY,
        output_type=OutputType.OUTLINE
    )

    # Character development with detailed analysis
    tool_input = StoryStructureInput(
        story_concept="Protagonist's journey through self-discovery",
        analysis_type=AnalysisType.CHARACTER_DEVELOPMENT,
        character_details={
            "name": "Alex Morgan",
            "background": "Former athlete turned detective"
        },
        detailed_analysis=True,
        output_type=OutputType.PROFILE
    )

    # World-building with consistency checking
    tool_input = StoryStructureInput(
        story_concept="A fantasy world where magic is tied to emotions",
        analysis_type=AnalysisType.WORLDBUILDING,
        world_details={
            "magic_system": "emotion_based",
            "societies": ["human", "elf", "dwarf"]
        },
        output_type=OutputType.TEMPLATE,
        include_validation=True
    )

    # Story consistency validation
    tool_input = StoryStructureInput(
        story_concept="Complete novel draft",
        analysis_type=AnalysisType.STORY_CONSISTENCY,
        output_type=OutputType.VALIDATION,
        template_file="templates/validation_config.json"
    )

    # Relationship mapping analysis
    tool_input = StoryStructureInput(
        story_concept="Complex family drama",
        analysis_type=AnalysisType.RELATIONSHIPS,
        character_details={
            "characters": ["Character A", "Character B", "Character C"],
            "relationships": ["parent-child", "siblings", "romantic"]
        },
        output_type=OutputType.ANALYSIS
    )

    Attributes:
        name: Name of the tool
        description: Clear description for agent usage
        args_schema: Pydantic input model schema
        template_path: Default path for templates
        output_path: Default path for generated outputs
    """

    name: str = "Advanced Story Structure Tool"
    description: str = (
        "A comprehensive tool for story structure analysis, character development, "
        "world-building frameworks, and narrative consistency validation. Supports "
        "multiple plot frameworks including Three-Act, Hero's Journey, and Save the Cat. "
        "Designed for shared use by plot_architect, character_designer, and world_builder agents."
    )
    args_schema: Type[BaseModel] = StoryStructureInput

    # Default paths (can be overridden)
    template_path: str = "templates/story_structure"
    output_path: str = "outputs/story_structure"

    def _run(
        self,
        story_concept: str,
        analysis_type: AnalysisType = AnalysisType.PLOT_STRUCTURE,
        plot_framework: Optional[PlotFramework] = None,
        output_type: OutputType = OutputType.ANALYSIS,
        character_details: Optional[Dict[str, Any]] = None,
        world_details: Optional[Dict[str, Any]] = None,
        template_file: Optional[str] = None,
        output_file: Optional[str] = None,
        include_validation: bool = True,
        detailed_analysis: bool = False
    ) -> str:
        """
        Execute story structure analysis and generation.

        Args:
            story_concept: Main story concept or premise
            analysis_type: Type of analysis to perform
            plot_framework: Plot structure framework to use
            output_type: Desired output format
            character_details: Character information for development
            world_details: World-building details for consistency
            template_file: Path to template file
            output_file: Path to save results
            include_validation: Whether to include validation
            detailed_analysis: Whether to use detailed analysis

        Returns:
            JSON-formatted string containing analysis results
        """
        try:
            # Create input object for consistency
            input_obj = StoryStructureInput(
                story_concept=story_concept,
                analysis_type=analysis_type,
                plot_framework=plot_framework,
                output_type=output_type,
                character_details=character_details,
                world_details=world_details,
                template_file=template_file,
                output_file=output_file,
                include_validation=include_validation,
                detailed_analysis=detailed_analysis
            )

            # Perform type-specific analysis
            result = self._analyze_by_type(input_obj)

            # Generate output in requested format
            output = self._generate_output(result, output_type, template_file)

            # Include validation if requested
            if include_validation and analysis_type != AnalysisType.STORY_CONSISTENCY:
                validation = self._validate_story_consistency(
                    story_concept, character_details, world_details
                )
                output["validation"] = validation.dict()

            # Save to file if requested
            if output_file:
                self._save_output(output, output_file)

            return json.dumps(output, indent=2, default=str)

        except Exception as e:
            return json.dumps({
                "error": f"Story structure analysis failed: {str(e)}",
                "story_concept": story_concept,
                "analysis_type": analysis_type.value if hasattr(analysis_type, 'value') else str(analysis_type)
            }, indent=2)

    def _analyze_by_type(self, input_obj: StoryStructureInput) -> Dict[str, Any]:
        """
        Perform type-specific analysis based on input.

        Args:
            input_obj: Input configuration object

        Returns:
            Dictionary containing analysis results
        """
        analysis_result = {
            "analysis_type": input_obj.analysis_type.value,
            "story_concept": input_obj.story_concept,
            "timestamp": str(input_obj.__dict__.get('_timestamp', 'N/A')),
            "detailed": input_obj.detailed_analysis
        }

        if input_obj.analysis_type == AnalysisType.PLOT_STRUCTURE:
            analysis_result["plot_structure"] = self._analyze_plot_structure(
                input_obj.story_concept, input_obj.plot_framework, input_obj.detailed_analysis
            )
        elif input_obj.analysis_type == AnalysisType.CHARACTER_DEVELOPMENT:
            analysis_result["character_development"] = self._analyze_character_development(
                input_obj.story_concept, input_obj.character_details, input_obj.detailed_analysis
            )
        elif input_obj.analysis_type == AnalysisType.WORLDBUILDING:
            analysis_result["worldbuilding"] = self._analyze_worldbuilding(
                input_obj.story_concept, input_obj.world_details, input_obj.detailed_analysis
            )
        elif input_obj.analysis_type == AnalysisType.RELATIONSHIPS:
            analysis_result["relationships"] = self._analyze_relationships(
                input_obj.story_concept, input_obj.character_details
            )
        elif input_obj.analysis_type == AnalysisType.STORY_CONSISTENCY:
            analysis_result["consistency"] = self._validate_story_consistency(
                input_obj.story_concept, input_obj.character_details, input_obj.world_details
            ).dict()
        elif input_obj.analysis_type == AnalysisType.NARRATIVE_FLOW:
            analysis_result["narrative_flow"] = self._analyze_narrative_flow(
                input_obj.story_concept, input_obj.detailed_analysis
            )

        return analysis_result

    def _analyze_plot_structure(
        self,
        concept: str,
        framework: Optional[PlotFramework],
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze plot structure using specified framework.

        Args:
            concept: Story concept to analyze
            framework: Plot framework to use
            detailed: Whether to include detailed analysis

        Returns:
            Plot structure analysis result
        """
        if not framework:
            framework = PlotFramework.THREE_ACT

        structure_data = self._get_plot_framework_structure(framework, concept)

        if detailed:
            structure_data.update({
                "pacing_analysis": self._analyze_pacing(structure_data),
                "conflict_hierarchy": self._analyze_conflicts(concept),
                "character_arc_analysis": self._analyze_character_arcs(concept)
            })

        return PlotStructureResult(**structure_data).dict()

    def _get_plot_framework_structure(
        self,
        framework: PlotFramework,
        concept: str
    ) -> Dict[str, Any]:
        """
        Get structure template for specified plot framework.

        Args:
            framework: Plot framework type
            concept: Story concept

        Returns:
            Framework-specific structure data
        """
        if framework == PlotFramework.THREE_ACT:
            return self._three_act_structure(concept)
        elif framework == PlotFramework.HERO_JOURNEY:
            return self._hero_journey_structure(concept)
        elif framework == PlotFramework.SAVE_THE_CAT:
            return self._save_the_cat_structure(concept)
        elif framework == PlotFramework.SEVEN_POINT:
            return self._seven_point_structure(concept)
        elif framework == PlotFramework.SNOWFLAKE:
            return self._snowflake_structure(concept)
        else:  # CUSTOM
            return self._custom_plot_structure(concept)

    def _three_act_structure(self, concept: str) -> Dict[str, Any]:
        """Generate Three-Act plot structure."""
        return {
            "framework": "three_act",
            "structure": {
                "act_1": {
                    "purpose": "Setup and introduction",
                    "percentage": 25,
                    "key_elements": ["Inciting incident", "Introduction of main characters", "Establish world"]
                },
                "act_2": {
                    "purpose": "Confrontation and rising action",
                    "percentage": 50,
                    "key_elements": ["Rising complications", "Character development", "Major conflicts"]
                },
                "act_3": {
                    "purpose": "Resolution and falling action",
                    "percentage": 25,
                    "key_elements": ["Climax", "Resolution", "Character growth conclusion"]
                }
            },
            "key_elements": [
                {"act": "1", "element": "Setup", "description": "World, characters, inciting incident"},
                {"act": "2", "element": "Confrontation", "description": "Rising action, complications"},
                {"act": "3", "element": "Resolution", "description": "Climax and resolution"}
            ],
            "timeline": self._generate_timeline(act_structure=True),
            "conflicts": ["Primary conflict", "Secondary conflicts", "Internal struggles"],
            "turning_points": ["Inciting incident", "Lock-in", "Major reversal", "Climax", "Resolution"]
        }

    def _hero_journey_structure(self, concept: str) -> Dict[str, Any]:
        """Generate Hero's Journey plot structure."""
        return {
            "framework": "hero_journey",
            "structure": {
                "ordinary_world": "Hero's normal life",
                "call_to_adventure": "Inciting incident that disrupts normalcy",
                "refusal_of_call": "Hero's reluctance to change",
                "meeting_the_mentor": "Guidance figure appears",
                "crossing_threshold": "Hero enters unknown world",
                "tests_allies_enemies": "Challenges and relationships form",
                "approach_inner_cave": "Preparation for major challenge",
                "supreme_ordeal": "Most difficult test or confrontation",
                "reward": "Hero gains reward for overcoming ordeal",
                "road_back": "Journey back to ordinary world",
                "resurrection": "Final confrontation or rebirth",
                "return_home": "Hero returns transformed"
            },
            "key_elements": [
                {"stage": "Departure", "elements": ["Ordinary World", "Call to Adventure", "Refusal", "Mentor", "Threshold"]},
                {"stage": "Initiation", "elements": ["Tests", "Allies & Enemies", "Ordeal", "Reward"]},
                {"stage": "Return", "elements": ["Road Back", "Resurrection", "Return with Elixir"]}
            ],
            "timeline": self._generate_timeline(hero_journey=True),
            "conflicts": ["Internal doubts", "External antagonists", "Moral dilemmas", "Personal growth obstacles"],
            "turning_points": ["Call to Adventure", "Crossing the Threshold", "Supreme Ordeal", "Resurrection"]
        }

    def _save_the_cat_structure(self, concept: str) -> Dict[str, Any]:
        """Generate Save the Cat plot structure."""
        return {
            "framework": "save_the_cat",
            "structure": {
                "opening_image": "Protagonist in their comfort zone",
                "theme_stated": "Theme or lesson explicitly stated",
                "setup": "Establish protagonist's world and problem",
                "catalyst": "Inciting incident that disrupts status quo",
                "debate": "Protagonist debates whether to act",
                "break_into_two": "Protagonist enters new world",
                "b_story": "Meeting with love interest or ally",
                "fun_and_games": "Hero learns new world rules",
                "midpoint": "Major shift or revelation",
                "bad_guys_close_in": "Forces of antagonism threaten hero",
                "all_is_lost": "Dark night of the soul - hope seems lost",
                "dark_night_of_soul": "Protagonist hits emotional low",
                "break_into_three": "Hero finds new inspiration",
                "finale": "Knot of conflict wound tighter and tighter",
                "final_image": "Protagonist transformed"
            },
            "key_elements": [
                {"act": "Act 1", "element": "Setup", "beats": ["Opening Image", "Theme Stated", "Setup", "Catalyst", "Debate", "Break into Two"]},
                {"act": "Act 2A", "element": "Confrontation", "beats": ["B Story", "Fun and Games", "Midpoint", "Bad Guys Close In"]},
                {"act": "Act 2B", "element": "Deepening Conflict", "beats": ["All Is Lost", "Dark Night of Soul", "Break into Three", "Finale"]},
                {"act": "Act 3", "element": "Resolution", "beats": ["Act Three", "Final Image"]}
            ],
            "timeline": self._generate_timeline(save_cat=True),
            "conflicts": ["Internal conflict", "Interpersonal conflict", "Antagonist opposition", "Situational conflict"],
            "turning_points": ["Catalyst", "Midpoint", "All Is Lost", "Break into Three", "Final Image"]
        }

    def _seven_point_structure(self, concept: str) -> Dict[str, Any]:
        """Generate Seven-Point plot structure."""
        return {
            "framework": "seven_point",
            "structure": {
                "hook": "Opening hook to grab attention",
                "plot_turn_1": "First major turning point",
                "pinch_1": "First pinch point raising stakes",
                "midpoint": "Story midpoint with major change",
                "pinch_2": "Second pinch point crisis moment",
                "plot_turn_2": "Second major turning point",
                "resolution": "Story climax and resolution"
            },
            "key_elements": [
                {"point": "1", "element": "Hook", "description": "Grab reader attention immediately"},
                {"point": "2", "element": "Plot Turn 1", "description": "First major change in story direction"},
                {"point": "3", "element": "Pinch 1", "description": "First crisis moment raising stakes"},
                {"point": "4", "element": "Midpoint", "description": "Major shift or revelation (halfway point)"},
                {"point": "5", "element": "Pinch 2", "description": "Second crisis with higher stakes"},
                {"point": "6", "element": "Plot Turn 2", "description": "Second major change"},
                {"point": "7", "element": "Resolution", "description": "Climax and story conclusion"}
            ],
            "timeline": self._generate_timeline(seven_point=True),
            "conflicts": ["Opening conflict", "Rising complications", "Midpoint crisis", "Final confrontation"],
            "turning_points": ["Hook", "Plot Turn 1", "Midpoint", "Plot Turn 2", "Resolution"]
        }

    def _snowflake_structure(self, concept: str) -> Dict[str, Any]:
        """Generate Snowflake Method structure."""
        return {
            "framework": "snowflake",
            "structure": {
                "single_sentence": "One-sentence summary of entire novel",
                "paragraph": "One-paragraph summary",
                "character_sketches": "One-page character descriptions",
                "expanded_summary": "Four-page story summary (one page per act)",
                "character_charts": "Character charts for each main character",
                "scene_list": "List of scenes with one-sentence descriptions",
                "scene_synopses": "One-paragraph synopses for each scene",
                "first_drafts": "Scene writing based on synopses"
            },
            "key_elements": [
                {"component": "Summary Layer", "elements": ["Single Sentence", "Paragraph", "Page Summary", "Four-Page Expanded"]},
                {"component": "Character Layer", "elements": ["Character Sketches", "Character Charts"]},
                {"component": "Scene Layer", "elements": ["Scene List", "Scene Synopses", "Full Scenes"]}
            ],
            "timeline": self._generate_timeline(snowflake=True),
            "conflicts": ["Conceptual conflicts", "Character conflicts", "Scene-level conflicts"],
            "turning_points": ["Expanding from core premise", "Each layer reveals new complexities"]
        }

    def _custom_plot_structure(self, concept: str) -> Dict[str, Any]:
        """Generate custom plot structure based on concept."""
        return {
            "framework": "custom",
            "structure": {
                "core_concept": concept,
                "custom_structure": "To be defined based on story requirements"
            },
            "key_elements": [{"element": "Custom Structure", "description": "Tailored to specific story needs"}],
            "timeline": self._generate_timeline(custom=True),
            "conflicts": ["Story-specific conflicts"],
            "turning_points": ["Concept-driven turning points"]
        }

    def _generate_timeline(
        self,
        act_structure: bool = False,
        hero_journey: bool = False,
        save_cat: bool = False,
        seven_point: bool = False,
        snowflake: bool = False,
        custom: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate timeline structure for different frameworks.

        Args:
            Various framework flags

        Returns:
            Timeline as list of dictionaries
        """
        if act_structure:
            return [
                {"position": "0%", "event": "Beginning"},
                {"position": "25%", "event": "Act 1 End"},
                {"position": "75%", "event": "Act 2 End"},
                {"position": "100%", "event": "Ending"}
            ]
        elif hero_journey:
            return [
                {"stage": "Ordinary World", "position": "Beginning"},
                {"stage": "Special World", "position": "Middle"},
                {"stage": "Final Conflict", "position": "Late Middle"},
                {"stage": "Return Home", "position": "End"}
            ]
        elif save_cat:
            return [
                {"position": "1%", "event": "Opening Image"},
                {"position": "5%", "event": "Theme Stated"},
                {"position": "10%", "event": "Setup"},
                {"position": "20%", "event": "Catalyst"},
                {"position": "20%", "event": "Debate"},
                {"position": "30%", "event": "Break into Two"},
                {"position": "55%", "event": "Midpoint"},
                {"position": "80%", "event": "Break into Three"},
                {"position": "110%", "event": "Final Image"}
            ]
        elif seven_point:
            return [
                {"point": 1, "position": "Beginning", "event": "Hook"},
                {"point": 2, "position": "Early", "event": "First Turning Point"},
                {"point": 3, "position": "First Quarter", "event": "First Pinch"},
                {"point": 4, "position": "Middle", "event": "Midpoint"},
                {"point": 5, "position": "Third Quarter", "event": "Second Pinch"},
                {"point": 6, "position": "Late", "event": "Second Turning Point"},
                {"point": 7, "position": "End", "event": "Resolution"}
            ]
        else:
            return [{"position": "Throughout", "event": "Story progression"}]

    def _analyze_pacing(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pacing and rhythm of the story structure."""
        return {
            "overall_pacing": "balanced",
            "tension_curve": "gradual build with climax",
            "recommendations": ["Maintain rising action", "Pace character development", "Build to satisfying climax"]
        }

    def _analyze_conflicts(self, concept: str) -> Dict[str, Any]:
        """Analyze conflict hierarchy in the story."""
        return {
            "primary_conflict": "Main story problem",
            "secondary_conflicts": ["Supporting challenges", "Character-level struggles"],
            "internal_conflicts": ["Character growth obstacles", "Moral dilemmas"]
        }

    def _analyze_character_arcs(self, concept: str) -> Dict[str, Any]:
        """Analyze character development arcs."""
        return {
            "protagonist_arc": "Growth through challenges",
            "supporting_arcs": ["Allies develop", "Antagonists revealed"],
            "transformation_points": ["Inciting incident", "Midpoint crisis", "Resolution change"]
        }

    def _analyze_character_development(
        self,
        concept: str,
        character_details: Optional[Dict[str, Any]],
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze character development and create profiles.

        Args:
            concept: Story concept
            character_details: Character information
            detailed: Whether to include psychological profiling

        Returns:
            Character development analysis
        """
        characters = []
        if character_details:
            # Process provided character details
            for name, details in character_details.items():
                if isinstance(details, dict):
                    profile = self._create_character_profile(name, details, detailed)
                    characters.append(profile.dict())

        if not characters:
            # Generate default character analysis based on concept
            characters.append(self._analyze_concept_characters(concept, detailed).dict())

        return {
            "characters_analyzed": len(characters),
            "character_profiles": characters,
            "development_frameworks": ["Hero's Journey Arc", "Psychological Growth", "Relationship Dynamics"],
            "recommendations": ["Deepen backstories", "Add conflicting motivations", "Ensure character agency"]
        }

    def _create_character_profile(
        self,
        name: str,
        details: Dict[str, Any],
        detailed: bool = False
    ) -> CharacterProfile:
        """
        Create comprehensive character profile.

        Args:
            name: Character name
            details: Character details
            detailed: Whether to include psychological analysis

        Returns:
            Character profile object
        """
        # Extract or infer character traits
        background = details.get("background", "")
        personality = self._infer_personality_traits(background, details)

        profile = CharacterProfile(
            name=name,
            archetype=details.get("archetype", "To be determined"),
            personality_traits=personality["traits"],
            motivations=[details.get("motivation", "Personal growth")] if details.get("motivation") else ["Unclear"],
            relationships=details.get("relationships", {}),
            character_arc=[
                {"stage": "Beginning", "state": "Current situation"},
                {"stage": "Middle", "state": "Transformation period"},
                {"stage": "End", "state": "Final state"}
            ]
        )

        if detailed:
            profile.psychological_profile = self._analyze_psychology(background, personality)

        return profile

    def _infer_personality_traits(self, background: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Infer personality traits from background and details."""
        return {
            "traits": ["Determined", "Resilient", "Complex"],
            "strengths": ["Problem-solving", "Adaptability"],
            "flaws": ["Stubbornness", "Risk-taking"],
            "behavior_patterns": ["Action-oriented", "Independent"]
        }

    def _analyze_psychology(self, background: str, personality: Dict[str, Any]) -> Dict[str, Any]:
        """Perform psychological analysis of character."""
        return {
            "psychological_profile": {
                "core_beliefs": ["Self-reliance", "Justice system"],
                "defense_mechanisms": ["Humor", "Rationalization"],
                "motivational_drivers": ["Achievement", "Social harmony"],
                "potential_development": ["Improved emotional intelligence", "Better conflict resolution"]
            },
            "cognitive_patterns": ["Analytical thinking", "Situational awareness"],
            "emotional_resilience": "Moderately resilient with room for growth"
        }

    def _analyze_concept_characters(self, concept: str, detailed: bool = False) -> CharacterProfile:
        """Analyze characters based on story concept."""
        return CharacterProfile(
            name="Protagonist",
            archetype="Hero",
            personality_traits=["Determined", "Complex"],
            motivations=["Self-discovery", "Achievement"],
            relationships={},
            character_arc=[
                {"stage": "Beginning", "state": "Unaware of potential"},
                {"stage": "Middle", "state": "Facing challenges"},
                {"stage": "End", "state": "Transformed"}
            ]
        )

    def _analyze_worldbuilding(
        self,
        concept: str,
        world_details: Optional[Dict[str, Any]],
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze world-building elements and consistency.

        Args:
            concept: Story concept
            world_details: World-building details
            detailed: Whether to include detailed consistency checking

        Returns:
            World-building analysis result
        """
        world_template = WorldBuildingTemplate(
            geographical={
                "settings": world_details.get("settings", []) if world_details else ["Various locations"],
                "climate": "To be defined",
                "terrain": "Diverse landscape"
            },
            societal={
                "cultures": world_details.get("societies", ["Human society"]) if world_details else ["Default society"],
                "governments": ["Mixed governance"],
                "socialstructures": ["Class-based", "Achievement-based"]
            },
            historical_events=[
                {"period": "Ancient", "event": "Foundation events"},
                {"period": "Recent", "event": "Recent significant events"}
            ],
            cultural_elements={
                "traditions": "Rich cultural heritage",
                "languages": "Multiple languages and dialects",
                "taboos": "Societal norms and restrictions"
            }
        )

        if world_details and "magic_system" in world_details:
            world_template.magical_rules = {
                "rules": world_details["magic_system"] + " based magical rules",
                "limitations": ["Energy requirements", "Personal cost"],
                "exceptions": ["Rare phenomena"]
            }

        if detailed:
            world_template.consistency_check = self._check_world_consistency(world_details or {})

        return world_template.dict()

    def _check_world_consistency(self, world_details: Dict[str, Any]) -> Dict[str, Any]:
        """Check world-building elements for consistency."""
        issues = []
        recommendations = []

        # Basic consistency checks
        if "magic_system" in world_details and "technology_level" in world_details:
            magic_system = world_details["magic_system"]
            tech_level = world_details["technology_level"]

            if magic_system == "high" and tech_level == "advanced":
                issues.append("Potential inconsistency: High magic with advanced technology")
                recommendations.append("Define how magic and technology coexist")

        return {
            "issues_found": issues,
            "severity_score": len(issues) * 2,  # Simple scoring
            "recommendations": recommendations,
            "consistency_rating": "Good" if len(issues) < 2 else "Needs attention"
        }

    def _analyze_relationships(
        self,
        concept: str,
        character_details: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze character relationships and social dynamics.

        Args:
            concept: Story concept
            character_details: Character relationship information

        Returns:
            Relationship analysis result
        """
        relationships = {}
        if character_details:
            relationships = self._map_character_relationships(character_details)

        return {
            "relationship_map": relationships,
            "social_dynamics": self._analyze_social_dynamics(relationships),
            "conflict_opportunities": ["Relationship tensions", "Loyalty conflicts", "Betrayals"],
            "development_opportunities": ["Deepened connections", "Reconciliations", "New alliances"]
        }

    def _map_character_relationships(self, character_details: Dict[str, Any]) -> Dict[str, Any]:
        """Create relationship map from character details."""
        character_list = character_details.get("characters", ["Character A", "Character B"])
        relationships = {}

        for char in character_list:
            relationships[char] = {
                "connections": [other for other in character_list if other != char],
                "relationship_types": ["To be defined"],
                "conflicts": ["Potential conflicts"],
                "growth_opportunities": ["Relationship development"]
            }

        return relationships

    def _analyze_social_dynamics(self, relationships: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social dynamics from relationships."""
        return {
            "power_structure": "To be analyzed",
            "alliance_patterns": ["Various alliances"],
            "conflict_types": ["Personal", "Ideological", "Social"],
            "group_dynamics": ["Team interactions", "Social hierarchies"]
        }

    def _analyze_narrative_flow(
        self,
        concept: str,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze narrative flow and pacing.

        Args:
            concept: Story concept
            detailed: Whether to include detailed pacing analysis

        Returns:
            Narrative flow analysis result
        """
        flow_analysis = {
            "story_arc": ["Setup", "Rising Action", "Climax", "Falling Action", "Resolution"],
            "pacing_markers": ["Opening hook", "Major turning points", "Climactic moments"],
            "momentum_changes": ["Building tension", "Emotional peaks", "Resolution rhythm"]
        }

        if detailed:
            flow_analysis.update({
                "rhythm_analysis": self._analyze_narrative_rhythm(concept),
                "tension_mapping": self._map_story_tension(concept),
                "transition_points": self._identify_transitions(concept)
            })

        return flow_analysis

    def _analyze_narrative_rhythm(self, concept: str) -> Dict[str, Any]:
        """Analyze narrative rhythm and pacing."""
        return {
            "scene_lengths": "Varied for effect",
            "tension_peaks": "Strategic placement",
            "emotional_pacing": "Balanced highs and lows"
        }

    def _map_story_tension(self, concept: str) -> List[Dict[str, Any]]:
        """Map story tension levels."""
        return [
            {"section": "Beginning", "tension_level": "Low", "purpose": "Setup"},
            {"section": "Middle", "tension_level": "Rising", "purpose": "Development"},
            {"section": "End", "tension_level": "High", "purpose": "Climax"}
        ]

    def _identify_transitions(self, concept: str) -> List[Dict[str, Any]]:
        """Identify major narrative transitions."""
        return [
            {"from": "Setup", "to": "Rising Action", "type": "Inciting Incident"},
            {"from": "Rising Action", "to": "Climax", "type": "Major Turning Point"},
            {"from": "Climax", "to": "Resolution", "type": "Final Resolution"}
        ]

    def _validate_story_consistency(
        self,
        concept: str,
        character_details: Optional[Dict[str, Any]],
        world_details: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate overall story consistency.

        Args:
            concept: Story concept
            character_details: Character information
            world_details: World-building details

        Returns:
            Validation result object
        """
        issues = []
        recommendations = []

        # Check concept coherence
        if not concept or len(concept.split()) < 5:
            issues.append({"type": "concept", "severity": "critical", "description": "Story concept too vague"})

        # Check character consistency
        if character_details:
            char_issues = self._validate_characters(character_details)
            issues.extend(char_issues)

        # Check world consistency
        if world_details:
            world_issues = self._validate_world(world_details)
            issues.extend(world_issues)

        # Check overall integration
        integration_issues = self._validate_integration(concept, character_details, world_details)
        issues.extend(integration_issues)

        # Generate recommendations
        recommendations = self._generate_validation_recommendations(issues)

        overall_score = max(0, 10 - len(issues) * 1.5)

        return ValidationResult(
            overall_score=overall_score,
            issues_found=issues,
            recommendations=recommendations,
            consistency_report={
                "total_issues": len(issues),
                "critical_issues": len([i for i in issues if i.get("severity") == "critical"]),
                "warnings": len([i for i in issues if i.get("severity") == "warning"])
            }
        )

    def _validate_characters(self, character_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate character consistency."""
        issues = []

        # Check for conflicting character traits
        # This would be more sophisticated in a real implementation
        if "conflicting_personalities" in str(character_details):
            issues.append({
                "type": "character",
                "severity": "warning",
                "description": "Potential conflicting personality traits"
            })

        return issues

    def _validate_world(self, world_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate world-building consistency."""
        issues = []

        # Check for world-building conflicts
        magic_system = world_details.get("magic_system", "")
        if magic_system and "inconsistent" in magic_system.lower():
            issues.append({
                "type": "worldbuilding",
                "severity": "critical",
                "description": "Inconsistent magic system setup"
            })

        return issues

    def _validate_integration(
        self,
        concept: str,
        character_details: Optional[Dict[str, Any]],
        world_details: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate integration between concept, characters, and world."""
        issues = []

        # Check if characters fit the concept
        if "mismatched_characters" in str(concept).lower() and character_details:
            issues.append({
                "type": "integration",
                "severity": "warning",
                "description": "Character profiles may not fit story concept"
            })

        return issues

    def _generate_validation_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate specific recommendations based on validation issues."""
        recommendations = []

        for issue in issues:
            issue_type = issue.get("type", "")
            if issue_type == "concept":
                recommendations.append("Develop a clearer, more detailed story concept")
            elif issue_type == "character":
                recommendations.append("Resolve conflicting character personality traits")
            elif issue_type == "worldbuilding":
                recommendations.append("Define consistent rules for your world-building elements")
            elif issue_type == "integration":
                recommendations.append("Ensure characters and world elements align with story concept")

        if not recommendations:
            recommendations.append("Consider adding more detail to strengthen story elements")

        return list(set(recommendations))  # Remove duplicates

    def _generate_output(
        self,
        result: Dict[str, Any],
        output_type: OutputType,
        template_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate output in requested format.

        Args:
            result: Analysis result
            output_type: Desired output format
            template_file: Template file path

        Returns:
            Formatted output dictionary
        """
        if output_type == OutputType.OUTLINE:
            return self._generate_outline_output(result, template_file)
        elif output_type == OutputType.PROFILE:
            return self._generate_profile_output(result, template_file)
        elif output_type == OutputType.ANALYSIS:
            return result  # Already in analysis format
        elif output_type == OutputType.TEMPLATE:
            return self._generate_template_output(result, template_file)
        else:  # VALIDATION
            return self._generate_validation_output(result, template_file)

    def _generate_outline_output(
        self,
        result: Dict[str, Any],
        template_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured outline output."""
        outline = {
            "title": "Story Outline",
            "structure": [],
            "chapters": [],
            "key_elements": []
        }

        # Extract structure from analysis
        if "plot_structure" in result:
            structure = result["plot_structure"]
            outline["structure"] = structure.get("structure", {})
            outline["key_elements"] = structure.get("key_elements", [])

        outline["summary"] = result.get("story_concept", "")

        return outline

    def _generate_profile_output(
        self,
        result: Dict[str, Any],
        template_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate character profile output."""
        profile = {
            "title": "Character Development Profile",
            "profiles": [],
            "relationships": {},
            "development_arcs": []
        }

        # Extract character information
        if "character_development" in result:
            char_dev = result["character_development"]
            profile["profiles"] = char_dev.get("character_profiles", [])

        if "relationships" in result:
            profile["relationships"] = result["relationships"]["relationship_map"]

        return profile

    def _generate_template_output(
        self,
        result: Dict[str, Any],
        template_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate template output."""
        template = {
            "title": "Story Structure Template",
            "framework": "",
            "elements": [],
            "guidelines": []
        }

        # Extract framework information
        if "plot_structure" in result:
            template["framework"] = result["plot_structure"].get("framework", "")
            template["elements"] = result["plot_structure"].get("key_elements", [])

        template["guidelines"] = [
            "Follow the framework structure",
            "Ensure consistent pacing",
            "Develop character arcs appropriately",
            "Maintain story consistency"
        ]

        return template

    def _generate_validation_output(
        self,
        result: Dict[str, Any],
        template_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate validation output."""
        return {
            "title": "Story Validation Report",
            "validation": result.get("consistency", {}),
            "issues": result.get("consistency", {}).get("issues_found", []),
            "recommendations": result.get("consistency", {}).get("recommendations", [])
        }

    def _save_output(self, output: Dict[str, Any], output_file: str):
        """
        Save output to file using FileWriteTool.

        Args:
            output: Output data to save
            output_file: Path to save file
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write output to file
            # Note: In a real implementation, this would use FileWriteTool
            # For now, we'll use standard Python file operations
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise Exception(f"Failed to save output to {output_file}: {str(e)}")

    def _load_template(self, template_file: str) -> Dict[str, Any]:
        """
        Load template file using FileReadTool.

        Args:
            template_file: Path to template file

        Returns:
            Template data as dictionary
        """
        try:
            # Note: In a real implementation, this would use FileReadTool
            # For now, we'll use standard Python file operations
            template_path = Path(template_file)

            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Return default template
                return {
                    "template_name": "default",
                    "structure": "basic",
                    "elements": []
                }

        except Exception as e:
            raise Exception(f"Failed to load template {template_file}: {str(e)}")