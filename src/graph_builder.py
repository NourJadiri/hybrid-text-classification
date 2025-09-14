"""
Dynamic graph builder for the LangGraph classification system.
Creates graphs based on configuration settings.
"""

from typing import Any, Dict, List, NotRequired
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from config_loader import ClassificationConfig
from label_info import flatten_taxonomy, load_taxonomy
from state import ClassificationState
from graph_nodes import (
    classify_category_node,
    handle_other_category_node,
    classify_narratives_node,
    validate_narratives_node,
    clean_narratives_node,
    classify_subnarratives_node,
    validate_subnarratives_node,
    clean_subnarratives_node,
    write_results_node,
    handle_empty_narratives_node
)
from graph_utils import route_after_category


class ConfigurableGraphBuilder:
    """Builds LangGraph classification graphs based on configuration."""
    
    def __init__(self, config: ClassificationConfig):
        """
        Initialize the graph builder with configuration.
        
        Args:
            config: Classification configuration
        """
        self.config = config
        self.config.validate()
        
        self.taxonomy = load_taxonomy()
        self.flat_narratives, self.flat_subnarratives = flatten_taxonomy(self.taxonomy)
        
        self.llms = self._initialize_llms(config)
        
        print(f"[GraphBuilder] Initialized with config:")
        print(f"  Default Model: {config.model_name}")
        print(f"  Input: {config.input_folder}")
        print(f"  Output: {config.output_file}")
        print(f"  Global Validation: {config.enable_validation}")
        print(f"  Narrative Validation: {config.is_narrative_validation_enabled()}")
        print(f"  Subnarrative Validation: {config.is_subnarrative_validation_enabled()}")
        print(f"  Cleaning: {config.enable_cleaning}")
        self._print_model_assignments()
    
    def _initialize_llms(self, config: ClassificationConfig) -> Dict[str, Any]:
        """Initialize LLMs for each node type and operation."""
        llms = {}
        
        llms['category_classification'] = init_chat_model(
            config.get_model_for_node('category', 'classification')
        )
        llms['category_validation'] = init_chat_model(
            config.get_model_for_node('category', 'validation')
        )
        
        llms['narratives_classification'] = init_chat_model(
            config.get_model_for_node('narratives', 'classification')
        )
        llms['narratives_validation'] = init_chat_model(
            config.get_model_for_node('narratives', 'validation')
        )
        
        llms['subnarratives_classification'] = init_chat_model(
            config.get_model_for_node('subnarratives', 'classification')
        )
        llms['subnarratives_validation'] = init_chat_model(
            config.get_model_for_node('subnarratives', 'validation')
        )
        
        return llms
    
    def _print_model_assignments(self):
        """Print model assignments for each node."""
        print("[GraphBuilder] Model assignments:")
        for key, llm in self.llms.items():
            node_type, operation = key.split('_', 1)
            # Different LLM classes have different attribute names for the model
            model_name = getattr(llm, 'model_name', getattr(llm, 'model', str(llm)))
            print(f"  {node_type.capitalize()} {operation}: {model_name}")
    
    def _create_category_node(self):
        """Create category classification node wrapper."""
        def _classify_category_node(state: ClassificationState) -> dict:
            return classify_category_node(state, self.llms['category_classification'])
        return _classify_category_node
    
    def _create_other_category_node(self):
        """Create other category handler node wrapper."""
        def _handle_other_category_node(state: ClassificationState) -> dict:
            return handle_other_category_node(state)
        return _handle_other_category_node
    
    def _create_empty_narratives_node(self):
        """Create empty narratives handler node wrapper."""
        def _handle_empty_narratives_node(state: ClassificationState) -> dict:
            return handle_empty_narratives_node(state)
        return _handle_empty_narratives_node
    
    def _create_narratives_node(self):
        """Create narratives classification node wrapper."""
        def _classify_narratives_node(state: ClassificationState) -> dict:
            return classify_narratives_node(state, self.llms['narratives_classification'])
        return _classify_narratives_node
    
    def _create_validate_narratives_node(self):
        """Create narratives validation node wrapper."""
        def _validate_narratives_node(state: ClassificationState) -> dict:
            return validate_narratives_node(state, self.llms['narratives_validation'], self.taxonomy)
        return _validate_narratives_node
    
    def _create_clean_narratives_node(self):
        """Create narratives cleaning node wrapper."""
        def _clean_narratives_node(state: ClassificationState) -> dict:
            return clean_narratives_node(state, self.flat_narratives)
        return _clean_narratives_node
    
    def _create_subnarratives_node(self):
        """Create subnarratives classification node wrapper."""
        async def _classify_subnarratives_node(state: ClassificationState) -> dict:
            return await classify_subnarratives_node(state, self.llms['subnarratives_classification'])
        return _classify_subnarratives_node
    
    def _create_validate_subnarratives_node(self):
        """Create subnarratives validation node wrapper."""
        def _validate_subnarratives_node(state: ClassificationState) -> dict:
            return validate_subnarratives_node(state, self.llms['subnarratives_validation'], self.taxonomy)
        return _validate_subnarratives_node
    
    def _create_clean_subnarratives_node(self):
        """Create subnarratives cleaning node wrapper."""
        def _clean_subnarratives_node(state: ClassificationState) -> dict:
            return clean_subnarratives_node(state, self.flat_subnarratives)
        return _clean_subnarratives_node
    
    def _create_results_node(self):
        """Create results writing node wrapper."""
        def _write_results_node(state: ClassificationState) -> dict:
            result = write_results_node(state, self.config.output_file)
            return result if result is not None else {}
        return _write_results_node
    
    def _create_routing_functions(self):
        """Create routing functions for the graph."""
        def route_after_category_with_config(state: ClassificationState) -> str:
            category = state.get("category")
            if category == "Other":
                return "handle_other_category"
            else:
                return "narratives"
        
        def route_after_narrative_validation(state: ClassificationState) -> str:
            """Route based on narrative validation feedback."""
            feedback = state.get("narrative_validation_feedback", "")
            retry_count = state.get("narrative_retry_count", 0)
            
            if feedback == "approved" or retry_count >= 3:
                if self.config.enable_cleaning:
                    return "clean_narratives"
                else:
                    return "subnarratives"
            else:
                return "narratives"
        
        def route_after_subnarrative_validation(state: ClassificationState) -> str:
            """Route based on subnarrative validation feedback."""
            feedback = state.get("subnarrative_validation_feedback", "")
            retry_count = state.get("subnarrative_retry_count", 0)
            
            if feedback == "approved" or retry_count >= 3:
                if self.config.enable_cleaning:
                    return "clean_subnarratives"
                else:
                    return "write_results"
            else:
                return "subnarratives"
        
        def route_after_clean_narratives(state: ClassificationState) -> str:
            """Route after cleaning narratives - check if empty."""
            narratives = state.get("narratives", [])
            if not narratives or all(n.narrative_name == "Other" for n in narratives):
                return "handle_empty_narratives"
            else:
                return "subnarratives"
        
        return {
            "route_after_category": route_after_category_with_config,
            "route_after_narrative_validation": route_after_narrative_validation,
            "route_after_subnarrative_validation": route_after_subnarrative_validation,
            "route_after_clean_narratives": route_after_clean_narratives
        }
    
    def build_graph(self):
        """
        Build the classification graph based on configuration.
        
        Returns:
            Compiled StateGraph
        """
        print(f"[GraphBuilder] Building classification graph...")
        
        builder = StateGraph(ClassificationState)
        
        builder.add_node("categories", self._create_category_node())
        builder.add_node("handle_other_category", self._create_other_category_node())
        builder.add_node("handle_empty_narratives", self._create_empty_narratives_node())
        builder.add_node("narratives", self._create_narratives_node())
        builder.add_node("subnarratives", self._create_subnarratives_node())
        builder.add_node("write_results", self._create_results_node())
        
        # Conditionally add validation nodes based on granular settings
        if self.config.is_narrative_validation_enabled():
            print("[GraphBuilder] Adding narrative validation node")
            builder.add_node("validate_narratives", self._create_validate_narratives_node())
        
        if self.config.is_subnarrative_validation_enabled():
            print("[GraphBuilder] Adding subnarrative validation node")
            builder.add_node("validate_subnarratives", self._create_validate_subnarratives_node())
        
        if self.config.enable_cleaning:
            print("[GraphBuilder] Adding cleaning nodes")
            builder.add_node("clean_narratives", self._create_clean_narratives_node())
            builder.add_node("clean_subnarratives", self._create_clean_subnarratives_node())
        
        self._build_graph_flow(builder)
        
        graph = builder.compile()
        print("[GraphBuilder] Graph compiled successfully")
        
        return graph
    
    def _build_graph_flow(self, builder: StateGraph) -> None:
        """Build the graph flow based on configuration."""
        routing_functions = self._create_routing_functions()
        
        builder.add_edge(START, "categories")
        builder.add_conditional_edges("categories", routing_functions["route_after_category"])
        
        builder.add_edge("handle_other_category", "write_results")
        
        builder.add_edge("handle_empty_narratives", "write_results")
        
        # Build narrative flow based on granular configuration
        narrative_validation = self.config.is_narrative_validation_enabled()
        subnarrative_validation = self.config.is_subnarrative_validation_enabled()
        cleaning = self.config.enable_cleaning
        
        # Narrative flow
        if narrative_validation:
            builder.add_edge("narratives", "validate_narratives")
            builder.add_conditional_edges("validate_narratives", routing_functions["route_after_narrative_validation"])
        else:
            # Direct to next stage
            if cleaning:
                builder.add_edge("narratives", "clean_narratives")
            else:
                builder.add_edge("narratives", "subnarratives")
        
        # Clean narratives flow (if enabled)
        if cleaning:
            builder.add_conditional_edges("clean_narratives", routing_functions["route_after_clean_narratives"])
        
        # Subnarrative flow
        if subnarrative_validation:
            builder.add_edge("subnarratives", "validate_subnarratives")
            builder.add_conditional_edges("validate_subnarratives", routing_functions["route_after_subnarrative_validation"])
        else:
            # Direct to next stage
            if cleaning:
                builder.add_edge("subnarratives", "clean_subnarratives")
            else:
                builder.add_edge("subnarratives", "write_results")
        
        # Clean subnarratives flow (if enabled)
        if cleaning:
            builder.add_edge("clean_subnarratives", "write_results")
        
        builder.add_edge("write_results", END)
        
        print(f"[GraphBuilder] Graph flow configured:")
        print(f"  - Narrative validation: {narrative_validation}")
        print(f"  - Subnarrative validation: {subnarrative_validation}")
        print(f"  - Cleaning enabled: {cleaning}")
    
    def get_execution_config(self) -> dict:
        """Get execution configuration for the graph."""
        return {
            "max_concurrency": self.config.max_concurrency
        }


def build_graph_from_config(config_path: str):
    """
    Build a classification graph from a configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Tuple of (compiled_graph, execution_config, loaded_config)
    """
    from pathlib import Path
    config = ClassificationConfig.from_yaml(Path(config_path))
    builder = ConfigurableGraphBuilder(config)
    graph = builder.build_graph()
    execution_config = builder.get_execution_config()
    
    return graph, execution_config, config
