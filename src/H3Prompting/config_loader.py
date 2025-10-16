"""
Configuration loader for the LangGraph classification system.
Handles loading and validation of YAML configuration files.
"""

import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ClassificationConfig:
    """Configuration schema for the classification system."""
    
    model_name: str
    
    input_folder: str
    output_file: str
    
    enable_validation: bool = True  # Global validation flag (backward compatibility)
    enable_cleaning: bool = True    # Global cleaning flag (backward compatibility)
    
    # Granular validation flags (optional - override global setting)
    enable_narrative_validation: Optional[bool] = None
    enable_subnarrative_validation: Optional[bool] = None

    # Optional text cleaning stage
    enable_text_cleaning: bool = False
    
    # Multi-agent classification settings
    num_agents: int = 1  # DEPRECATED: Use num_narrative_agents and num_subnarrative_agents instead
    num_narrative_agents: int = 1  # Number of agents for narrative classification
    num_subnarrative_agents: int = 1  # Number of agents for subnarrative classification
    aggregation_method: str = "union"  # "union" or "intersection"
    
    max_concurrency: int = 20
    
    category: Optional[Dict[str, Any]] = None
    narratives: Optional[Dict[str, Any]] = None
    subnarratives: Optional[Dict[str, Any]] = None
    cleaning: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'ClassificationConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            ClassificationConfig instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        required_fields = ['model_name', 'input_folder', 'output_file']
        for field in required_fields:
            if field not in yaml_data:
                raise ValueError(f"Required field '{field}' missing from configuration")
        
        # Allow templated output_file using placeholders like {model_name} or {model}
        raw_output = yaml_data['output_file']
        # sanitize model name for filesystem use
        raw_model = yaml_data['model_name']
        safe_model = raw_model.replace(':', '_').replace('/', '-').replace(' ', '-')
        # support both {model_name} and {model} placeholders
        formatted_output = raw_output.format(model_name=safe_model, model=safe_model)

        # Handle backward compatibility: if num_agents is set but not the specific ones, use it for both
        num_agents = yaml_data.get('num_agents', 1)
        num_narrative_agents = yaml_data.get('num_narrative_agents', num_agents)
        num_subnarrative_agents = yaml_data.get('num_subnarrative_agents', 1)  # Default to 1 for subnars
        
        return cls(
            model_name=yaml_data['model_name'],
            input_folder=yaml_data['input_folder'],
            output_file=formatted_output,
            enable_validation=yaml_data.get('enable_validation', True),
            enable_cleaning=yaml_data.get('enable_cleaning', True),
            max_concurrency=yaml_data.get('max_concurrency', 20),
            num_agents=num_agents,  # Keep for backward compatibility
            num_narrative_agents=num_narrative_agents,
            num_subnarrative_agents=num_subnarrative_agents,
            aggregation_method=yaml_data.get('aggregation_method', 'union'),
            category=yaml_data.get('category'),
            narratives=yaml_data.get('narratives'),
            subnarratives=yaml_data.get('subnarratives'),
            enable_narrative_validation=yaml_data.get('enable_narrative_validation'),
            enable_subnarrative_validation=yaml_data.get('enable_subnarrative_validation'),
            enable_text_cleaning=yaml_data.get('enable_text_cleaning', False),
            cleaning=yaml_data.get('cleaning')
        )
    
    def validate(self) -> None:
        """Validate the configuration values."""
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        
        if not self.input_folder.strip():
            raise ValueError("input_folder cannot be empty")
        
        if not self.output_file.strip():
            raise ValueError("output_file cannot be empty")
        
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        
        input_path = Path(self.input_folder)
        if not input_path.exists():
            raise ValueError(f"Input folder does not exist: {self.input_folder}")
        
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def get_model_for_node(self, node_type: str, operation: str) -> str:
        """
        Get the model name for a specific node and operation.
        
        Args:
            node_type: 'category', 'narratives', or 'subnarratives'
            operation: 'classification' or 'validation'
            
        Returns:
            Model name to use (falls back to default if not specified)
        """
        node_config = getattr(self, node_type, None)
        if node_config and operation in node_config:
            return node_config[operation].get('model', self.model_name)
        return self.model_name
    
    def is_narrative_validation_enabled(self) -> bool:
        """Check if narrative validation is enabled."""
        if self.enable_narrative_validation is not None:
            return self.enable_narrative_validation
        return self.enable_validation
    
    def is_subnarrative_validation_enabled(self) -> bool:
        """Check if subnarrative validation is enabled."""
        if self.enable_subnarrative_validation is not None:
            return self.enable_subnarrative_validation
        return self.enable_validation
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        per_node_info = ""
        if self.category or self.narratives or self.subnarratives or self.cleaning:
            per_node_info = "\n  Per-node models:"
            for node in ['category', 'narratives', 'subnarratives', 'cleaning']:
                node_config = getattr(self, node, None)
                if node_config:
                    per_node_info += f"\n    {node}: {node_config}"
        
        # Add granular validation info
        validation_info = ""
        if (self.enable_narrative_validation is not None or 
            self.enable_subnarrative_validation is not None):
            validation_info = "\n  Granular validation:"
            validation_info += f"\n    narratives: {self.is_narrative_validation_enabled()}"
            validation_info += f"\n    subnarratives: {self.is_subnarrative_validation_enabled()}"
        
        return (
            f"ClassificationConfig(\n"
            f"  model_name='{self.model_name}',\n"
            f"  input_folder='{self.input_folder}',\n"
            f"  output_file='{self.output_file}',\n"
            f"  enable_validation={self.enable_validation},\n"
            f"  enable_cleaning={self.enable_cleaning},\n"
            f"  max_concurrency={self.max_concurrency}{validation_info}{per_node_info}\n"
            f")"
        )


def create_example_config(output_path: str = "config.yaml") -> None:
    """
    Create an example configuration file.
    
    Args:
        output_path: Path where to save the example config
    """
    example_config = {
        'model_name': 'openai:gpt-5-nano',
        'input_folder': 'testset/EN/subtask-2-documents/',
        'output_file': 'results/experiment_results.txt',
        'enable_validation': True,
        'enable_cleaning': True,
        'max_concurrency': 20
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Example configuration saved to: {output_path}")


if __name__ == "__main__":
    create_example_config()
