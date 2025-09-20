from multiprocessing import pool
import torch
import torch.nn as nn
from transformers import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.modeling_outputs import SequenceClassifierOutput
import re

def sanitize_name(name):
    """Creates a safe name for dictionary keys. Must be IDENTICAL to the one in preprocessing."""
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name.lower()

class MultiHeadDebertaForHierarchicalClassification(DebertaV2PreTrainedModel):
    
    def __init__(self, config, parent_label2id, child_label_maps, pos_weight_dict = None):
        super().__init__(config)
        
        self.parent_label2id = parent_label2id
        self.child_label_maps = child_label_maps
        self.pos_weight_dict = pos_weight_dict
        self.num_parent_labels = len(parent_label2id)
        
        self.deberta = DebertaV2Model(config)
        
        self.parent_classifier = nn.Linear(config.hidden_size, self.num_parent_labels)
        
        self.child_classifiers = nn.ModuleDict()
        for parent_name, child_map in child_label_maps.items():
            num_child_labels = len(child_map['label2id'])
            self.child_classifiers[sanitize_name(parent_name)] = nn.Linear(config.hidden_size, num_child_labels)
            
        self.post_init()
        
    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, parent_labels = None, **kwargs):
        outputs=self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        pooled_output = outputs[0][:,0]
        
        parent_logits = self.parent_classifier(pooled_output)
        
        child_logits = {}
        
        for sanitized_key, child_head in self.child_classifiers.items():
            child_logits[sanitized_key] = child_head(pooled_output)

        loss = None
        if parent_labels is not None:
            total_loss = 0
            device = parent_labels.device
            
            parents_weights = self.pos_weight_dict['parent_labels'].to(device) if self.pos_weight_dict and 'parent_labels' in self.pos_weight_dict else None
            parent_loss_fct = nn.BCEWithLogitsLoss(pos_weight=parents_weights)
            parent_loss = parent_loss_fct(parent_logits, parent_labels.float())
            total_loss += parent_loss

            for original_parent_name, child_info in self.child_label_maps.items():
                sanitized_name = sanitize_name(original_parent_name)
                child_label_key = f"child_labels_{sanitized_name}"

                if child_label_key in kwargs:
                    child_label_tensor = kwargs[child_label_key]

                    parent_id = self.parent_label2id[original_parent_name]
                    parent_mask = parent_labels[:, parent_id] > 0
                    
                    if parent_mask.sum() > 0:
                        active_child_logits = child_logits[sanitized_name][parent_mask]
                        active_child_labels = child_label_tensor[parent_mask]
                        
                        child_weights = self.pos_weight_dict[child_label_key].to(device) if self.pos_weight_dict and child_label_key in self.pos_weight_dict else None
                        
                        loss_fct_child = nn.BCEWithLogitsLoss(pos_weight=child_weights)
                        child_loss = loss_fct_child(active_child_logits, active_child_labels.float())
                        total_loss += child_loss
                        
            loss = total_loss
            
        return {
            "loss": loss,
            "parent_logits": parent_logits,
            "child_logits": child_logits,
        }