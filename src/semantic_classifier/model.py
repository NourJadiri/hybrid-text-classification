import torch
from transformers import AutoConfig, AutoModel

class NarrativesClassifier(torch.nn.Module):
    
    def __init__(self, model_name: str, label_embeddings: torch.Tensor, pos_weights: torch.Tensor):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.register_buffer("label_embeddings", label_embeddings)
        self.register_buffer("pos_weights", pos_weights)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = outputs.last_hidden_state[:, 0, :]
        text_embedding = self.dropout(text_embedding)
        
        text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=1)
        label_embeddings = torch.nn.functional.normalize(self.label_embeddings, p=2, dim=1)
        
        logits = torch.matmul(text_embedding, label_embeddings.t())
        
        loss = None
        if labels is not None:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
            loss = loss_fn(logits, labels.float())
            return {
                "loss": loss,
                "logits": logits
            }
        
        return logits
    
    def gradient_checkpointing_enable(self):
        self.bert.gradient_checkpointing_enable()
        