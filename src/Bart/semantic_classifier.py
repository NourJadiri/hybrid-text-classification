
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class SemanticClassifier(nn.Module):
    """
    A custom model for meaning-aware hierarchical classification.

    This model uses a pre-trained transformer to create a rich embedding of the
    input text. It then calculates the cosine similarity between this text
    embedding and pre-computed embeddings of the hierarchical labels.
    """
    def __init__(self, model_name: str, label_embeddings: torch.Tensor, pos_weights: torch.Tensor):
        """
        Initializes the SemanticClassifier.

        Args:
            model_name (str): The name of the pre-trained transformer model 
                              from the Hugging Face Hub (e.g., 'bert-base-uncased').
            label_embeddings (torch.Tensor): A 2D tensor of shape 
                                             [num_labels, embedding_dim] containing the
                                             pre-computed embeddings for each label.
        """
        super().__init__()
        
        # 1. Load the configuration and the base transformer model.
        # We use AutoModel to get the raw hidden states, without any specific
        # classification head attached. This is our "text understanding" engine.
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        
        # 2. Register the label embeddings as a non-trainable buffer.
        # A "buffer" is a tensor that is part of the model's state (it gets
        # saved and moved to devices like a GPU), but it is not considered
        # a model parameter, so the optimizer will not update it during training.
        # This is perfect for our static, pre-computed label meanings.
        self.register_buffer("label_embeddings", label_embeddings)
        self.register_buffer("pos_weights", pos_weights)
        
        # 3. Add a dropout layer for regularization.
        # This helps prevent overfitting by randomly zeroing out some of the
        # activations during training.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Defines the forward pass of the model.
        """
        # 1. Get text embeddings from the base transformer.
        # The output contains the hidden states for all tokens.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Create a single vector representation for the entire input text.
        # A common and effective strategy is to use the hidden state of the
        # special [CLS] token, which is always the first token in the sequence.
        # Shape: [batch_size, hidden_size]
        text_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout to the text embedding for regularization.
        text_embedding = self.dropout(text_embedding)
        
        # 3. The Core Logic: Semantic Matching.
        # We normalize both the text and label embeddings to unit length.
        # The dot product of two unit vectors is their cosine similarity.
        text_embedding_norm = F.normalize(text_embedding, p=2, dim=1)
        label_embeddings_norm = F.normalize(self.label_embeddings, p=2, dim=1)  # type: ignore[arg-type]
        
        # Calculate the cosine similarity between the text embedding and *all*
        # label embeddings. This is done efficiently with a matrix multiplication.
        # The result is our "logits" - a raw score for each label.
        # Shape: [batch_size, num_labels]
        logits = torch.matmul(text_embedding_norm, label_embeddings_norm.t())
        
        # 4. Calculate the loss if labels are provided (i.e., during training).
        loss = None
        if labels is not None:
            # For multi-label classification, the appropriate loss function is
            # Binary Cross-Entropy with Logits. It's numerically stable and
            # handles the sigmoid activation and loss calculation in one step.
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
            loss = loss_fct(logits, labels.float())

        # Return a dictionary, which is compatible with the Hugging Face Trainer.
        return {
            "loss": loss,
            "logits": logits
        }