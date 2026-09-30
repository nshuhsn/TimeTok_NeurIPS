import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VARAutoregressiveWrapper(nn.Module):
    """
    Wrapper for VAR models that handles both training and generation.
    
    Training: Uses next-scale prediction (coarse-to-fine)
    Generation: Can do either VAR-style (scale-by-scale) or standard AR
    """
    def __init__(self, net, ignore_index=-100, pad_value=0):
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.ignore_index = ignore_index
        self.pad_value = pad_value

    def forward(self, x, y_true, dataset_id, return_loss_dict=False):
        """
        Forward pass for training using next-scale prediction.
        
        Args:
            x: [batch, seq_len] token sequence (including BOS)
            y_true: [batch] class labels
            dataset_id: [batch] dataset IDs
            return_loss_dict: bool, return per-scale losses
        
        Returns:
            loss: scalar or (loss, loss_dict) if return_loss_dict=True
        """
        # Use the model's forward which implements next-scale prediction
        return self.net(x, y_true, dataset_id, return_loss_dict=return_loss_dict)

    @torch.no_grad()
    def generate_var_style(
        self,
        start_tokens,
        y_true,
        dataset_id,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        """
        VAR-style generation: generate scale-by-scale (coarse-to-fine).
        All tokens within a scale are generated in parallel.
        
        Args:
            start_tokens: [batch, prompt_len] starting tokens.
            y_true: [batch] class labels
            dataset_id: [batch] dataset IDs
            temperature: float, sampling temperature
            top_k: int, top-k sampling
            top_p: float, nucleus sampling
        
        Returns:
            generated: [batch, full_seq_len - prompt_len] generated sequence
        """
        b, prompt_len = start_tokens.shape
        device = start_tokens.device
        mask_id = self.net.mask_id
        
        # Start with prompt
        generated = start_tokens.clone()
        current_scale_id_list = self.net.get_scale_id_from_position(torch.arange(prompt_len, device=device), has_bos=True)
        if current_scale_id_list[-1] == self.net.bos_scale_id:
            current_scale_idx = 0
        else:
            current_scale_idx = current_scale_id_list[-1].item() + 1
        # Generate scale by scale
        for scale_idx in range(current_scale_idx, len(self.net.scale_lengths)):
            if scale_idx == 0:
                mask_length = self.net.scale_lengths[scale_idx]
            else:
                mask_length = self.net.scale_lengths[scale_idx] - self.net.scale_lengths[scale_idx-1]
            
            # Create input with mask_id for the entire new scale
            mask_tokens = torch.full((b, mask_length), mask_id, dtype=torch.long, device=device)
            x_input = torch.cat([generated, mask_tokens], dim=1)
            
            # Get scale IDs
            positions = torch.arange(x_input.size(1), device=device)
            scale_ids = self.net.get_scale_id_from_position(positions, has_bos=True)
            scale_ids = scale_ids.unsqueeze(0).expand(b, -1)
            
            # Forward pass - generates all tokens at this scale simultaneously
            logits = self.net.var(x_input, y_true, dataset_id, scale_ids)
            
            # Extract logits for the new scale positions
            mask_logits = logits[:, -mask_length:, :]
            
            # Sample all tokens at this scale
            predicted_tokens = self._sample_tokens(mask_logits, temperature, top_k, top_p)
            
            # Append to generated sequence
            generated = torch.cat([generated, predicted_tokens], dim=1)
        
        return generated[:, prompt_len:]
    
    def _sample_tokens(self, logits, temperature, top_k, top_p):
        """
        Sample tokens from logits with temperature, top-k, and top-p.
        
        Args:
            logits: [batch, seq_len, vocab_size]
            
        Returns:
            samples: [batch, seq_len]
        """
        b, n, v = logits.shape
        
        # Apply temperature
        logits = logits / temperature
        
        # Flatten for easier processing
        logits_flat = logits.reshape(-1, v)
        
        # Top-k sampling
        if top_k is not None:
            v_topk, _ = torch.topk(logits_flat, min(top_k, v))
            logits_flat[logits_flat < v_topk[:, [-1]]] = -float('Inf')
        
        # Top-p sampling
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits_flat, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits_flat[indices_to_remove] = -float('Inf')
        
        # Sample
        probs = F.softmax(logits_flat, dim=-1)
        samples = torch.multinomial(probs, 1).squeeze(-1)
        
        # Reshape back
        samples = samples.reshape(b, n)
        
        return samples
