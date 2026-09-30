import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x, eps=1e-8):
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return norm_x * self.gamma


class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization with modulation.
    Takes conditioning signal and outputs scale and shift parameters.
    """
    def __init__(self, conditioning_dim, num_modulations):
        super().__init__()
        self.linear = nn.Linear(conditioning_dim, num_modulations, bias=True)
        # Initialize to zero for stable training (AdaLN-Zero)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, conditioning):
        """
        Args:
            conditioning: [batch_size, conditioning_dim]
        Returns:
            modulations: [batch_size, num_modulations]
        """
        return self.linear(conditioning)


class FeedForward(nn.Module):
    """Feed forward network with SwiGLU activation"""
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class VARAttention(nn.Module):
    """
    VAR-style causal attention with multi-scale masking.
    - Tokens attend to ALL tokens at previous (coarser) scales
    - Tokens attend to ALL tokens within the same scale (bidirectional)
    - Tokens do NOT attend to tokens at future (finer) scales
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bos_scale_id=-1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.bos_scale_id = bos_scale_id
        inner_dim = dim_head * heads
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def create_var_causal_mask(self, scale_ids):
        """
        Create VAR-style causal mask based on scale IDs.
        
        Args:
            scale_ids: [seq_len] - scale index for each position
            
        Returns:
            mask: [seq_len, seq_len] - True where attention should be masked
        """
        b, n = scale_ids.shape
        device = scale_ids.device
        
        # Create scale matrix: scale_ids[i] for each row
        scale_i = scale_ids.unsqueeze(2).expand(b, n, n)
        scale_j = scale_ids.unsqueeze(1).expand(b, n, n)
        
        # Mask out future scales: position i can only attend to position j if scale_j <= scale_i
        # True means masked (blocked), False means allowed
        mask = scale_j > scale_i
        
        
        # Allow all tokens to attend to BOS (scale_id == -1)
        bos_mask = (scale_j == self.bos_scale_id)
        mask = mask & ~bos_mask
        return mask

    def forward(self, x, scale_ids=None):
        """
        Args:
            x: [batch_size, seq_len, dim]
            scale_ids: [seq_len] - scale index for each position (required for VAR masking)
        """
        b, n, d = x.shape
        h = self.heads
        
        # Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Attention scores
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # Apply VAR causal mask. VAR-style: attend to all previous scales + same scale
        var_mask = self.create_var_causal_mask(scale_ids)
        var_mask = var_mask.unsqueeze(1)
        sim = sim.masked_fill(var_mask, -torch.finfo(sim.dtype).max)
        
        # Softmax and apply to values
        attn = F.softmax(sim, dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Merge heads and project
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class VARTransformerBlock(nn.Module):
    """
    VAR Transformer block with AdaLN conditioning.
    Uses adaptive layer normalization to condition on class/dataset information.
    """
    def __init__(self, dim, heads=8, dim_head=64, ff_mult=4, dropout=0.0, bos_scale_id=-1):
        super().__init__()
        self.dim = dim
        self.bos_scale_id = bos_scale_id
        # Attention and feedforward
        self.attn = VARAttention(dim, heads, dim_head, dropout, bos_scale_id=bos_scale_id)
        self.ff = FeedForward(dim, ff_mult, dropout)
        
        # Layer norms
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # AdaLN modulation layers
        # We need 6 parameters: scale and shift for norm1, gate for attn, scale and shift for norm2, gate for ff
        self.adaLN_modulation = AdaLNModulation(dim, num_modulations=6 * dim)

    def forward(self, x, conditioning, scale_ids=None):
        """
        Args:
            x: [batch_size, seq_len, dim]
            conditioning: [batch_size, conditioning_dim]
            scale_ids: [seq_len] optional scale indices for VAR masking
        """
        # Get modulation parameters from conditioning
        modulations = self.adaLN_modulation(conditioning)
        if len(modulations.shape) == 1:
            modulations = modulations.unsqueeze(0)
        modulations = rearrange(modulations, 'b (n d) -> b n d', n=6, d=self.dim)
        
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = modulations.unbind(dim=1)
        
        # Self attention with AdaLN modulation
        normed = self.norm1(x)
        # Apply scale and shift
        normed = normed * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        attn_out = self.attn(normed, scale_ids=scale_ids)
        # Apply gate (scale the residual contribution)
        x = x + gate_attn.unsqueeze(1) * attn_out
        
        # Feed forward with AdaLN modulation
        normed = self.norm2(x)
        # Apply scale and shift
        normed = normed * (1 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
        ff_out = self.ff(normed)
        # Apply gate
        x = x + gate_ff.unsqueeze(1) * ff_out
        
        return x


class VARTransformer(nn.Module):
    """
    Visual Autoregressive Transformer with multi-scale generation.
    Adapted for sequence modeling with AdaLN conditioning.
    """
    def __init__(
        self,
        num_tokens,
        dim=512,
        depth=8,
        heads=8,
        dim_head=64,
        ff_mult=4,
        max_class_num=30,
        max_dataset_num=18,
        max_seq_len=2048,
        dropout=0.0,
        condition_on_class=True,
        condition_on_dataset=False,
        num_scales=4,
        bos_scale_id=-1,
        classifier_free_guidance_prob=0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_scales = num_scales
        self.bos_scale_id = bos_scale_id
        self.condition_on_class = condition_on_class
        self.condition_on_dataset = condition_on_dataset
        self.classifier_free_guidance_prob = classifier_free_guidance_prob
        # Token embeddings
        self.token_emb = nn.Embedding(num_tokens, dim)
        
        # Positional embeddings (learnable)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        # Scale embeddings (for multi-scale modeling)
        self.scale_emb = nn.Embedding(num_scales, dim)

        has_condition = condition_on_class or condition_on_dataset
        
        if condition_on_class:
            self.class_emb = nn.Embedding(max_class_num, dim)
        
        if condition_on_dataset:
            self.dataset_emb = nn.Embedding(max_dataset_num, dim)
        

        self.uncond_emb = nn.Parameter(torch.randn(1, dim))

        self.conditioning_dim = dim
        
        # Transformer blocks with AdaLN
        self.blocks = nn.ModuleList([
            VARTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                bos_scale_id=bos_scale_id,
            )
            for _ in range(depth)
        ])
        
        # Final layer norm and projection
        self.final_norm = RMSNorm(dim)
        
        # Final AdaLN modulation (scale and shift for final norm)
        self.final_adaLN = AdaLNModulation(dim, num_modulations=2 * dim)
        
        # Output projection
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices"""
        # Token and position embeddings
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.scale_emb.weight, std=0.02)
        
        # Class and dataset embeddings
        if self.condition_on_class:
            nn.init.normal_(self.class_emb.weight, std=0.02)
        if self.condition_on_dataset:
            nn.init.normal_(self.dataset_emb.weight, std=0.02)
        
        # Output projection with smaller initialization
        nn.init.normal_(self.to_logits.weight, std=0.02)
        
        # Apply special initialization to linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if not isinstance(module, AdaLNModulation):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def get_conditioning_vector(self, y_true, dataset_id, batch_size, device, dropout_prob=0.1):
        """
        Construct the conditioning vector from class and dataset information.
        """
        conditioning_vectors = []
        
        if self.condition_on_class:
            class_emb = self.class_emb(y_true)
            conditioning_vectors.append(class_emb)
        
        if self.condition_on_dataset:
            dataset_emb = self.dataset_emb(dataset_id)
            conditioning_vectors.append(dataset_emb)
        
        if len(conditioning_vectors) == 0:
            conditioning = self.uncond_emb.expand(batch_size, -1)
        elif len(conditioning_vectors) == 1:
            conditioning = conditioning_vectors[0]
        else:
            conditioning = torch.stack(conditioning_vectors, dim=0).mean(dim=0)
        
        # Classifier-Free Guidance: per-sample dropout during training
        if self.training:
            # Create mask: [batch_size, 1]
            drop_mask = torch.rand(batch_size, 1, device=device) < self.classifier_free_guidance_prob
            uncond = self.uncond_emb.expand(batch_size, -1)
            conditioning = torch.where(drop_mask, uncond, conditioning)
        
        return conditioning


    def forward(self, x, y_true, dataset_id, scale_ids=None):
        """
        Forward pass with AdaLN conditioning.
        
        Args:
            x: [batch_size, seq_len] token indices
            y_true: [batch_size] class labels
            dataset_id: [batch_size] dataset IDs
            scale_ids: [seq_len] optional scale indices for each position (for VAR masking)
        
        Returns:
            logits: [batch_size, seq_len, num_tokens]
        """
        b, n = x.shape
        device = x.device
        
        # Get conditioning vector
        
        conditioning = self.get_conditioning_vector(y_true, dataset_id, b, device)
        
        # Token embeddings
        x = self.token_emb(x)

        
        # Add positional embeddings
        pos = torch.arange(n, device=device)
        x = x + self.pos_emb(pos)
        
        # Add scale embeddings if provided
        if scale_ids is not None:
            # scale_ids is [seq_len], need to expand for batch
            if scale_ids.dim() == 1:
                scale_ids_batch = scale_ids.unsqueeze(0).expand(b, -1)
            else:
                scale_ids_batch = scale_ids
            x = x + self.scale_emb(scale_ids_batch)
        
        # Pass through transformer blocks with conditioning and scale info
        for block in self.blocks:
            x = block(x, conditioning, scale_ids=scale_ids)
        
        # Final normalization with AdaLN
        final_modulations = self.final_adaLN(conditioning)
        if len(final_modulations.shape) == 1:
            final_modulations = final_modulations.unsqueeze(0)
        final_modulations = rearrange(final_modulations, 'b (n d) -> b n d', n=2, d=self.dim)
        shift_final, scale_final = final_modulations.unbind(dim=1)
        
        x = self.final_norm(x)
        x = x * (1 + scale_final.unsqueeze(1)) + shift_final.unsqueeze(1)
        
        # Project to vocabulary
        logits = self.to_logits(x)
        
        return logits


class MultiScaleVARTransformer(nn.Module):
    """
    Extension of VAR for explicit multi-scale autoregressive generation.
    This version handles different scales explicitly and generates coarse-to-fine.
    """
    def __init__(
        self,
        num_tokens,
        dim=512,
        depth=8,
        heads=8,
        dim_head=64,
        ff_mult=4,
        max_class_num=30,
        max_dataset_num=18,
        max_seq_len=2048,
        dropout=0.0,
        mask_id = -1,
        condition_on_class=True,
        condition_on_dataset=False,
        scale_lengths=[1, 2, 4, 8, 16, 32, 64, 128],
        classifier_free_guidance_prob=0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.scale_lengths = scale_lengths
        self.num_scales = len(scale_lengths) + 1
        self.bos_scale_id = self.num_scales - 1
        self.mask_id = mask_id
        
        # Use base VAR transformer
        self.var = VARTransformer(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            max_class_num=max_class_num,
            max_dataset_num=max_dataset_num,
            max_seq_len=max_seq_len,
            dropout=dropout,
            condition_on_class=condition_on_class,
            condition_on_dataset=condition_on_dataset,
            num_scales=self.num_scales,
            bos_scale_id=self.bos_scale_id,
            classifier_free_guidance_prob=classifier_free_guidance_prob,
        )
    
    def get_scale_id_from_position(self, positions, has_bos=True):
        """
        Determine which scale each position belongs to.

        Args:
            positions: [seq_len] position indices (including <BOS> if present)
            positions should be a value that is a multiple of 2 square + 1 (BOS).
            has_bos (bool): whether the sequence includes a BOS token

        Returns:
            scale_ids: [seq_len] scale index for each position
        """
        scale_ids = torch.zeros_like(positions)

        if has_bos:
            # BOS always gets the last scale index
            scale_ids[positions == 0] = self.bos_scale_id
            # Start counting scales after the BOS token
            offset = 1
        else:
            offset = 0

        # Assign scales to the rest
        prev_length = 0
        for scale_idx, length in enumerate(self.scale_lengths):
            scale_ids[offset+prev_length:offset+length] = scale_idx
            prev_length = length
        return scale_ids

    
    def forward_single_scale(self, x, y_true, dataset_id, target_scale):
        """
        Forward pass for a single scale (for next-scale prediction).
        
        Args:
            x: [batch, seq_len] full sequence including BOS
            y_true: [batch] class labels
            dataset_id: [batch] dataset IDs
            target_scale: int, which scale to predict
        
        Returns:
            logits: [batch, num_tokens_at_scale, vocab_size] logits for target scale
            target_tokens: [batch, num_tokens_at_scale] ground truth tokens
            num_tokens: int, number of tokens at this scale
        """
        b, n = x.shape
        device = x.device
        
        # Get scale IDs for full sequence
        positions = torch.arange(n, device=device)
        scale_ids = self.get_scale_id_from_position(positions, has_bos=True)
        
        # Find positions belonging to target scale
        target_mask = (scale_ids == target_scale)
        target_positions = torch.where(target_mask)[0]
        
        if len(target_positions) == 0:
            return None, None, 0
        
        # Create input: BOS + all scales up to and including target scale
        # This allows the model to use teacher forcing within the target scale
        input_end = target_positions.max().item() + 1
        x_input = x[:, :input_end].clone()

        x_input[:, target_positions] = self.mask_id

        scale_ids_input = scale_ids[:input_end].unsqueeze(0).expand(b, -1)
        
        # Forward pass
        logits = self.var(x_input, y_true, dataset_id, scale_ids_input)
        
        # Extract logits only for target scale positions
        target_logits = logits[:, target_positions, :]
        target_tokens = x[:, target_positions]
        
        return target_logits, target_tokens, len(target_positions)
    
    def forward(self, x, y_true, dataset_id, return_loss_dict=False):
        """
        Forward pass with next-scale prediction (for training).
        Computes loss scale-by-scale following VAR paradigm.
        
        Args:
            x: [batch, seq_len] full token sequence (including BOS at position 0)
            y_true: [batch] class labels
            dataset_id: [batch] dataset IDs
            return_loss_dict: bool, whether to return per-scale losses
        
        Returns:
            total_loss: scalar loss
            loss_dict (optional): dict with per-scale losses
        """
        total_loss = 0.0
        loss_dict = {}
        total_tokens = 0
        
        # Iterate through each scale and compute loss
        for scale_idx in range(len(self.scale_lengths)):
            target_logits, target_tokens, num_tokens = self.forward_single_scale(
                x, y_true, dataset_id, target_scale=scale_idx
            )
            
            if num_tokens > 0:
                # Compute cross-entropy loss for this scale
                scale_loss = F.cross_entropy(
                    rearrange(target_logits, 'b n v -> (b n) v'),
                    rearrange(target_tokens, 'b n -> (b n)'),
                    reduction='sum'
                )
                
                total_loss += scale_loss
                total_tokens += num_tokens * x.size(0)
                
                if return_loss_dict:
                    loss_dict[f'scale_{scale_idx}'] = (scale_loss / (num_tokens * x.size(0))).item()
        
        # Average loss across all tokens
        if total_tokens > 0:
            total_loss = total_loss / total_tokens
        
        if return_loss_dict:
            loss_dict['total'] = total_loss.item()
            return total_loss, loss_dict
        
        return total_loss
