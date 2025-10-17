
from pathlib import Path

import numpy as np
from tinygrad import Tensor, nn #type:ignore

from transformer import TransformerBlock
from data import arc_agi


class SimpleModel:
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, layers=2):
        self.num_colors = 11  # 0-9 colors + padding token
        self.embed_dim = embed_dim
        self.token_embed = Tensor.scaled_uniform(self.num_colors, embed_dim)
        self.pos_embed = Tensor.scaled_uniform(900, embed_dim)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(layers)]
        self.output_proj = Tensor.scaled_uniform(embed_dim, self.num_colors)
        self.ln_final = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    
    def forward(self, x, mask=None):
        """
        x: [batch, 900] flattened 30x30 grids (integer tokens)
        mask: [batch, 900] attention mask (optional)
        """
        bs, seq_len = x.shape
        x_onehot = x.int().one_hot(self.num_colors)  # [batch, 900, 11]
        x_embed = x_onehot.dot(self.token_embed)  # [batch, 900, embed_dim]
        
        pos_indices = Tensor.arange(seq_len).unsqueeze(0).expand([bs, seq_len])
        pos_onehot = pos_indices.int().one_hot(900)  # [batch, 900, 900]
        pos_embed = pos_onehot.dot(self.pos_embed)  # [batch, 900, embed_dim]
        x = x_embed + pos_embed
        
        for block in self.blocks:
            x = block(x)
        x = x.layernorm().linear(*self.ln_final)
        logits = x.dot(self.output_proj)  # [batch, 900, num_colors]
        return logits
    
    def __call__(self, x, mask=None):
        logits = self.forward(x, mask)
        return logits.argmax(axis=-1)  # [batch, 900]

def train_step(model, x, y, mask_x, mask_y, optimizer):
    """Single training step"""
    Tensor.training = True
    x_tensor = Tensor(x.astype(np.int32))
    y_tensor = Tensor(y.astype(np.int32))
    mask_tensor = Tensor(mask_x.astype(np.float32))
    
    logits = model.forward(x_tensor, mask_tensor)  # [batch, 900, num_colors]
    
    y_flat = y_tensor.reshape(-1)
    logits_flat = logits.reshape(-1, model.num_colors)
    mask_flat = mask_tensor.reshape(-1)
    
    loss = logits_flat.sparse_categorical_crossentropy(y_flat)
    loss = (loss * mask_flat).sum() / mask_flat.sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.numpy()

if __name__ == "__main__":
    data_path = "ARC-AGI/data"
    split = "training"
    n_samples = 10

    # hyperparams
    lr = 1e-3
    batch_size = 32
    num_epochs = 10
    
    data_dir = Path(data_path) / split
    json_files = sorted(data_dir.glob("*.json"))[:n_samples]
    
    # load and augment data
    print("Loading and augmenting data...")
    X_train, Y_train, masks_X, masks_Y = arc_agi(json_files)
    
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"Y_train: {Y_train.shape}")
    print(f"masks_X: {masks_X.shape}")
    
    model = SimpleModel(embed_dim=64, num_heads=4)
    optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=lr) #type:ignore
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) == 0:
                continue
                
            batch_x = X_train[batch_idx]
            batch_y = Y_train[batch_idx]
            batch_mask_x = masks_X[batch_idx]
            batch_mask_y = masks_Y[batch_idx]
            
            loss = train_step(
                model, batch_x, batch_y, batch_mask_x, batch_mask_y, optimizer)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    test_input = X_train[:1]
    test_mask = masks_X[:1]
    test_tensor = Tensor(test_input.astype(np.int32))
    test_mask_tensor = Tensor(test_mask.astype(np.float32))
    
    predictions = model(test_tensor, test_mask_tensor)
    print(f"\nTest prediction shape: {predictions.shape}")
    pred_grid = predictions.numpy().reshape(30, 30)
    print(f"Reshaped grid shape: {pred_grid.shape}")
