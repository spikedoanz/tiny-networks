import json
from tinygrad import Tensor, nn #type:ignore
from transformer import TransformerBlock
from tinygrad.nn import optim
from pathlib import Path
import numpy as np

COLORS = {
    0: "\033[48;5;16m  \033[0m",  # Black
    1: "\033[48;5;21m  \033[0m",  # Blue
    2: "\033[48;5;196m  \033[0m",  # Red
    3: "\033[48;5;46m  \033[0m",  # Green
    4: "\033[48;5;226m  \033[0m",  # Yellow
    5: "\033[48;5;244m  \033[0m",  # Gray
    6: "\033[48;5;201m  \033[0m",  # Magenta
    7: "\033[48;5;208m  \033[0m",  # Orange
    8: "\033[48;5;51m  \033[0m",  # Light blue
    9: "\033[48;5;88m  \033[0m",  # Brown
}

PAD_TOKEN = 10  # Outside normal color range
SEED = 1337

def display_grid(grid: np.ndarray, title: str = "", raw: bool = False) -> None:
    """Display a grid with colored blocks or as raw array."""
    print(f"\n{title}")
    if raw:
        print(grid)
    else:
        print("-" * (grid.shape[1] * 2))
        for row in grid:
            print("".join(COLORS.get(cell, "  ") for cell in row))

# Data augmentation functions
def rotate_90(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate grid by 90*k degrees"""
    return np.rot90(grid, k)

def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=1)

def flip_vertical(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=0)

def permute_colors(grid: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Randomly permute color labels"""
    np.random.seed(seed)
    unique_colors = np.unique(grid)
    # Keep padding token unchanged
    unique_colors = unique_colors[unique_colors != PAD_TOKEN]
    
    permutation = np.random.permutation(unique_colors)
    mapping = {old: new for old, new in zip(unique_colors, permutation)}
    mapping[PAD_TOKEN] = PAD_TOKEN  # Don't change padding
    
    result = grid.copy()
    for old, new in mapping.items():
        result[grid == old] = new
    return result

def pad_to_30x30(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Pad grid to 30x30 and return grid, mask, and original shape"""
    original_shape = grid.shape
    padded = np.full((30, 30), PAD_TOKEN, dtype=np.int32)
    h, w = grid.shape
    padded[:h, :w] = grid
    
    # Create mask (1 for valid, 0 for padding)
    mask = (padded != PAD_TOKEN).astype(np.float32)
    
    return padded, mask, original_shape

def augment_sample(x: np.ndarray, y: np.ndarray, num_augmentations: int = 10):
    """Generate augmented versions of a sample"""
    augmented_x, augmented_y, masks_x, masks_y = [], [], [], []
    
    for i in range(num_augmentations):
        # Apply same transforms to both input and output
        aug_x, aug_y = x.copy(), y.copy()
        
        # Random rotation
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            aug_x = rotate_90(aug_x, k)
            aug_y = rotate_90(aug_y, k)
        
        # Random flip
        if np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                aug_x = flip_horizontal(aug_x)
                aug_y = flip_horizontal(aug_y)
            else:
                aug_x = flip_vertical(aug_x)
                aug_y = flip_vertical(aug_y)
        
        # Color permutation
        if np.random.rand() > 0.5:
            seed = np.random.randint(1000000)
            aug_x = permute_colors(aug_x, seed)
            aug_y = permute_colors(aug_y, seed)
        
        # Pad to 30x30
        padded_x, mask_x, _ = pad_to_30x30(aug_x)
        padded_y, mask_y, _ = pad_to_30x30(aug_y)
        
        augmented_x.append(padded_x.flatten())
        augmented_y.append(padded_y.flatten())
        masks_x.append(mask_x.flatten())
        masks_y.append(mask_y.flatten())
    
    return np.array(augmented_x), np.array(augmented_y), np.array(masks_x), np.array(masks_y)

def load_and_augment_data(json_files, num_augmentations=10):
    """Load data and apply augmentation"""
    all_x, all_y, all_masks_x, all_masks_y = [], [], [], []
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        
        # Process training examples
        for example in data.get("train", []):
            x = np.array(example["input"])
            y = np.array(example["output"])
            
            aug_x, aug_y, mask_x, mask_y = augment_sample(x, y, num_augmentations)
            all_x.append(aug_x)
            all_y.append(aug_y)
            all_masks_x.append(mask_x)
            all_masks_y.append(mask_y)
    
    # Stack all augmented samples
    if all_x:
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        all_masks_x = np.concatenate(all_masks_x, axis=0)
        all_masks_y = np.concatenate(all_masks_y, axis=0)
    else:
        all_x = np.array([])
        all_y = np.array([])
        all_masks_x = np.array([])
        all_masks_y = np.array([])
    
    return all_x, all_y, all_masks_x, all_masks_y

class SimpleModel:
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, layers=2):
        self.num_colors = 11  # 0-9 colors + padding token
        self.embed_dim = embed_dim
        # Token embedding
        self.token_embed = Tensor.scaled_uniform(self.num_colors, embed_dim)
        # Positional embedding for 30x30 grid (900 positions)
        self.pos_embed = Tensor.scaled_uniform(900, embed_dim)
        # Transformer blocks
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(layers)]
        # Output projection
        self.output_proj = Tensor.scaled_uniform(embed_dim, self.num_colors)
        # Layer norm
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

# Training setup
def train_step(model, x, y, mask_x, mask_y, optimizer):
    """Single training step"""
    Tensor.training = True
    x_tensor = Tensor(x.astype(np.int32))
    y_tensor = Tensor(y.astype(np.int32))
    mask_tensor = Tensor(mask_x.astype(np.float32))
    
    # Forward pass
    logits = model.forward(x_tensor, mask_tensor)  # [batch, 900, num_colors]
    
    # Compute loss only on non-padded positions
    y_flat = y_tensor.reshape(-1)
    logits_flat = logits.reshape(-1, model.num_colors)
    mask_flat = mask_tensor.reshape(-1)
    
    # Cross entropy loss
    loss = logits_flat.sparse_categorical_crossentropy(y_flat)
    loss = (loss * mask_flat).sum() / mask_flat.sum()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.numpy()

# Main execution
if __name__ == "__main__":
    # Load data
    data_path = "ARC-AGI/data"
    split = "training"
    n_samples = 10  # Use subset for testing
    
    data_dir = Path(data_path) / split
    json_files = sorted(data_dir.glob("*.json"))[:n_samples]
    
    # Load and augment data
    print("Loading and augmenting data...")
    X_train, Y_train, masks_X, masks_Y = load_and_augment_data(json_files, num_augmentations=5)
    
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"Y_train: {Y_train.shape}")
    print(f"masks_X: {masks_X.shape}")
    
    # Create model
    model = SimpleModel(embed_dim=64, num_heads=4)
    optimizer = optim.Adam(nn.state.get_parameters(model), lr=1e-3) #type:ignore
    
    # Simple training loop
    batch_size = 32
    num_epochs = 10
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            if len(batch_idx) == 0:
                continue
                
            batch_x = X_train[batch_idx]
            batch_y = Y_train[batch_idx]
            batch_mask_x = masks_X[batch_idx]
            batch_mask_y = masks_Y[batch_idx]
            
            loss = train_step(model, batch_x, batch_y, batch_mask_x, batch_mask_y, optimizer)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    test_input = X_train[:1]  # Single sample
    test_mask = masks_X[:1]
    test_tensor = Tensor(test_input.astype(np.int32))
    test_mask_tensor = Tensor(test_mask.astype(np.float32))
    
    predictions = model(test_tensor, test_mask_tensor)
    print(f"\nTest prediction shape: {predictions.shape}")  # Should be [1, 900]
    
    # Reshape back to grid for visualization
    pred_grid = predictions.numpy().reshape(30, 30)
    print(f"Reshaped grid shape: {pred_grid.shape}")
