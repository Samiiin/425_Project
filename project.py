# Install dependencies with compatible versions
!pip uninstall -y datasets fsspec gcsfs torch torchvision torchaudio
!pip install datasets==2.20.0 fsspec==2023.10.0 gcsfs==2023.10.0
!pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
!pip install sentence-transformers scikit-learn matplotlib seaborn

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shutil
import os
from huggingface_hub import login
from IPython.display import Image

#Authenticate with Hugging Face
print("Authenticated with Hugging Face.")

#Clear Hugging Face cache
cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cleared Hugging Face cache.")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and Preprocess Dataset
print("Loading dataset...")
dataset = load_dataset("openai/gsm8k", "main", split="train")
questions = [item["question"] for item in dataset]

#Generate Text Embeddings using Sentence-BERT
print("Generating text embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = embedder.encode(questions, batch_size=16, show_progress_bar=True)
text_embeddings = torch.tensor(text_embeddings).float()

# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

#Hyperparameters
input_dim = text_embeddings.shape[1]
hidden_dim = 128
latent_dim = 64
learning_rate = 0.001
epochs = 30
batch_size = 16
num_clusters = 5

#Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder(input_dim, hidden_dim, latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Normalize embeddings for training
text_embeddings = (text_embeddings - text_embeddings.mean()) / text_embeddings.std()
text_embeddings = text_embeddings.to(device)

#Training Loop
print("Training autoencoder...")
model.train()
for epoch in range(epochs):
    for i in range(0, len(text_embeddings), batch_size):
        batch = text_embeddings[i:i+batch_size]
        optimizer.zero_grad()
        _, reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Generate Latent Embeddings
model.eval()
with torch.no_grad():
    latent_embeddings, _ = model(text_embeddings)
latent_embeddings = latent_embeddings.cpu().numpy()

#Clustering with K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(latent_embeddings)

#Evaluation
silhouette_avg = silhouette_score(latent_embeddings, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Visualization with t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(latent_embeddings)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=cluster_labels, palette="deep")
plt.title("t-SNE Visualization of Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster")
plt.savefig("cluster_visualization.png")
print("Cluster visualization saved as 'cluster_visualization.png'")

# Display the plot in Colab
Image("cluster_visualization.png")

#Sample Cluster Analysis
print("\nSample questions from each cluster:")
for cluster in range(num_clusters):
    print(f"\nCluster {cluster}:")
    cluster_indices = np.where(cluster_labels == cluster)[0]
    for idx in cluster_indices[:3]:
        print(f"- {questions[idx]}")
