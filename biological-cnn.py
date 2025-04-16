import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import time
import csv

MNIST_FASHIONMNIST_CLASSES = 10

# xxxxxxxxxxxxxxxxxxxxxxxxx
# Utility Functions
# xxxxxxxxxxxxxxxxxxxxxxxxx


# adds noise to images to simulate real-world conditions
# and to test the robustness of the model
def add_noise(images, noise_std: float) -> torch.Tensor:
    noise = torch.randn_like(images) * noise_std
    noisy_images = images + noise
    print(
        "Difference between images and noisy images: ",
        torch.mean(torch.abs(images - noisy_images)),
    )
    return torch.clamp(noisy_images, -1.0, 1.0)


# Trains the model for one epoch
def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_interval: int = 50,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}"
            )


# Tests the model on the test set
def test(model: nn.Module, device: torch.device, test_loader: DataLoader) -> tuple:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset) * 100.0
    return test_loss, accuracy


# Tests the model on the test set with added noise
def test_noisy(
    model: nn.Module, device: torch.device, test_loader: DataLoader, noise_std: float
) -> tuple:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            noisy_data = add_noise(data, noise_std)
            output = model(noisy_data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset) * 100.0
    return test_loss, accuracy


# Visualizes the features using PCA and t-SNE
def feature_analysis(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    num_samples: int = 500,
    config_name: str = "Baseline",
    inhibition_strength: float = 0.0,
    noise_std: float = 0.0,
    dataset_name: str = "MNIST",
    show_plots: bool = False,
) -> None:
    os.makedirs(config_name, exist_ok=True)
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, feature_map = model(data, return_features=True)
            feature_map = feature_map.view(feature_map.size(0), -1)
            features.append(feature_map.cpu().numpy())
            labels.append(target.cpu().numpy())
            if len(np.concatenate(features)) >= num_samples:
                break
    features = np.concatenate(features, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]

    # PCA 2D
    pca_2d = PCA(n_components=2)
    features_pca_2d = pca_2d.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        features_pca_2d[:, 0],
        features_pca_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Classes")
    plt.title(
        f"PCA 2D of Intermediate Features {config_name} {inhibition_strength} {noise_std} {dataset_name}"
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(
        os.path.join(
            config_name,
            f"pca_2d_{config_name}_{inhibition_strength}_{noise_std}_{dataset_name}.png",
        )
    )
    if show_plots:
        plt.show()
    plt.close()

    # PCA 3D
    pca_3d = PCA(n_components=3)
    features_pca_3d = pca_3d.fit_transform(features)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        features_pca_3d[:, 0],
        features_pca_3d[:, 1],
        features_pca_3d[:, 2],
        c=labels,
        cmap="tab10",
        alpha=0.7,
    )
    ax.set_title(
        f"PCA 3D of Intermediate Features {config_name} {inhibition_strength} {noise_std} {dataset_name}"
    )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.colorbar(scatter, label="Classes")
    plt.savefig(
        os.path.join(
            config_name,
            f"pca_3d_{config_name}_{inhibition_strength}_{noise_std}_{dataset_name}.png",
        )
    )
    if show_plots:
        plt.show()
    plt.close()

    # t-SNE 2D
    tsne_2d = TSNE(n_components=2, perplexity=30, init="random", random_state=42)
    features_tsne_2d = tsne_2d.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        features_tsne_2d[:, 0],
        features_tsne_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Classes")
    plt.title(
        f"t-SNE 2D of Intermediate Features {config_name} {inhibition_strength} {noise_std} {dataset_name}"
    )
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.savefig(
        os.path.join(
            config_name,
            f"tsne_2d_{config_name}_{inhibition_strength}_{noise_std}_{dataset_name}.png",
        )
    )
    if show_plots:
        plt.show()
    plt.close()

    # t-SNE 3D
    tsne_3d = TSNE(n_components=3, perplexity=30, init="random", random_state=42)
    features_tsne_3d = tsne_3d.fit_transform(features)
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        features_tsne_3d[:, 0],
        features_tsne_3d[:, 1],
        features_tsne_3d[:, 2],
        c=labels,
        cmap="tab10",
        alpha=0.7,
    )
    ax.set_title(
        f"t-SNE 3D of Intermediate Features {config_name} {inhibition_strength} {noise_std} {dataset_name}"
    )
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_zlabel("t-SNE Dim 3")
    plt.colorbar(scatter, label="Classes")
    plt.savefig(
        os.path.join(
            config_name,
            f"tsne_3d_{config_name}_{inhibition_strength}_{noise_std}_{dataset_name}.png",
        )
    )
    if show_plots:
        plt.show()
    plt.close()


# xxxxxxxxxxxxxxxxxxxxxx
# Biologically inspired layers
# xxxxxxxxxxxxxxxxxxxxxx


# Lateral inhibition layer to simulate the behavior of neurons in the brain
class LateralInhibition(nn.Module):
    def __init__(self, channels: int, inhibition_strength: float = 0.5):
        super(LateralInhibition, self).__init__()
        self.inhibition_strength = inhibition_strength
        self.channels = channels
        # A fixed 3x3 kernel that averages the 8 neighbors excluding the center pixel
        kernel = torch.ones((1, 1, 3, 3)) / 8.0
        kernel[0, 0, 1, 1] = 0.0  # Center pixel is not included
        self.register_buffer("kernel", kernel.expand(channels, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inhibition = F.conv2d(x, self.kernel, padding=1, groups=x.shape[1])
        return x - self.inhibition_strength * inhibition


# Non-grid convolution layer to simulate the non-grid connectivity of neurons in the brain
class NonGridConv2d(nn.Conv2d):
    def __init__(self, *args, sparsity: float = 0.7, **kwargs):
        super(NonGridConv2d, self).__init__(*args, **kwargs)
        mask = torch.bernoulli(torch.full(self.weight.shape, sparsity))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.mask
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


# xxxxxxxxxxxxxxxxxxxxxx
# CNN Model
# xxxxxxxxxxxxxxxxxxxxxx


class SimpleCNN(nn.Module):
    def __init__(
        self,
        use_lateral_inhibition: bool = False,
        inhibition_strength: float = 0.0,
        use_non_grid: bool = False,
        plasticity: bool = False,
    ):
        super(SimpleCNN, self).__init__()
        self.use_lateral_inhibition = use_lateral_inhibition
        self.placticity = plasticity
        self.initial_inhibition_strength = inhibition_strength
        self.current_inhibition_strength = inhibition_strength

        if use_non_grid:
            self.conv1 = NonGridConv2d(1, 16, kernel_size=3, padding=1, sparsity=0.7)
        else:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        if use_lateral_inhibition:
            self.lateral1 = LateralInhibition(16, self.current_inhibition_strength)
        else:
            self.lateral1 = nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)

        if use_non_grid:
            self.conv2 = NonGridConv2d(16, 32, kernel_size=3, padding=1, sparsity=0.7)
        else:
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        if use_lateral_inhibition:
            self.lateral2 = LateralInhibition(32, self.current_inhibition_strength)
        else:
            self.lateral2 = nn.Identity()

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, MNIST_FASHIONMNIST_CLASSES)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.lateral1(x)
        x = self.pool(x)
        # Save features
        feature_map = x.clone()
        x = self.conv2(x)
        x = F.relu(x)
        x = self.lateral2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if return_features:
            return x, feature_map
        return x

    def update_inhibition_strength(self, new_strength: float) -> None:
        self.current_inhibition_strength = new_strength
        if self.use_lateral_inhibition:
            self.lateral1.inhibition_strength = new_strength
            self.lateral2.inhibition_strength = new_strength


# xxxxxxxxxxxxxxxxxxxxxx
# Main Experiment Function
# xxxxxxxxxxxxxxxxxxxxx
CSV_FILE = "experiment_results.csv"
CSV_FIELDS = [
    "dataset",
    "config_name",
    "trial",
    "noise_std",
    "inhibition_strength",
    "clean_loss",
    "clean_acc",
    "noisy_loss",
    "noisy_acc",
    "runtime_sec",
]


def run_experiment(
    dataset_name: str,
    inhibition_strength: float,
    noise_std: float,
    epochs: int,
    trials: int,
    batch_size: int,
    test_batch_size: int,
    use_lateral_inhibition: bool,
    use_non_grid: bool,
    plasticity: bool,
    config_name: str,
    save_model: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment on {dataset_name} using device: {device}")

    # Normalize so pixels are in [-1, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError("Dataset not supported.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()

        all_clean_acc = []
        all_noisy_acc = []

        for trial in range(trials):
            print(f"\n--- Trial {trial + 1}/{trials} ---")
            trial_start_time = time.time()
            try:
                model = SimpleCNN(
                    use_lateral_inhibition=use_lateral_inhibition,
                    inhibition_strength=inhibition_strength,
                    use_non_grid=use_non_grid,
                    plasticity=plasticity,
                ).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

                for epoch in range(1, epochs + 1):
                    train(model, device, train_loader, optimizer, epoch)
                    if plasticity:
                        # For simplicity decay by 15% each epoch
                        new_strength = inhibition_strength * (0.85**epoch)
                        model.update_inhibition_strength(new_strength)

                clean_loss, clean_acc = test(model, device, test_loader)
                noisy_loss, noisy_acc = test_noisy(
                    model, device, test_loader, noise_std
                )
                runtime_sec = time.time() - trial_start_time
                print(
                    f"Trial {trial + 1} Results: Clean Accuracy: {clean_acc:.4f}, Clean Loss: {clean_loss:.4f} | Noisy Accuracy: {noisy_acc:.4f}, Noisy Loss: {noisy_loss:.4f} | Runtime: {runtime_sec:.2f} sec"
                )

                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "config_name": config_name,
                        "trial": trial + 1,
                        "noise_std": noise_std,
                        "inhibition_strength": inhibition_strength,
                        "clean_loss": clean_loss,
                        "clean_acc": clean_acc,
                        "noisy_loss": noisy_loss,
                        "noisy_acc": noisy_acc,
                        "runtime_sec": runtime_sec,
                    }
                )
                file.flush()

                all_clean_acc.append(clean_acc)
                all_noisy_acc.append(noisy_acc)

                if save_model:
                    model_dir = os.path.join(config_name, "models")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(
                        model_dir, f"{dataset_name}_trial_{trial + 1}.pth"
                    )
                    torch.save(model.state_dict(), model_path)

                if trial == 0:
                    feature_analysis(
                        model,
                        device,
                        test_loader,
                        num_samples=500,
                        config_name=config_name,
                        inhibition_strength=inhibition_strength,
                        noise_std=noise_std,
                        dataset_name=dataset_name,
                        show_plots=False,
                    )
            except Exception as e:
                print(f"Error during trial {trial + 1}: {e}")
                continue
            finally:
                # Clean up the model to free memory
                del model
                torch.cuda.empty_cache()

    print(f"\nSummary over {trials} trials for {dataset_name}:")
    print(
        f"Clean Test Accuracy: Mean = {np.mean(all_clean_acc):.4f}, Std = {np.std(all_clean_acc):.4f}"
    )
    print(
        f"Noisy Test Accuracy: Mean = {np.mean(all_noisy_acc):.4f}, Std = {np.std(all_noisy_acc):.4f}"
    )


# Main function to run the experiment with different configurations


def main():
    trials = 3
    epochs = 5
    noise_stds = [0.3, 0.4, 0.5]
    inhibition_strengths = [0.3, 0.4, 0.5]

    # Progressively more brainlike configurations
    # 1. No lateral inhibition, no non-grid, no plasticity Baseline CNN pure ML
    # 2. Lateral inhibition, no non-grid, no plasticity Single Biological feature
    # 3. Lateral inhibition, non-grid, no plasticity Multiple Biological features
    # 4. Lateral inhibition, non-grid, plasticity All Biological features
    configurations = [
        {
            "use_lateral_inhibition": False,
            "use_non_grid": False,
            "plasticity": False,
            "config_name": "Baseline",
        },
        {
            "use_lateral_inhibition": True,
            "use_non_grid": False,
            "plasticity": False,
            "config_name": "Single Biological",
        },
        {
            "use_lateral_inhibition": True,
            "use_non_grid": True,
            "plasticity": False,
            "config_name": "Multiple Biological",
        },
        {
            "use_lateral_inhibition": True,
            "use_non_grid": True,
            "plasticity": True,
            "config_name": "All Biological",
        },
    ]

    datasets = ["MNIST", "FashionMNIST"]

    for dataset in datasets:
        for config in configurations:
            for noise_std in noise_stds:
                for inhibition_strength in inhibition_strengths:
                    print("\n" + "=" * 50)
                    print(f"Running on {dataset} with config:")
                    print(f"  Lateral Inhibition: {config['use_lateral_inhibition']}")
                    print(f"  Non-Grid Connectivity: {config['use_non_grid']}")
                    print(f"  Synaptic Plasticity: {config['plasticity']}")
                    print(f"  Noise Std: {noise_std}")
                    print(f"  Inhibition Strength: {inhibition_strength}")
                    print("=" * 50)

                    run_experiment(
                        dataset_name=dataset,
                        inhibition_strength=inhibition_strength,
                        noise_std=noise_std,
                        epochs=epochs,
                        trials=trials,
                        batch_size=64,
                        test_batch_size=1000,
                        use_lateral_inhibition=config["use_lateral_inhibition"],
                        use_non_grid=config["use_non_grid"],
                        plasticity=config["plasticity"],
                        config_name=config["config_name"],
                        save_model=True,
                    )


def simpleMain():
    run_experiment(
        dataset_name="MNIST",
        inhibition_strength=0.2,
        noise_std=0.2,
        epochs=5,
        trials=3,
        batch_size=64,
        test_batch_size=1000,
        use_lateral_inhibition=True,
        use_non_grid=True,
        plasticity=True,
        config_name="tests",
        save_model=True,
    )


if __name__ == "__main__":
    # simpleMain()
    main()
