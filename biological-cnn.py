import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

MNIST_FASHIONMNIST_CLASSES = 10


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_interval: int = 2500,
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
            # Define non-grid connectivity layers here
            pass
        else:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        if use_lateral_inhibition:
            # Define lateral inhibition layers here
            pass
        else:
            self.lateral1 = nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)

        if use_non_grid:
            # Define non-grid connectivity layers here
            pass
        else:
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        if use_lateral_inhibition:
            # Define lateral inhibition layers here
            pass
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
            # Update lateral inhibition strength here
            pass


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

    all_clean_acc = []
    all_noisy_acc = []

    for trial in range(trials):
        print(f"\n--- Trial {trial + 1}/{trials} ---")
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
                new_strength = inhibition_strength * (0.95**epoch)
                model.update_inhibition_strength(new_strength)

        clean_loss, clean_acc = test(model, device, test_loader)
        all_clean_acc.append(clean_acc)
        print(f"Clean Accuracy: {clean_acc:.4f} | Loss: {clean_loss:.4f}")

    print(f"\nSummary over {trials} trials for {dataset_name}:")
    print(
        f"Clean Test Accuracy: Mean = {np.mean(all_clean_acc):.4f}, Std = {np.std(all_clean_acc):.4f}"
    )


def main():
    trials = 5
    epochs = 5
    noise_stds = [0.0, 0.1, 0.2, 0.3, 0.4]
    inhibition_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Progressively more brainlike configurations
    # 1. No lateral inhibition, no non-grid, no plasticity Baseline CNN pure ML
    # 2. Lateral inhibition, no non-grid, no plasticity Single Biological feature
    # 3. Lateral inhibition, non-grid, no plasticity Multiple Biological features
    # 4. Lateral inhibition, non-grid, plasticity All Biological features
    configurations = [
        {"use_lateral_inhibition": False, "use_non_grid": False, "plasticity": False},
        {"use_lateral_inhibition": True, "use_non_grid": False, "plasticity": False},
        {"use_lateral_inhibition": True, "use_non_grid": True, "plasticity": False},
        {"use_lateral_inhibition": True, "use_non_grid": True, "plasticity": True},
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


def simpleMain():
    run_experiment(
        dataset_name="MNIST",
        inhibition_strength=0.1,
        noise_std=0.1,
        epochs=5,
        trials=3,
        batch_size=64,
        test_batch_size=1000,
        use_lateral_inhibition=False,
        use_non_grid=False,
        plasticity=False,
    )


if __name__ == "__main__":
    simpleMain()
    # main()
