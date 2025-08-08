import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

class FashionMNISTLoader:
    """
    Fashion MNIST dataset loader with PyTorch
    """

    def __init__(self, batch_size=64, download=True, data_dir='./data'):
        """
        Initialize Fashion MNIST loader

        Args:
            batch_size (int): Batch size for data loading
            download (bool): Whether to download the dataset
            data_dir (str): Directory to store/load the dataset
        """
        self.batch_size = batch_size
        self.data_dir = data_dir

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

        # Load training data
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=download,
            transform=self.transform
        )

        # Load test data
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=download,
            transform=self.transform
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        # Fashion MNIST class names
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

    def get_data_info(self):
        """Get information about the dataset"""
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Image size: {self.train_dataset[0][0].shape}")
        print(f"Number of classes: {len(self.class_names)}")
        print("\nClass names:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")

    def visualize_samples(self, num_samples=16, figsize=(12, 12)):
        """
        Visualize random samples from the dataset

        Args:
            num_samples (int): Number of samples to visualize
            figsize (tuple): Figure size for the plot
        """
        # Get random samples
        indices = torch.randperm(len(self.train_dataset))[:num_samples]

        # Create subplot grid
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_samples > 1 else [axes]

        for i, idx in enumerate(indices):
            # Get sample
            img, label = self.train_dataset[idx]

            # Denormalize image (convert from [-1, 1] to [0, 1])
            img_denorm = (img + 1) / 2

            # Convert to numpy and remove channel dimension
            img_np = img_denorm.squeeze().numpy()

            # Plot
            axes[i].imshow(img_np, cmap='gray')
            axes[i].set_title(f'{self.class_names[label]} ({label})')
            axes[i].axis('off')

        # Hide empty subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_batch(self, batch_idx=0, figsize=(15, 10)):
        """
        Visualize a specific batch from the training data

        Args:
            batch_idx (int): Index of the batch to visualize
            figsize (tuple): Figure size for the plot
        """
        # Get a batch
        data_iter = iter(self.train_loader)
        for i in range(batch_idx + 1):
            images, labels = next(data_iter)

        # Create subplot grid
        batch_size = images.shape[0]
        rows = int(np.sqrt(batch_size))
        cols = int(np.ceil(batch_size / rows))

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if batch_size > 1 else [axes]

        for i in range(batch_size):
            # Denormalize image
            img_denorm = (images[i] + 1) / 2
            img_np = img_denorm.squeeze().numpy()

            # Plot
            axes[i].imshow(img_np, cmap='gray')
            axes[i].set_title(f'{self.class_names[labels[i]]} ({labels[i]})')
            axes[i].axis('off')

        # Hide empty subplots
        for i in range(batch_size, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def get_class_distribution(self):
        """Get and visualize class distribution"""
        # Count samples per class
        class_counts = torch.zeros(len(self.class_names))

        for _, label in self.train_dataset:
            class_counts[label] += 1

        # Plot distribution
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(self.class_names)), class_counts.numpy())
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Fashion MNIST Class Distribution')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{int(count)}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        return class_counts


def main():
    """Main function to demonstrate the Fashion MNIST loader"""
    print("Loading Fashion MNIST dataset...")

    # Initialize loader
    loader = FashionMNISTLoader(batch_size=32)

    # Display dataset information
    print("\n" + "="*50)
    loader.get_data_info()

    # Visualize random samples
    print("\n" + "="*50)
    print("Visualizing random samples...")
    loader.visualize_samples(num_samples=16)

    # Visualize a batch
    print("\n" + "="*50)
    print("Visualizing a batch...")
    loader.visualize_batch(batch_idx=0)

    # Show class distribution
    print("\n" + "="*50)
    print("Class distribution...")
    loader.get_class_distribution()


if __name__ == "__main__":
    main()
