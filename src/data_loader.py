from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMG_SIZE = 224
SEED = 42


class CustomDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(data_dir)

    class_names = full_dataset.classes
    num_classes = len(class_names)

    samples = full_dataset.samples
    labels = [label for _, label in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.3,
        stratify=labels,
        random_state=SEED
    )

    temp_labels = [label for _, label in temp_samples]

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.5,
        stratify=temp_labels,
        random_state=SEED
    )

    train_dataset = CustomDataset(train_samples, transform=train_transform)
    val_dataset = CustomDataset(val_samples, transform=eval_transform)
    test_dataset = CustomDataset(test_samples, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset, class_names, num_classes