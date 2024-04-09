from detecto import core, utils, visualize
from torchvision import transforms

augmentations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(saturation=0.5),
    transforms.ToTensor(),
    utils.normalize_transform(),
])


dataset = core.Dataset('Data/', transform=augmentations)
loader = core.DataLoader(dataset, batch_size=2, shuffle=True)
model = core.Model(['occupied', 'unoccupied'])
model.fit(loader)