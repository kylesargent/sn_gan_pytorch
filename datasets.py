import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset_iter(dataset, dataset_path, batch_size):
	if dataset == "cifar10":
		return get_cifar10_iter(dataset_path, batch_size)
	else:
		raise NotImplementedError("Dataset loader not implemented")

def cifar10_preprocess(tensor):
    transformed_tensor = 2. * tensor - 1.
    transformed_tensor += torch.rand(*transformed_tensor.shape) / 128.
    return transformed_tensor
    
def get_cifar10_iter(dataset_path, batch_size):
	transform = transforms.Compose(
	    [
	        transforms.ToTensor(),
	        cifar10_preprocess
	    ]
	)
	trainset = torchvision.datasets.CIFAR10(
		root=dataset_path, 
		train=True,
		download=True, 
		transform=transform
	)
	trainloader = torch.utils.data.DataLoader(
		trainset, 
		batch_size=batch_size,
		shuffle=True, 
		num_workers=1
	)
	return trainloader