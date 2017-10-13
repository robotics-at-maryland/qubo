import torch
from torchvision import transforms, datasets


deepseadb = dict()
deepseadb['root']= ''
deepseadb['data_transform']=transforms.Compose(
        [ transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) ]
                              )

deepsea_dataset = datasets.ImageFolder(root=deepseadb['root'],
                                           transform=deepseadb['data_transform'])

dataset_loader = torch.utils.data.DataLoader(deepsea_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
