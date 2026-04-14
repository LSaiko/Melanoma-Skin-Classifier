from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), #randomly mirror the image 
    transforms.RandomRotation(20), #randomly rotate the image by 20 degrees
    transforms.ToTensor(), #convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]) #normalize the image
])