from resnet import ResNet , bottleNeck , baseBlock
import torchvision.transforms as transforms 
import torchvision 
import torch

def test():
        #To convert data from PIL to tensor
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    #Load train and test set:
    train = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    trainset = torch.utils.data.DataLoader(train,batch_size=128,shuffle=True)

    test = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    testset = torch.utils.data.DataLoader(test,batch_size=128,shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #ResNet-18 
    #net = ResNet(baseBlock,[2,2,2,2],10)

    #ResNet-50
    net =  ResNet(bottleNeck,[3,4,6,3])
    net.to(device)
    costFunc = torch.nn.CrossEntropyLoss()
    optimizer =  torch.optim.SGD(net.parameters(),lr=0.02,momentum=0.9)

    for epoch in range(100):
        closs = 0
        for i,batch in enumerate(trainset,0):
            data,output = batch
            data,output = data.to(device),output.to(device)
            prediction = net(data)
            loss = costFunc(prediction,output)
            closs = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print every 1000th time
            if i%100 == 0:
                print('[%d  %d] loss: %.4f'% (epoch+1,i+1,closs/1000))
                closs = 0
        correctHits=0
        total=0
        for batches in testset:
            data,output = batches
            data,output = data.to(device),output.to(device)
            prediction = net(data)
            _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
            total += output.size(0)
            correctHits += (prediction==output).sum().item()
        print('Accuracy on epoch ',epoch+1,'= ',str((correctHits/total)*100))

    correctHits=0
    total=0
    for batches in testset:
        data,output = batches
        data,output = data.to(device),output.to(device)
        prediction = net(data)
        _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
        total += output.size(0)
        correctHits += (prediction==output).sum().item()
    print('Accuracy = '+str((correctHits/total)*100))

if __name__ == '__main__':
    test()
