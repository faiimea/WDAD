# resnet网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-1])
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)