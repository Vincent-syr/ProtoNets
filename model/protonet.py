import torch.nn as nn
from utils import euclidean_metric
from model.convnet import Convnet
from model.resnet import ResNet

class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model_type == 'ConvNet':
            self.encoder = Convnet()
            # print('ConvNet')
        elif args.model_type == 'ResNet':
            self.encoder = ResNet()
        
        else:
            raise ValueError('model_type must be one of ConvNet, ResNet')


    def forward(self, data_shot, data_query):
        proto = self.encoder(data_shot)
        # print(proto.shape)   # torch.Size([5, 64])
        proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)    # proto:(task*n_way, feat); query()
       
        logits = euclidean_metric(self.encoder(data_query), proto)
        return logits

    