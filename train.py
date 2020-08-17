import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import IOStream, set_gpu, ensure_path, Timer, count_acc,  Averager, compute_confidence_interval
from dataset.dataset import MiniImagenet
from dataset.sampler import CategoriesSampler
from model.protonet import ProtoNet
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings('ignore')




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mini_imagenet')
parser.add_argument('--distance', default='l2')
parser.add_argument('--shot', default=5, type=int)
parser.add_argument('--way', default=5, type=int)

# parser.add_argument('--train_way', default=30, type=int)
# parser.add_argument('--test_way', default=5, type=int)
parser.add_argument('--query', default=5, type=int)
parser.add_argument('--n_batch', default=100, type=int)
parser.add_argument('--max_epoch', default=100, type=int)

parser.add_argument('--gpu', default=0)
# optimizer
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--step_size', type=int, default=10)  # step size and gamma use to lr scheduler
parser.add_argument('--gamma', type=float, default=0.2)
# model
parser.add_argument('--init_weights', type=str, default=None)
parser.add_argument('--model_type', default='ConvNet', type=str, choices=['ConvNet', 'ResNet', 'AmdimNet'])
# io
parser.add_argument('--save_path', type=str, default='./')


args = parser.parse_args()
io = IOStream('log/run.log')

def cprint(text):
    io.cprint(text)


  


if __name__ == "__main__":
    cprint(str(args))

    set_gpu(args.gpu)
    save_path1 = '-'.join([args.dataset, args.model_type, 'ProtoNet'])
    save_path2 = '_'.join([str(args.shot), str(args.query), str(args.way)])
    args.save_path = osp.join(args.save_path, osp.join(save_path1, save_path2))

    # ensure_path(save_path1, remove=False)
    ensure_path(args.save_path)  

    trainset = MiniImagenet('train', args)
    train_sampler = CategoriesSampler(trainset.label, args.n_batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=2, pin_memory=True)
    
    valset = MiniImagenet('val', args)   
    # only 16 classes in val
    val_sampler = CategoriesSampler(valset.label, args.n_batch, args.way, args.shot + args.query)  
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=2, pin_memory=True)

    model = ProtoNet(args)
    if args.model_type == 'ConvNet':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model_type == 'ResNet':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)        

            # logits = model(data_shot, data_query)
            # loss = F.cross_entropy(logits, label)

    model_dict = model.state_dict()
    # load pretrained model initialization, according to my test, it can't work
    if args.init_weights is not None:
        model_detail = torch.load(args.init_weights)
        if 'params' in model_detail:
            pretrained_dict = model_detail['params']
            # remove weights for FC
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
        
        else:
            pretrained_dict = model_detail['model']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
            # pretrained_dict is empty
            model_dict.update(pretrained_dict)           
    
    model.load_state_dict(model_dict)    

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))


    # log component
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    timer = Timer()
    global_count = 0
    writer = SummaryWriter(logdir=osp.join(args.save_path, 'summary'))


    for epoch in range(1, args.max_epoch + 1):

        lr_scheduler.step()
        model.train()
        tl = Averager()  # average train loss of the epoch
        ta = Averager()  # train acc of the epoch

        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]

            label = torch.arange(args.way).repeat(args.query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)
            logits = model(data_shot, data_query)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            writer.add_scalar(osp.join(args.save_path, 'summary/tr_loss'), float(loss), global_count)
            writer.add_scalar(osp.join(args.save_path, 'summary/tr_acc'), float(acc), global_count)
            # writer.add_scalar('data/loss', float(loss), global_count)
            # writer.add_scalar('data/acc', float(acc), global_count)

            cprint('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        tl = tl.item()  
        ta = ta.item()

        # validation of the epoch
        model.eval()

        vl = Averager()  # average val loss of the epoch
        va = Averager()  # average val acc of the epoch


        cprint('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        #  
        with torch.no_grad():  
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                label = torch.arange(args.way).repeat(args.query)
                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)

                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]

                logits = model(data_shot, data_query)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)    
                vl.add(loss.item())
                va.add(acc)
        
        vl = vl.item()
        va = va.item()
        writer.add_scalar(osp.join(args.save_path, 'summary/val_loss'), float(vl), epoch)
        writer.add_scalar(osp.join(args.save_path, 'summary/val_acc'), float(va), epoch)        

        # writer.add_scalar('data/val_loss', float(vl), epoch)
        # writer.add_scalar('data/val_acc', float(va), epoch)        
        cprint('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        save_model('epoch-last')
        
        # total time, average time per epoch
        cprint('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()


    # Test Phase
    cprint("begin testing----------------------------------------------------------")

    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    test_set = MiniImagenet('test', args)
    sampler = CategoriesSampler(test_set.label, 10000, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=2, pin_memory=True)
    test_acc_record = np.zeros((10000,))
    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    model.eval()

    ave_acc = Averager()
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)

    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]
    
            logits = model(data_shot, data_query)
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            cprint('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    m, pm = compute_confidence_interval(test_acc_record)
    cprint('Val Best Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc'], ave_acc.item()))
    cprint('Test Acc {:.4f} + {:.4f}'.format(m, pm))
    cprint('Val Best Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc'], ave_acc.item()))
    cprint('Test Acc {:.4f} + {:.4f}'.format(m, pm))