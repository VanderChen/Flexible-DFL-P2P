import os
import time
import socket
import pickle
import argparse
import asyncio
import concurrent.futures
import threading
import math
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pulp import *
import random
from config import ClientConfig, CommonConfig
from client_comm_utils import *
from training_utils import train, test
import datasets, models

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--idx', type=str, default="0",
                    help='index of worker')
parser.add_argument('--master_ip', type=str, default="127.0.0.1",
                    help='IP address for controller or ps')
parser.add_argument('--master_port', type=int, default=58000, metavar='N',
                    help='')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

if args.visible_cuda == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = str((int(args.idx)) % 2 + 0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def main():
    client_config = ClientConfig(
        common_config=CommonConfig()
    )
    recorder = SummaryWriter("log_"+str(args.idx))
    # receive config
    master_socket = connect_get_socket(args.master_ip, args.master_port)
    config_received = get_data_socket(master_socket)
    #这里跟服务器通信然后获取配置文件，get_data_socket是堵塞的。
    for k, v in config_received.__dict__.items():
        setattr(client_config, k, v)

    computation = client_config.custom["computation"]
    dynamics=client_config.custom["dynamics"]

    common_config = CommonConfig()
    common_config.model_type = client_config.common_config.model_type
    common_config.dataset_type = client_config.common_config.dataset_type
    common_config.batch_size = client_config.common_config.batch_size
    common_config.data_pattern=client_config.common_config.data_pattern
    common_config.lr = client_config.common_config.lr
    common_config.decay_rate = client_config.common_config.decay_rate
    common_config.min_lr=client_config.common_config.min_lr
    common_config.epoch = client_config.common_config.epoch
    common_config.momentum = client_config.common_config.momentum
    common_config.weight_decay = client_config.common_config.weight_decay
    common_config.para_nums=client_config.common_config.para_nums

    # init config
    print(common_config.__dict__)

    # create model
    local_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    #local_model.load_state_dict(client_config.para)
    torch.nn.utils.vector_to_parameters(client_config.para, local_model.parameters())
    local_model.to(device)
    #para_nums = torch.nn.utils.parameters_to_vector(local_model.parameters()).nelement()

    # create dataset
    print(len(client_config.custom["train_data_idxes"]))
    train_dataset, test_dataset = datasets.load_datasets(common_config.dataset_type)
    train_loader = datasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, selected_idxs=client_config.custom["train_data_idxes"])
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8,)
    tasks = []
    for _, (neighbor_ip, send_port, listen_port) in client_config.custom["neighbor_info"].items():
        tasks.append(loop.run_in_executor(executor, connect_send_socket, neighbor_ip, send_port))
        tasks.append(loop.run_in_executor(executor, connect_get_socket, args.master_ip, listen_port))
    loop.run_until_complete(asyncio.wait(tasks))
    neighbor_list=list()
    for neighbor_idx, neighbor_name in enumerate(client_config.custom["neighbor_info"].keys()):
        client_config.send_socket_dict[neighbor_name] = tasks[neighbor_idx*2].result()
        client_config.get_socket_dict[neighbor_name] = tasks[neighbor_idx*2+1].result()
        neighbor_list.append(neighbor_name)
    loop.close()


    epoch_lr = common_config.lr
    local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    old_para = copy.deepcopy(local_para)
    memory_para= torch.zeros_like(local_para)
    for epoch in range(1, 1+common_config.epoch):
        
        local_steps,compre_ratio=get_data_socket(master_socket)

        if epoch > 1 and epoch % 1 == 0:
            epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
        print("epoch-{} lr: {}".format(epoch, epoch_lr))
        print("local steps: ", local_steps)
        print("Compression Ratio: ", compre_ratio)

        #print("***")
        start_time = time.time()
        if common_config.momentum<0:
            optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
        else:
            optimizer = optim.SGD(local_model.parameters(),momentum=common_config.momentum, lr=epoch_lr, weight_decay=common_config.weight_decay)
        train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device, model_type=common_config.model_type)
        local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
        memory_para,compressed_paras = compress_model_top_with_memory(local_para,old_para,memory_para,compre_ratio)
        print("send and get")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            tasks = []
            for neighbor_name in client_config.custom["neighbor_info"].keys():
                tasks.append(loop.run_in_executor(executor, send_para, 
                                        compressed_paras,client_config.send_socket_dict[neighbor_name]))
                tasks.append(loop.run_in_executor(executor, get_compressed_model_top, 
                                        client_config,neighbor_name))
            loop.run_until_complete(asyncio.wait(tasks))
        except SystemExit:
            master_socket.close()
        loop.close()
        local_para = aggregate_model_with_memory(local_para, client_config)
        old_para=copy.deepcopy(local_para)
        torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())

        send_data_socket(True, master_socket)
        epoch_complete =  get_data_socket(master_socket)

        train_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        while train_time>10 or train_time<1:
             train_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        train_time=train_time/10
        print(train_time)

        # send_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))
        # while send_time>10 or send_time<1:
        #      send_time = np.random.normal(loc=computation, scale=np.sqrt(dynamics))

        test_loss, acc = test(local_model, test_loader, device, model_type=common_config.model_type)
        if epoch%1 ==0:
            send_data_socket((epoch, train_time, acc, test_loss,train_loss), master_socket)
        recorder.add_scalar('acc_worker-' + str(args.idx), acc, epoch)
        recorder.add_scalar('test_loss_worker-' + str(args.idx), test_loss, epoch)
        recorder.add_scalar('train_loss_worker-' + str(args.idx), train_loss, epoch)
        print("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss, test_loss, acc))
    
    for neighbor_name in client_config.custom["neighbor_info"].keys():
        client_config.send_socket_dict[neighbor_name].shutdown(2)
        client_config.send_socket_dict[neighbor_name].close()

        client_config.get_socket_dict[neighbor_name].shutdown(2)
        client_config.get_socket_dict[neighbor_name].close()
    master_socket.shutdown(2)
    master_socket.close()

def compress_model_rand(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        select_n = int(send_para.nelement() * ratio)
        rd_seed = np.random.randint(0, np.iinfo(np.uint32).max)
        rng = np.random.RandomState(rd_seed)
        indices = rng.choice(send_para.nelement(), size=select_n, replace=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)
    return (select_para, select_n, rd_seed)

def compress_model_top(local_para, ratio):
    start_time = time.time()
    with torch.no_grad():
        send_para = local_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(local_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)

    return (select_para, indices)

def compress_model_top_with_memory(local_para, old_para, memory_para,ratio):
    start_time = time.time()
    with torch.no_grad():
        old_para=local_para-old_para+memory_para
        send_para = local_para.detach()
        topk = int(send_para.nelement() * ratio)
        _, indices = torch.topk(old_para.abs(), topk, largest=True, sorted=False)
        select_para = send_para[indices]
    print("compressed time: ", time.time() - start_time)
    restored_model = torch.zeros(send_para.nelement()).to(device)
    restored_model[indices] = select_para
    memory_para=local_para - restored_model
    #model_size = select_para.nelement() * 4 / 1024 / 1024
    #print("model_size:",model_size)
    #print("memory_para:",memory_para)
    return (memory_para,(select_para, indices))

def send_para(local_para,send_socket):
    start_time = time.time()
    send_data_socket(local_para, send_socket)
    print("send time: ", time.time() - start_time)
    pass

def get_compressed_model_top(client_config, name):
    start_time = time.time()
    nelement=client_config.common_config.para_nums
    received_para, indices = get_data_socket(client_config.get_socket_dict[name])
    received_para.to(device)
    print("get time: ", time.time() - start_time)

    restored_model = torch.zeros(nelement).to(device)
    
    restored_model[indices] = received_para
    client_config.neighbor_paras[name] = restored_model.data
    # print("get:")
    # print(client_config.neighbor_paras[name])
    client_config.neighbor_indices[name] = indices

def aggregate_model_with_memory(local_para, client_config):
    with torch.no_grad():
        para_delta = torch.zeros_like(local_para)
        for neighbor_name in client_config.custom["neighbor_info"].keys():
            print("idx: {}, weight: {}".format(neighbor_name, client_config.weight[neighbor_name]))
            indice = client_config.neighbor_indices[neighbor_name]
            selected_indicator = torch.zeros_like(local_para)
            selected_indicator[indice] = 1.0
            model_delta = (client_config.neighbor_paras[neighbor_name]-local_para)*selected_indicator
            #model_delta = client_config.neighbor_paras[neighbor_name]
            para_delta += client_config.weight[neighbor_name] * model_delta
            #para_delta += 0.45 * model_delta
        #print(para_delta)
        local_para += para_delta

    return local_para

if __name__ == '__main__':
    main()
