import os
import sys
import argparse
import socket
import pickle
import asyncio
import concurrent.futures
import json
import random
import time
import numpy as np
import threading
import torch
import copy
import math
from config import *
import torch.nn.functional as F
import datasets, models
from training_utils import test

#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='CIFAR10')
parser.add_argument('--model_type', type=str, default='AlexNet')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.000)
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
master_listen_port_base = 57600+random.randint(0, 20) * 20
p2p_listen_port_base = 56000 + random.randint(0, 20) * 20
RESULT_PATH = 'result_record'

def main():

    # init config
    common_config = CommonConfig()
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.data_pattern=args.data_pattern
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.min_lr=args.min_lr
    common_config.epoch = args.epoch
    common_config.momentum = args.momentum
    common_config.weight_decay = args.weight_decay

    #read the worker_config.json to init the worker node
    with open("worker_config.json") as json_file:
        workers_config = json.load(json_file)

    worker_num = len(workers_config['worker_config_list'])

    global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    common_config.para_nums=init_para.nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("para num: {}".format(common_config.para_nums))
    print("Model Size: {} MB".format(model_size))


    adjacency_matrix = np.zeros((worker_num, worker_num), dtype=np.int)
    B=[[0, 6, 3, 5, 2, 7, 2, 9, 3, 4],
        [6, 0, 3, 9, 4, 4, 6, 6, 8, 6],
        [3, 3, 0, 9, 2, 1, 5, 9, 1, 4],
        [5, 9, 9, 0, 4, 5, 6, 3, 4, 6],
        [2, 4, 2, 4, 0, 5, 3, 5, 2, 7],
        [7, 4, 1, 5, 5, 0, 4, 6, 1, 8],
        [2, 6, 5, 6, 3, 4, 0, 6, 5, 4],
        [9, 6, 9, 3, 5, 6, 6, 0, 3, 7],
        [3, 8, 1, 4, 2, 1, 5, 3, 0, 5],
        [4, 6, 4, 6, 7, 8, 4, 7, 5, 0]]
    
    '''
    for worker_idx in range(worker_num):
        adjacency_matrix[worker_idx][worker_idx-1] = 1
        adjacency_matrix[worker_idx][(worker_idx+1)%worker_num] = 1
    '''
    for worker_idx1 in range(worker_num):
        for worker_idx2 in range(worker_num):
            if worker_idx1!=worker_idx2:
                adjacency_matrix[worker_idx1][worker_idx2] = 1
    
    p2p_port = np.zeros_like(adjacency_matrix)
    curr_port = p2p_listen_port_base
    for idx_row in range(len(adjacency_matrix)):
        for idx_col in range(len(adjacency_matrix[0])):
            if adjacency_matrix[idx_row][idx_col] != 0:
                curr_port += 1
                p2p_port[idx_row][idx_col] = curr_port  
    print(p2p_port)

    # create workers
    worker_list: List[Worker] = list()
    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
        custom = dict()
        custom["neighbor_info"] = dict()
        custom["average_weight"] = dict()
        custom["computation"] = worker_config["computation"]
        custom["dynamics"] = worker_config["dynamics"]
        worker_list.append(
            Worker(config=ClientConfig(common_config=common_config,custom=custom),
                    idx=worker_idx,
                    client_ip=worker_config['ip_address'],
                    user_name=worker_config['user_name'],
                    pass_wd=worker_config['pass_wd'],
                    remote_scripts_path=workers_config['scripts_path']['remote'],
                    master_port=master_listen_port_base+worker_idx,
                    location='local'
                    )
        )
    #到了这里，worker已经启动了

    for worker_idx, worker_config in enumerate(workers_config['worker_config_list']):
            for neighbor_idx, link in enumerate(adjacency_matrix[worker_idx]):
                if link == 1:
                    neighbor_config = workers_config['worker_config_list'][neighbor_idx]
                    neighbor_name = neighbor_config['user_name']
                    neighbor_ip = "127.0.0.1"
                    #neighbor_ip = neighbor_config['ip_address']
                    worker_list[worker_idx].config.custom["neighbor_info"][neighbor_name] = \
                            (neighbor_ip, p2p_port[worker_idx][neighbor_idx], p2p_port[neighbor_idx][worker_idx])
                    worker_list[worker_idx].config.custom["average_weight"][neighbor_name] = 1.0/worker_num

    for worker in worker_list:
        for neighbor_name in worker.config.custom["neighbor_info"].keys():
            worker.config.weight[neighbor_name] = 1.0/(len(worker.config.custom["neighbor_info"])+1)

    train_data_partition, test_data_partition = partition_data(common_config.dataset_type, common_config.data_pattern)

    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)

    # connect socket and send init config
    communication_parallel(worker_list, action="init")

    recoder: SummaryWriter = SummaryWriter()

    global_model.to(device)
    total_time=0.0
    #local_steps_list=[50,40,50,30,50,40,30,50,40,30]
    #compre_ratio_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    local_steps_list=[50,50,50,50,50,50,50,50,50,50]
    compre_ratio_list=[1,1,1,1,1,1,1,1,1,1]
    computation_resource=[3,1,6,7,7,5,5,2,6,2]
    bandwith_resource=[5,6,8,1,5,8,2,4,4,2]
    #computation_resource=[9,3,5,1,4,6,4,1,4,7]
    #bandwith_resource=[8,7,7,6,9,5,3,3,5,4]
    total_resource=0.0
    total_bandwith=0.0
    #computation_resource,bandwith_resource=random_RC(10)
    path=os.getcwd()
    print (path)
    path=path+"//"+RESULT_PATH
    if not os.path.exists(path):
        os.makedirs(path)
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    path=path+"//"+now+"_record.txt"
    result_out = open(path, 'a+')

    print(common_config.__dict__,file=result_out)
    result_out.write('\n')
    result_out.write("epoch_idx, total_time, total_bandwith, total_resource, acc, test_loss")
    result_out.write('\n')

    #local_steps,compre_ratio=40,0.5
    for epoch_idx in range(1, 1+common_config.epoch):
        communication_parallel(worker_list, action="send_para", data=None)

        is_one_epoch_end=0
        while is_one_epoch_end < worker_num:
            if is_one_epoch_end>0:
                for worker in worker_list:
                    send_data_socket(False, worker.socket)
            for worker in worker_list:
                data=get_data_socket(worker.socket)
                if data == True:
                    is_one_epoch_end+=1

        print("get begin")
        communication_parallel(worker_list, action="send_model", data=True)
        communication_parallel(worker_list, action="get_para")
        print("get end")

        avg_acc = 0.0
        avg_test_loss = 0.0
        avg_train_loss = 0.0
        min_time = 10000.0
        max_time = 0.0
        for worker in worker_list:
            worker_epoch, worker_time, acc, test_loss,train_loss = worker.train_info
            worker.config.train_time=worker_time
            avg_acc += acc
            avg_test_loss += test_loss
            min_time = min(min_time, worker_time)
            max_time = max(max_time,worker_time)
            avg_train_loss += train_loss

        total_time += min_time*50
        #total_time += max_time*30
        avg_acc /= worker_num
        avg_test_loss /= worker_num
        avg_train_loss /= worker_num
        
        recoder.add_scalar('Accuracy/average', avg_acc, epoch_idx)
        recoder.add_scalar('Test_loss/average', avg_test_loss, epoch_idx)
        
        print("Epoch: {}, average accuracy: {}, average test_loss: {}, average train_loss: {}\n".format(epoch_idx, avg_acc, avg_test_loss, avg_train_loss))

        local_steps_list,compre_ratio_list=update_E(worker_list,local_steps_list,compre_ratio_list)
        total_resource=total_resource+Sum(computation_resource,local_steps_list)
        total_bandwith=total_bandwith+Sum(bandwith_resource,compre_ratio_list)
        print(total_time,total_resource,total_bandwith)
        recoder.add_scalar('Accuracy/average_time', avg_acc, total_time)
        recoder.add_scalar('Test_loss/average_time', avg_test_loss, total_time)
        recoder.add_scalar('resource_time', total_resource, total_time)
        recoder.add_scalar('bandwith_time', total_bandwith, total_time)
        recoder.add_scalar('resource_epoch', total_resource, epoch_idx)
        recoder.add_scalar('bandwith_epoch', total_bandwith, epoch_idx)
        result_out.write('{} {:.2f} {:.2f} {:.2f} {:.4f} {:.4f}'.format(epoch_idx,total_time,total_bandwith,total_resource,acc,test_loss))
        result_out.write('\n')
        print(local_steps_list)
        print(compre_ratio_list)
        
    # close socket
    result_out.close()
    for worker in worker_list:
        worker.socket.shutdown(2)
    
def Sum(list1,list2):
    sum=0.0
    for idx in range(0, len(list1)):
        sum=sum+float(list1[idx])*float(list2[idx])
    return sum

def random_RC(num):
    computation_resource=np.random.randint(1,num,num)
    bandwith_resourc=np.random.randint(1,num,num)
    return computation_resource,bandwith_resourc

def update_E(worker_list,local_steps_list,compre_ratio_list):
    local_steps=random.randint(40,60)
    compre_ratio=local_steps/200.0
    train_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    send_time_list=[0.8,0.7,0.8,0.6,0.8,0.7,0.6,0.8,0.7,0.6]
    min_train_time=10000.0
    min_train_time_idx=1
    min_send_time=10000.0
    min_send_time_idx=1
    sum_local_steps=0
    for worker in worker_list:
        train_time_list[worker.idx]=worker.config.train_time
        send_time_list[worker.idx]=worker.config.send_time
        if train_time_list[worker.idx]<min_train_time:
            min_train_time=train_time_list[worker.idx]
            min_train_time_idx=worker.idx
        if send_time_list[worker.idx]<min_send_time:
            min_send_time=send_time_list[worker.idx]
            min_send_time_idx=worker.idx
    for worker in worker_list:
        worker.config.local_steps=int((train_time_list[min_train_time_idx]/train_time_list[worker.idx])*local_steps)
        worker.config.compre_ratio=(train_time_list[min_train_time_idx]/train_time_list[worker.idx])*compre_ratio
        #(send_time_list[min_train_time_idx]/send_time_list[worker.idx])*compre_ratio
        worker.config.local_steps=50
        #worker.config.local_steps=int(local_steps/2)+3
        worker.config.compre_ratio=0.3
        local_steps_list[worker.idx]=worker.config.local_steps
        compre_ratio_list[worker.idx]=worker.config.compre_ratio
        sum_local_steps=sum_local_steps+worker.config.local_steps
    for worker in worker_list:
        worker.config.average_weight=(1.0*worker.config.local_steps)/(sum_local_steps)
        
    max_train_time=max(train_time_list)
    max_send_time=max(send_time_list)
    total_time=max_train_time*30*0.4
    #local_steps/2*0.9
    #total_time=min_train_time*50+min_train_time*40
    #total_time=min_train_time*local_steps/2.0
    return local_steps_list,compre_ratio_list

def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list),)
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get_para":
                tasks.append(loop.run_in_executor(executor, worker.get_config))
            elif action == "send_model":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
            elif action == "send_para":
                data=(worker.config.local_steps,worker.config.compre_ratio)
                tasks.append(loop.run_in_executor(executor, worker.send_data,data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)

def non_iid_partition(ratio, worker_num=10):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num-1))

    for worker_idx in range(worker_num):
        partition_sizes[worker_idx][worker_idx] = ratio

    return partition_sizes

def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR100" or dataset_type == "image100":
        test_partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((100, worker_num)) * (1 / (worker_num-data_pattern))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx
            for _ in range(data_pattern):
                partition_sizes[tmp_idx*worker_num:(tmp_idx+1)*worker_num, worker_idx] = 0
                tmp_idx = (tmp_idx + 1) % 10
                
    elif dataset_type == 'tinyImageNet':
        test_partition_sizes = np.ones((200, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((200, worker_num))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx*20
            for _ in range(int(data_pattern/10)):
                partition_sizes[tmp_idx:tmp_idx+10, worker_idx] = 0
                tmp_idx = (tmp_idx + 10) % 200
        axis = np.sum(partition_sizes, axis=1)
        for i in range(200):
            for j in range(worker_num):
                if partition_sizes[i][j] == 1:
                    partition_sizes[i][j] = 1/axis[i]

    elif dataset_type == "EMNIST":
        test_partition_sizes = np.ones((62, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((62, worker_num))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx*6
            for _ in range(int(data_pattern/2)):
                partition_sizes[tmp_idx:tmp_idx+2, worker_idx] = 0
                tmp_idx = (tmp_idx + 2) % 62
        axis = np.sum(partition_sizes, axis=1)
        for i in range(62):
            for j in range(worker_num):
                if partition_sizes[i][j] == 1:
                    partition_sizes[i][j] = 1/axis[i]

    elif dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)
        if data_pattern == 0:
            partition_sizes = np.ones((10, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            partition_sizes = [
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.1482,0.111],
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.1482,0.111],
                                [0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482, 0.1482, 0.1482, 0.148,0.111],
                                [0.148, 0.1482, 0.1482, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482,0.111],
                                [0.1482, 0.148, 0.1482, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1482,0.111],
                                [0.1482, 0.1482, 0.148, 0.0,    0.0,    0.0,    0.1482, 0.1482, 0.1472,0.112],
                                [0.1482,  0.1482, 0.1482, 0.148, 0.1482, 0.1482, 0.0,    0.0,    0.0  , 0.111],
                                [0.1482,  0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.0,    0.0,    0.0  , 0.111],
                                [0.1482,  0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.0,    0.0,    0.0  , 0.111],
                                [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.112, 0.0],
                                ]
        elif data_pattern == 2:
            partition_sizes = [
                    [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                    [0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                    [0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                    ]
        elif data_pattern == 3:
            partition_sizes = [[0.1428,  0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0],
                                [0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0],
                                [0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0],
                                [0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432],
                                [0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428, 0.1428],
                                [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0,    0.0,    0.0,    0.1428],
                                ]
        elif data_pattern == 4:
            partition_sizes = [[0.125,  0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0],
                                [0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0],
                                [0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125, 0.125],
                                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0,   0.0,   0.125],
                                ]
        elif data_pattern == 5:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 6:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 7:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 8:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 9:
            non_iid_ratio = 0.9
            partition_sizes = non_iid_partition(non_iid_ratio)
        # elif data_pattern == 10:
        #     non_iid_ratio = 0.5
        #     partition_sizes = non_iid_partition(non_iid_ratio)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    # test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)
    
    return train_data_partition, test_data_partition

if __name__ == "__main__":
    main()
