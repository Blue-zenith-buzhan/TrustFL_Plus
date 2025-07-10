import os
import warnings
import argparse
import numpy as np
import torch
import shutil
import torch.nn as nn
import random
import json
import pickle
import Net
import Net.VGG16
# import Net.transformer
import Net.resnet50
import Net.VGG19
import Net.GPT2
import torchvision.transforms as transforms

class roundEnvironment:
    def __init__(self):
        self.max_iter,\
        self.save_random_seeds_flag,\
        self.seed_path,\
        self.random_seeds_json,\
        self.batch_size,\
        self.ckpt_base_path,\
        self.round_amount,\
        self.round_mode,\
        self.round_path,\
        self.net_name,\
        self.rng_state,\
        self.verify,\
        self.save_parameter = get_args()

def initEnvironment():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description = "arguments for verifiable training")
    parser.add_argument("-mi", "--max_iter", default=200, type=int, help = "max iter")
    parser.add_argument("-srs", "--save_or_get_random_seeds", default=0, type=int, help = "save random seeds -> 1; get random seeds -> 0")
    parser.add_argument("-rsj", "--random_seeds_json", default="./seed/seed.json", help = "random seeds json path")
    parser.add_argument("-sp", "--prf_seed_path", default="./seed/seed", help = "prf seed path")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help = "batch size") #VGG16 128, transformer 64, ResNet 64
    parser.add_argument("-bp", "--ckpt_base_path", default="./Parameter/", help = "ckpt base path")
    parser.add_argument("-ra", "--round_amount", default=28, type=int, help = "round amount")
    parser.add_argument("-m", "--mode", default=1, type=int, help = "save round log -> 1")
    parser.add_argument("-rp", "--round_path", default="./round_log", help = "round log path")
    parser.add_argument("-net", "--net", default="GPT2", help = "choose net to train or test")
    parser.add_argument("-rgp", "--random_state_path", default="./rng_state", help = "random state path")
    parser.add_argument("-vf", "--if_verify", default=0, type=int, help = "if verify")
    parser.add_argument("-ps", "--save_parameter", default=0, type=int, help = "whether to save parameter")

    args = parser.parse_args()
    max_iter = args.max_iter
    save_random_seeds_flag = args.save_or_get_random_seeds
    random_seeds_json = args.random_seeds_json
    seed_path = args.prf_seed_path
    batch_size = args.batch_size
    ckpt_base_path = args.ckpt_base_path
    round_amount = args.round_amount
    round_mode = args.mode
    round_path = args.round_path
    use_net = args.net
    rng_state = args.random_state_path
    if_verify = args.if_verify
    save_parameter = args.save_parameter
    if if_verify:
        round_mode = 0
        save_random_seeds_flag = 0

    return max_iter, save_random_seeds_flag, seed_path, random_seeds_json, batch_size, ckpt_base_path, round_amount, round_mode, round_path, use_net, rng_state, if_verify, save_parameter

def prf(seed_val, name=None):
    seed_val = np.array(seed_val, dtype='uint32')
    seed_val = seed_val * np.array(1103515245, dtype='uint32') + np.array(12345, dtype='uint32')
    return int(seed_val // 65536)

def float64_to_binary(value):
    """将 float64 转换为 IEEE 754 二进制表示的字符串"""
    # 使用 numpy 将值转换为 64 位浮点数的二进制表示
    [binary] = np.array([value], dtype=np.float64).view(np.uint64)
    return f"{binary:064b}"  # 转换为 64 位二进制字符串

def save_conv(conv, path="par.txt", bin_style=False):
    if bin_style:
        ii = conv.weight.shape[0]
        jj = conv.weight.shape[1]
        with open(path, "w") as f:  # 使用文本模式写入
            # 保存卷积层权重
            for i in range(ii):
                for j in range(jj):
                    for weight in conv.weight[i][j].cpu().detach().numpy().astype(np.float64).flat:
                        f.write(float64_to_binary(weight) + "\n")
            # 保存卷积层偏置
            for bias in conv.bias.cpu().detach().numpy().astype(np.float64).flat:
                f.write(float64_to_binary(bias) + "\n")
        return
    fmtt = '%s'
    ii = conv.weight.shape[0]
    jj = conv.weight.shape[1]
    with open(path, "wb") as f:
        for i in range(ii):
            for j in range(jj):
                np.savetxt(f, conv.weight[i][j].cpu().detach().numpy(), fmt=fmtt)
        np.savetxt(f, conv.bias.cpu().detach().numpy().reshape((1, -1)), fmt=fmtt)

def save_fc(fc, path="fc.txt", bin_style=False):
    if bin_style:
        with open(path, "w") as f:
            # 保存权重
            for weight in fc.weight.cpu().detach().numpy().astype(np.float64).flat:
                f.write(float64_to_binary(weight) + "\n")
            # 保存偏置
            for bias in fc.bias.cpu().detach().numpy().astype(np.float64).flat:
                f.write(float64_to_binary(bias) + "\n")
        return
    fmtt = '%s'
    with open(path, "wb") as f:
        np.savetxt(f, fc.weight.cpu().detach().numpy(), fmt=fmtt)
        np.savetxt(f, fc.bias.cpu().detach().numpy().reshape((1, -1)), fmt=fmtt)

def save_net_VGG16(net, base_path, bin_style=False):
    folder = os.path.exists(base_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(base_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        save_conv(net.conv1, base_path + "/conv1.txt", bin_style)
        save_conv(net.conv2, base_path + "/conv2.txt", bin_style)
        save_fc(net.fc1, base_path + "/fc1.txt", bin_style)
        save_fc(net.fc2, base_path + "/fc2.txt", bin_style)
        save_fc(net.fc3, base_path + "/fc3.txt", bin_style)

def init_checkpoint(path="./Parameter/"):
    if not os.path.exists(path):
    #     if os.listdir(path):
    #         for root, dirs, files in os.walk(path, topdown=False):
    #             for name in files:
    #                 os.remove(os.path.join(root, name))
    #             for name in dirs:
    #                 os.rmdir(os.path.join(root, name))
    # else:
        os.makedirs(path)

def init_write_log(folder_path='./round_log'):
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 删除文件夹
            # shutil.rmtree(folder_path) 
            # 重新创建文件夹
            os.makedirs(folder_path)
    except Exception as e:
        print(f"操作文件夹时出错：{e}")
    return

def init_rng_state_log(folder_path='./rng_state', round_mode=1):
    if round_mode:
        try:
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        except Exception as e:
            print(f"操作文件夹时出错：{e}")
    return

def save_random_seeds_to_file(file_path):
    # 保存当前随机种子值
    seed_dict = {
        "torch_cuda_seed": torch.cuda.initial_seed() if torch.cuda.is_available() else None,
        "torch_seed": torch.initial_seed(),
        "np_seed": np.random.randint(0, 2**31 - 1),
        "random_seed": random.randint(0, 2**31 - 1),
    }
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    torch.use_deterministic_algorithms(True) 
    #disable tensor float32
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False 

    # 将种子字典保存到文件
    with open(file_path, "w") as f:
        json.dump(seed_dict, f)

def load_random_seeds_from_file(file_path):
    try:
        # 从文件加载种子字典
        with open(file_path, "r") as f:
            seed_dict = json.load(f)

        # 恢复随机种子
        if "torch_cuda_seed" in seed_dict and seed_dict["torch_cuda_seed"] is not None:
            torch.cuda.manual_seed(seed_dict["torch_cuda_seed"])
        if "torch_seed" in seed_dict:
            torch.manual_seed(seed_dict["torch_seed"])
        if "np_seed" in seed_dict:
            np.random.seed(seed_dict["np_seed"])
        if "random_seed" in seed_dict:
            random.seed(seed_dict["random_seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        
        torch.use_deterministic_algorithms(True) 
        #disable tensor float32
        torch.backends.cuda.matmul.allow_tf32 = False 
        torch.backends.cudnn.allow_tf32 = False 
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，无法加载随机种子。")

def init_random_seed_and_device(save_random_seeds_flag, seed_path, random_seeds_json, if_verify):

    with open(seed_path, "r") as f:
        seed = f.read()
        seed = int(seed)

    if (save_random_seeds_flag):
        save_random_seeds_to_file(random_seeds_json)
    else:
        load_random_seeds_from_file(random_seeds_json)
        
    if if_verify:
        device = torch.device("cpu")
        print("[Trusted]: start training with device:{}, seed:{}".format(device, seed))
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("[Untrusted]: start training with device:{}, seed:{}".format(device, seed))
    
    return device, seed

def save_rng_state(file_path):
    """
    保存 PyTorch 的随机状态到指定文件。
    
    参数:
        file_path (str): 保存随机状态的文件路径。
    """
    # 获取当前的随机状态
    rng_state = torch.get_rng_state()
    # 使用 pickle 将随机状态保存到文件
    with open(file_path, 'wb') as f:
        pickle.dump(rng_state, f)
    # print(f"随机状态已保存到文件：{file_path}")

def load_rng_state(file_path):
    """
    从指定文件加载 PyTorch 的随机状态。
    
    参数:
        file_path (str): 包含随机状态的文件路径。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    
    # 从文件中加载随机状态
    with open(file_path, 'rb') as f:
        rng_state = pickle.load(f)
    # 设置随机状态
    torch.set_rng_state(rng_state)
    # print(f"随机状态已从文件 {file_path} 恢复")

def save_net(name, path, optimizer, times, net):
    torch.save({'epoch': times,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate()}, path + '/' + name + '_' + str(times) + '.pth')

def load_net(device, name, path, times, net, optimizer):
    checkpoint = torch.load(path + '/' + name + '_' + str(times) + '.pth', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['torch_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['python_rng_state'])

def random_epoch(epoch):
    # 使用 SystemRandom 类生成真随机数
    true_random = random.SystemRandom()

    # 生成五个指定范围内的随机整数
    random_numbers = [true_random.randint(0, epoch - 1) for _ in range(int(epoch * 0.025))]

    return random_numbers

def append_to_loss(num, file_name):
    if not os.path.exists(file_name):
        # 文件不存在，创建文件并写入num
        with open(file_name, "w") as file:
            file.write(f"{num}\n")
    else:
        # 文件已存在，追加一行记录num
        with open(file_name, "a") as file:
            file.write(f"{num}\n")

def init_Net(name, vocab_size=None):
    if name == "VGG16":
        return Net.VGG16.DNN()
    if name == "VGG19":
        return Net.VGG19.DNN()
    if name == "transformer":
        return Net.transformer.TransformerDecoderModel(vocab_size=vocab_size)
    if name == "ResNet":
        return Net.resnet50.ResNet(Net.resnet50.ResidualBlock, [3, 4, 6, 3])
    if name == "GPT2":
        return Net.GPT2.GPT2(
            vocab_size=vocab_size,
            embed_dim=768,
            num_heads=12,
            num_layers=6,
            ff_dim=3072,
            max_seq_len=512
        )