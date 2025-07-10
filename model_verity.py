from initDevice import *
from roundTool import *
from Net.GPT2Train import download_and_process_dataset, create_training_sequences
from hash import model_hash_sha256
import torch.optim as optim
import time

def verify(device, seed, rE=roundEnvironment()):
    if rE.net_name == "GPT2":
        # 调用函数获取数据
        full_corpus, tokenizer = download_and_process_dataset()

        # 创建训练样本
        dataset = create_training_sequences(full_corpus, seq_length=512)
        # verify_epoch = random_epoch(rE.max_iter)
        verify_epoch = [0,1,2,3,4,5,6,7,8,9]
        print(f"[Trusted]: Epoch {verify_epoch} are going to be verified.")

        rT = roundTool(rE.round_amount, rE.round_mode)
        rT.set_verify_state()

        net = init_Net(rE.net_name, tokenizer.get_vocab_size())
        
        total_steps = len(dataset) // rE.batch_size
        net.to(torch.float32).to(torch.float64).to(device)
        net.train()

        for t in range(len(verify_epoch)):
            current_time = time.perf_counter()
            forward_hooks = rT.add_forward_hooks(net)
            backward_hooks = rT.add_backward_hooks(net)
            backward_hooks_ = []
            i = verify_epoch[t]

            input_seqs = []
            target_seqs = []
            epoch = i // total_steps
            
            optimizer = optim.SGD(net.parameters(), lr=1e-4)

            for j in range(rE.batch_size):
                # 生成伪随机索引
                idx = prf(seed * (epoch + 1) + i * rE.batch_size + j) % len(dataset)
                
                # 获取数据
                in_seq, tar_seq = dataset[idx]
                
                # 转换为Tensor
                in_tensor = torch.LongTensor(in_seq)
                tar_tensor = torch.LongTensor(tar_seq)
                
                input_seqs.append(in_tensor)
                target_seqs.append(tar_tensor)

            # 堆叠batch数据
            inputs = torch.stack(input_seqs).to(device)
            targets = torch.stack(target_seqs).to(device)

            rT.round_csv_to_arrays(times=i, path=rE.round_path, name=rE.net_name)
            load_rng_state(rE.rng_state + '/' + rE.net_name + '_' + str(i))
            load_net(device, rE.net_name, rE.ckpt_base_path, i, net, optimizer)
            print(f"[Trusted]: Epoch {i} is being verified.")

            backward_hooks_ = []
            
            optimizer.zero_grad()

            # 前向传播
            logits = net(inputs)

            backward_hooks_.append(logits.register_hook(partial(rT.rounding_backward_hook_64, "output")))
            
            # 计算损失（交叉熵）
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.token_to_id("[PAD]"))

            backward_hooks_.append(loss.register_hook(partial(rT.rounding_backward_hook_64, "loss")))

            loss = loss.to(dtype=torch.float32).to(dtype=torch.float64)
            # 反向传播
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            del loss
            del logits
            del input_seqs
            del target_seqs

            with torch.no_grad():
                for n, p in net.named_parameters():
                    p.data = rT.round_to_low_bits_final(p.data)

            for h in backward_hooks_:
                h.remove()

            rT.remove_hooks(forward_hooks)
            rT.remove_hooks(backward_hooks)

            net1_hash = model_hash_sha256(net)

            net2 = init_Net(rE.net_name, tokenizer.get_vocab_size())
            net2 = net2.to(torch.float32).to(torch.float64)
            net2 = net2.to(device)
            load_net(device, rE.net_name, rE.ckpt_base_path, i + 1, net2, optimizer)
            net2_hash = model_hash_sha256(net2)

            del net2

            if net1_hash == net2_hash:
                print("两个模型的参数一致")
            else:
                print("两个模型的参数不一致")
            now = time.perf_counter()
            print(f"time used: {now - current_time}")
    # if rE.net_name == "VGG16":
    #     trainset, testloader = Net.VGG16.load_data(batch_size=rE.batch_size)

    #     verify_epoch = random_epoch(rE.max_iter)
    #     verify_epoch = [0, 1, 2, 3, 4]     
    #     print(f"[Trusted]: Epoch {verify_epoch} are going to be verified.")

    #     net = init_Net(rE.net_name)
    #     net = net.to(torch.float32).to(torch.float64)
    #     net = net.to(device)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(net.parameters(), lr=0.001)

    #     for t in range(len(verify_epoch)):

    #         i = verify_epoch[t]
    #         print(f"[Trusted]: Epoch {i} is being verified.")
            
    #         load_rng_state(rE.rng_state + '/' + rE.net_name + '_' + str(i))
    #         load_net(device, rE.net_name, rE.ckpt_base_path, i, net, optimizer)

    #         inputs = torch.zeros((rE.batch_size, 3, 224, 224))
    #         labels = torch.LongTensor(rE.batch_size)
    #         for j in range(rE.batch_size):
    #             index = prf(seed * i + j, rE.net_name) % len(trainset)  # 确保index在有效范围内
    #             labels[j] = trainset[index][1]
    #             inputs[j] = trainset[index][0]
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         inputs = inputs.to(torch.float32).to(torch.float64)

    #         optimizer.zero_grad()

    #         outputs = net(inputs)

    #         loss = criterion(outputs, labels)

    #         loss = loss.to(dtype=torch.float32).to(dtype=torch.float64)

    #         loss.backward()

    #         optimizer.step()

    #         net2 = init_Net(rE.net_name)
    #         net2 = net2.to(torch.float32).to(torch.float64)
    #         net2 = net2.to(device)
    #         load_net(device, rE.net_name, rE.ckpt_base_path, i + 1, net2, optimizer)

    #         if compare_models(net, net2):
    #             print("两个模型的参数一致")
    #         else:
    #             print("两个模型的参数不一致")

    # if rE.net_name == "VGG19":
    #     trainset, testloader = Net.VGG19.load_data(batch_size=rE.batch_size)

    #     verify_epoch = random_epoch(rE.max_iter)
    #     verify_epoch = [0, 1, 2, 3, 4]     
    #     print(f"[Trusted]: Epoch {verify_epoch} are going to be verified.")

    #     rT = roundTool(rE.round_amount, rE.round_mode)
    #     rT.set_verify_state()

    #     net = init_Net(rE.net_name)
    #     net = net.to(torch.float32).to(torch.float64)
    #     net = net.to(device)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(net.parameters(), lr=0.001)

    #     for t in range(len(verify_epoch)):

    #         forward_hooks = rT.add_forward_hooks(net)
    #         backward_hooks = rT.add_backward_hooks(net)
    #         backward_hooks_ = []

    #         i = verify_epoch[t]
    #         print(f"[Trusted]: Epoch {i} is being verified.")
    #         rT.round_csv_to_arrays(times=i, path=rE.round_path, name=rE.net_name)
            
    #         load_rng_state(rE.rng_state + '/' + rE.net_name + '_' + str(i))
    #         load_net(device, rE.net_name, rE.ckpt_base_path, i, net, optimizer)

    #         inputs = torch.zeros((rE.batch_size, 3, 224, 224))
    #         labels = torch.LongTensor(rE.batch_size)
    #         for j in range(rE.batch_size):
    #             index = prf(seed * i + j, rE.net_name) % len(trainset)  # 确保index在有效范围内
    #             labels[j] = trainset[index][1]
    #             inputs[j] = trainset[index][0]
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         inputs = inputs.to(torch.float32).to(torch.float64)

    #         optimizer.zero_grad()

    #         outputs = net(inputs)

    #         backward_hooks_.append(outputs.register_hook(partial(rT.rounding_backward_hook_64, "output")))

    #         loss = criterion(outputs, labels)

    #         backward_hooks_.append(loss.register_hook(partial(rT.rounding_backward_hook_64, "loss")))

    #         loss = loss.to(dtype=torch.float32).to(dtype=torch.float64)

    #         loss.backward()

    #         optimizer.step()

    #         with torch.no_grad():
    #             for n, p in net.named_parameters():
    #                 p.data = rT.round_to_low_bits_final(p.data)
    #         for h in backward_hooks_:
    #             h.remove()

    #         rT.remove_hooks(forward_hooks)
    #         rT.remove_hooks(backward_hooks)
    #         net2 = init_Net(rE.net_name)
    #         net2 = net2.to(torch.float32).to(torch.float64)
    #         net2 = net2.to(device)
    #         load_net(device, rE.net_name, rE.ckpt_base_path, i + 1, net2, optimizer)

    #         if compare_models(net, net2):
    #             print("两个模型的参数一致")
    #         else:
    #             print("两个模型的参数不一致")

    # if rE.net_name == "ResNet":
    #     trainset, testloader = Net.resnet50.load_data(rE.batch_size)     
    #     verify_epoch = [0, 1, 2, 3, 4]         
    #     print(f"[Trusted]: Epoch {verify_epoch} are going to be verified.")

    #     rT = roundTool(rE.round_amount, rE.round_mode)
    #     rT.set_verify_state()
    #     net = init_Net(rE.net_name)
    #     net.to(device).to(torch.float64)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(net.parameters(), lr=0.00003)
    #     # optimizer = optim.Adam(net.parameters(), lr=0.00003)
    #     net.train()

    #     for t in range(len(verify_epoch)):

    #         forward_hooks = rT.add_forward_hooks(net)
    #         backward_hooks = rT.add_backward_hooks(net)
    #         backward_hooks_ = []

    #         i = verify_epoch[t]
    #         print(f"[Trusted]: Epoch {i} is being verified.")
    #         rT.round_csv_to_arrays(times=i, path=rE.round_path, name=rE.net_name)
            
    #         load_rng_state(rE.rng_state + '/' + rE.net_name + '_' + str(i))
    #         load_net(device, rE.net_name, rE.ckpt_base_path, i, net, optimizer)

    #         inputs = torch.zeros((rE.batch_size, 3, 32, 32))
    #         labels = torch.LongTensor(rE.batch_size)
    #         for j in range(rE.batch_size):
    #             index = prf(seed * i + j) % len(trainset)
    #             labels[j] = trainset[index][1]
    #             inputs[j] = trainset[index][0]
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         inputs = inputs.to(torch.float32).to(torch.float64)

    #         optimizer.zero_grad()

    #         outputs = net(inputs)

    #         backward_hooks_.append(outputs.register_hook(partial(rT.rounding_backward_hook_64, "output")))

    #         loss = criterion(outputs, labels)

    #         backward_hooks_.append(loss.register_hook(partial(rT.rounding_backward_hook_64, "loss")))

    #         loss = loss.to(dtype=torch.float32).to(dtype=torch.float64)

    #         loss.backward()

    #         optimizer.step()

    #         with torch.no_grad():
    #             for n, p in net.named_parameters():
    #                 p.data = rT.round_to_low_bits(p.data)
    #         for h in backward_hooks_:
    #             h.remove()

    #         rT.remove_hooks(forward_hooks)
    #         rT.remove_hooks(backward_hooks)
    #         net2 = init_Net(rE.net_name)
    #         net2 = net2.to(torch.float32).to(torch.float64)
    #         net2 = net2.to(device)
    #         load_net(device, rE.net_name, rE.ckpt_base_path, i + 1, net2, optimizer)

    #         if compare_models(net, net2):
    #             print("两个模型的参数一致")
    #         else:
    #             print("两个模型的参数不一致")
        
    # # if rE.net_name == "transformer":
    # #     dataset = Net.transformer.TextDataset('data/sentence.txt')

    # #     verify_epoch = [0,1,2,3,4,5]
    # #     print(f"[Trusted]: Epoch {verify_epoch} are going to be verified.")
        
    # #     rT = roundTool(rE.round_amount, rE.round_mode)
    # #     rT.set_verify_state()

    # #     net = init_Net(rE.net_name, dataset.vocab_size)
    # #     net.to(device).to(torch.float64)
    # #     criterion = nn.CrossEntropyLoss()
    # #     net.train()
    # #     for t in range(len(verify_epoch)):
    # #         forward_hooks = rT.add_forward_hooks(net)
    # #         backward_hooks = rT.add_backward_hooks(net)
    # #         backward_hooks_ = []
    # #         i = verify_epoch[t]

    # #         # 用来收集本轮 iteration 中所有样本的输入与目标
    # #         input_seqs = []
    # #         target_seqs = []
    # #         # optimizer = optim.Adam(net.parameters(), lr=0.0005)
    # #         optimizer = optim.SGD(net.parameters(), lr=0.0005)
            
    # #         # 依次根据 prf 生成索引，手动取出 dataset 中的那条记录
    # #         for j in range(rE.batch_size):
    # #             # 生成下标时要注意可能越界，
    # #             # 可以对 dataset 的长度取一个模，防止 index 越界
    # #             idx = prf(seed * i + j) % len(dataset)
                
    # #             # 从 dataset 中取出一条数据
    # #             in_seq, tar_seq = dataset[idx]
                
    # #             input_seqs.append(in_seq)
    # #             target_seqs.append(tar_seq)
                        
    # #         # 把刚才收集到的 input 序列合并成一个 batch
    # #         # 使用 pad_sequence 进行填充
    # #         inputs_padded = Net.transformer.pad_sequence(input_seqs, batch_first=True, padding_value=0)
    # #         # 目标可以直接堆叠（如果都是一维或者标量）
    # #         targets_stacked = torch.stack(target_seqs, dim=0)
    # #         # 将它们放到设备上
    # #         inputs = inputs_padded.to(device)
    # #         targets = targets_stacked.to(device)
    # #         inputs = inputs.t() 

    # #         rT.round_csv_to_arrays(times=i, path=rE.round_path, name=rE.net_name)
    # #         load_rng_state(rE.rng_state + '/' + rE.net_name + '_' + str(i))
    # #         load_net(device, rE.net_name, rE.ckpt_base_path, i, net, optimizer)
    # #         print(f"[Trusted]: Epoch {i} is being verified.")

    # #         optimizer.zero_grad()

    # #         outputs = net(inputs)
    # #         outputs = outputs[-1] 

    # #         # encdata = torch.load('encdata.pth')
    # #         # encint64 = encdata.view(torch.int64)
    # #         # oenc = net.positional_encoding.data.view(torch.int64)

    # #         backward_hooks_.append(outputs.register_hook(partial(rT.rounding_backward_hook_64, "output")))

    # #         loss = criterion(outputs, targets)

    # #         backward_hooks_.append(loss.register_hook(partial(rT.rounding_backward_hook_64, "loss")))

    # #         loss = loss.to(dtype=torch.float32).to(dtype=torch.float64)

    # #         loss.backward()
                
    # #         optimizer.step()

    # #         with torch.no_grad():
    # #             for n, p in net.named_parameters():
    # #                 p.data = rT.round_to_low_bits_final(p.data)
    # #         for h in backward_hooks_:
    # #             h.remove()

    # #         rT.remove_hooks(forward_hooks)
    # #         rT.remove_hooks(backward_hooks)
    # #         net2 = init_Net(rE.net_name, dataset.vocab_size)
    # #         net2 = net2.to(torch.float32).to(torch.float64)
    # #         net2 = net2.to(device)
    # #         load_net(device, rE.net_name, rE.ckpt_base_path, i + 1, net2, optimizer)

    # #         if compare_models(net, net2):
    # #             print("两个模型的参数一致")
    # #         else:
    # #             print("两个模型的参数不一致")



def compare_models(model1, model2):
    k = True
    m = 0
    for key_item1, key_item2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if key_item1[0] != key_item2[0]:
            print(f"参数名不一致: {key_item1[0]} vs {key_item2[0]}")
            return False
        if not torch.equal(key_item1[1], key_item2[1]) and not 'running_mean' in key_item1[0] and not 'running_var' in key_item1[0]:
            # print(f"参数值不一致: {key_item1[0]}")
            # k1 = key_item1[1].view(torch.int64)
            # k2 = key_item2[1].view(torch.int64)
            k1 = key_item1[1]
            k2 = key_item2[1]
            # print(f"key_item1: {key_item1[1].view(torch.int64)}")
            # print(f"key_item2: {key_item2[1].view(torch.int64)}")
            # abk1 = torch.where(k1 != k2, k1, 999).flatten()
            # abk2 = torch.where(k1 != k2, k2, 999).flatten()
            # abk1 = abk1[abk1!=999]
            # abk2 = abk2[abk2!=999]
            # 去掉 0 元素
            # print(f"key_item1: {k1}")
            # print(f"key_item2: {k2}")
            # print(f"abnormal key_item1: {abk1}")
            # print(f"abnormal key_item2: {abk2}")
            # print(f"max: {torch.max(torch.abs(k1 - k2))}")
            m = max(torch.max(torch.abs(k1 - k2)), m)
            k = False
    print(f"m is : {m}")
    if k:
        return True
    else:
        return False
    
def find_max_mantissa_diff(a, b):
    assert a.shape == b.shape, "张量形状必须相同"
    
    # 将张量转换为int64以访问位表示
    a_int = a.view(torch.int64)
    b_int = b.view(torch.int64)
    
    # 尾数掩码
    mantissa_mask = 0x000FFFFFFFFFFFFF
    
    # 提取尾数部分
    mantissa_a = a_int & mantissa_mask
    mantissa_b = b_int & mantissa_mask
    
    # 计算绝对差值并找到最大值的位置
    diff = (mantissa_a - mantissa_b).abs()
    max_diff, linear_idx = torch.max(diff.view(-1), dim=0)
    
    # 转换为多维索引
    idx = torch.unravel_index(linear_idx, a.shape)
    
    # 获取对应的原始数值
    a_val = a[idx].item()
    b_val = b[idx].item()
    
    return idx, a_val, b_val, max_diff.item()