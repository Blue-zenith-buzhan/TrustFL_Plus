import time
from tqdm import tqdm
from initDevice import *
from roundTool import *
from Net.GPT2Train import download_and_process_dataset, create_training_sequences
import torch.optim as optim

def train(device, seed, rE=roundEnvironment()):
    if rE.net_name == "GPT2":
        # 调用函数获取数据
        full_corpus, tokenizer = download_and_process_dataset()

        # 创建训练样本
        dataset = create_training_sequences(full_corpus, seq_length=512)
        # print(f"创建 {len(dataset)} 个训练样本")

        net = init_Net(rE.net_name, tokenizer.get_vocab_size())
        net.to(torch.float32).to(torch.float64).to(device)
        optimizer = optim.AdamW(net.parameters(), lr=1e-4)
        net.train()
        rT = roundTool(rE.round_amount, rE.round_mode)
        forward_hooks = rT.add_forward_hooks(net)
        backward_hooks = rT.add_backward_hooks(net)

        total_steps = len(dataset) // rE.batch_size
        print("[Untrusted]: Training...")

        total_loss = 0

        if rE.save_parameter:
            save_net(rE.net_name, rE.ckpt_base_path, optimizer, 0, net)
            save_rng_state(rE.rng_state + '/' + rE.net_name + '_0')

        for i in tqdm(range(rE.max_iter * 100)):
            current_time = time.perf_counter()
            # 生成当前batch的样本
            input_seqs = []
            target_seqs = []
            epoch = i // total_steps
            
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
            
            backward_hooks_ = []
            
            torch.cuda.synchronize()
            optimizer.zero_grad()
            torch.cuda.synchronize()
        
            # 前向传播
            logits = net(inputs)

            backward_hooks_.append(logits.register_hook(partial(rT.rounding_backward_hook_64, "output")))
            
            # 计算损失（交叉熵）
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.token_to_id("[PAD]"))
            
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                with open("loss_GPT2.txt", "a", encoding="utf-8") as file:
                    # 遍历列表中的每个元素
                    file.write(str(total_loss/100) + "\n")
                    total_loss = 0

            backward_hooks_.append(loss.register_hook(partial(rT.rounding_backward_hook_64, "loss")))

            loss = loss.to(dtype=torch.float32).to(dtype=torch.float64)


            # 反向传播
            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            with torch.no_grad():
                for n, p in net.named_parameters():
                    p.data = rT.round_to_low_bits_final(p.data)

            now = time.perf_counter()

            for h in backward_hooks_:
                h.remove()
                
            if rE.save_parameter:
                save_net(rE.net_name, rE.ckpt_base_path, optimizer, i + 1, net)
                rT.save_round_log(times=i, path=rE.round_path, name=rE.net_name)
                save_rng_state(rE.rng_state + '/' + rE.net_name + '_' + str(i + 1))

            rT.free_log()
        rT.remove_hooks(forward_hooks)
        rT.remove_hooks(backward_hooks)

    if rE.net_name == "VGG16":
        trainset, testloader = Net.VGG16.load_data(batch_size=rE.batch_size)

        net = init_Net(rE.net_name)
        net = net.to(torch.float32).to(torch.float64)
        net = net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001)
        net.train()
        
        rT = roundTool(rE.round_amount, rE.round_mode)

        forward_hooks = rT.add_forward_hooks(net)
        backward_hooks = rT.add_backward_hooks(net)

        print("[Untrusted]: Training...")
        
        if rE.save_parameter:
            save_net(rE.net_name, rE.ckpt_base_path, optimizer, 0, net)
            save_rng_state(rE.rng_state + '/' + rE.net_name + '_0')
        for i in tqdm(range(rE.max_iter)):
            inputs = torch.zeros((rE.batch_size, 3, 224, 224))
            labels = torch.LongTensor(rE.batch_size)
            for j in range(rE.batch_size):
                index = prf(seed * i + j, rE.net_name) % len(trainset)  # 确保index在有效范围内
                labels[j] = trainset[index][1]
                inputs[j] = trainset[index][0]
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.to(torch.float32).to(torch.float64)
            # zero the parameter gradients
            torch.cuda.synchronize()
            optimizer.zero_grad()
            torch.cuda.synchronize()

            backward_hooks_ = []

            outputs = net(inputs)

            backward_hooks_.append(logits.register_hook(partial(rT.rounding_backward_hook_64, "output")))

            loss = criterion(outputs, labels)

            append_to_loss(loss.item(), "loss" + str(rE.round_amount) + ".txt")

            backward_hooks_.append(loss.register_hook(partial(rT.rounding_backward_hook_64, "loss")))

            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()

            with torch.no_grad():
                for n, p in net.named_parameters():
                    p.data = rT.round_to_low_bits_final(p.data)

            for h in backward_hooks_:
                h.remove()

            if rE.save_parameter:
                save_net(rE.net_name, rE.ckpt_base_path, optimizer, i + 1, net)
                save_rng_state(rE.rng_state + '/' + rE.net_name + '_' + str(i + 1))
            rT.free_log()
        rT.remove_hooks(forward_hooks)
        rT.remove_hooks(backward_hooks)
    if rE.net_name == "VGG19":
        trainset, testloader = Net.VGG19.load_data(batch_size=rE.batch_size)

        net = init_Net(rE.net_name)
        net = net.to(torch.float32).to(torch.float64)
        net = net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001)
        
        rT = roundTool(rE.round_amount, rE.round_mode)

        forward_hooks = rT.add_forward_hooks(net)
        backward_hooks = rT.add_backward_hooks(net)

        print("[Untrusted]: Training...")
        
        if rE.save_parameter:
            save_net(rE.net_name, rE.ckpt_base_path, optimizer, 0, net)
            save_rng_state(rE.rng_state + '/' + rE.net_name + '_0')
        for i in tqdm(range(rE.max_iter)):
            backward_hooks_ = []
            
            inputs = torch.zeros((rE.batch_size, 3, 224, 224))
            labels = torch.LongTensor(rE.batch_size)
            for j in range(rE.batch_size):
                index = prf(seed * i + j, rE.net_name) % len(trainset)  # 确保index在有效范围内
                labels[j] = trainset[index][1]
                inputs[j] = trainset[index][0]
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.to(torch.float32).to(torch.float64)
            # zero the parameter gradients

            torch.cuda.synchronize()
            optimizer.zero_grad()
            torch.cuda.synchronize()

            outputs = net(inputs)

            backward_hooks_.append(outputs.register_hook(partial(rT.rounding_backward_hook_64, "output")))

            loss = criterion(outputs, labels)

            backward_hooks_.append(loss.register_hook(partial(rT.rounding_backward_hook_64, "loss")))

            append_to_loss(loss.item(), "loss" + str(rE.round_amount) + ".txt")

            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()

            with torch.no_grad():
                for n, p in net.named_parameters():
                    p.data = rT.round_to_low_bits(p.data)
            if rE.save_parameter:
                save_net(rE.net_name, rE.ckpt_base_path, optimizer, i + 1, net)
                rT.save_round_log(times=i, path=rE.round_path, name=rE.net_name)
                save_rng_state(rE.rng_state + '/' + rE.net_name + '_' + str(i + 1))
            for h in backward_hooks_:
                h.remove()

            rT.free_log()
        rT.remove_hooks(forward_hooks)
        rT.remove_hooks(backward_hooks)

    if rE.net_name == "ResNet":
        trainset, testloader = Net.resnet50.load_data(rE.batch_size)

        net = init_Net(rE.net_name)
        net = net.to(torch.float32).to(torch.float64)
        net = net.to(device)

        optimizer = optim.Adam(net.parameters(), lr=0.00003)
        criterion = nn.CrossEntropyLoss()
        net.train()
        rT = roundTool(rE.round_amount, rE.round_mode)

        forward_hooks = rT.add_forward_hooks(net)
        backward_hooks = rT.add_backward_hooks(net)

        print("[Untrusted]: Training...")

        if rE.save_parameter:
            # save_net_VGG16(net, os.path.join(rE.ckpt_base_path, str(0)), bin_style=False)
            save_net(rE.net_name, rE.ckpt_base_path, optimizer, 0, net)
            save_rng_state(rE.rng_state + '/' + rE.net_name + '_0')
        for i in tqdm(range(rE.max_iter)):
            backward_hooks_ = []
            
            inputs = torch.zeros((rE.batch_size, 3, 32, 32))
            labels = torch.LongTensor(rE.batch_size)
            for j in range(rE.batch_size):
                index = prf(seed * i + j, rE.net_name) % len(trainset)  # 确保index在有效范围内
                labels[j] = trainset[index][1]
                inputs[j] = trainset[index][0]
            
            # 转移数据到设备
            inputs = inputs.to(device).to(torch.float32).to(torch.float64)
            labels = labels.to(device)
            # zero the parameter gradients
            torch.cuda.synchronize()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            outputs = net(inputs)

            backward_hooks_.append(outputs.register_hook(partial(rT.rounding_backward_hook_64, "output")))

            loss = criterion(outputs, labels)

            backward_hooks_.append(loss.register_hook(partial(rT.rounding_backward_hook_64, "loss")))

            append_to_loss(loss.item(), "loss" + str(rE.round_amount) + ".txt")

            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()

            with torch.no_grad():
                for n, p in net.named_parameters():
                    p.data = rT.round_to_low_bits(p.data)
            if rE.save_parameter:
                save_net(rE.net_name, rE.ckpt_base_path, optimizer, i + 1, net)
                rT.save_round_log(times=i, path=rE.round_path, name=rE.net_name)
                save_rng_state(rE.rng_state + '/' + rE.net_name + '_' + str(i + 1))
            for h in backward_hooks_:
                h.remove()
            rT.free_log()
        rT.remove_hooks(forward_hooks)
        rT.remove_hooks(backward_hooks)