import torch
from functools import partial
import csv
import numpy as np
import os

class roundTool:
    def __init__(self, round_amount, round_mode):
        self.ROUND_LOG_AUTO = []
        self.ROUND_LOG_FORCE = []
        self.SIGN_FP64 = -9223372036854775808
        self.EXPONENT_FP64 = 0x7FF0000000000000
        self.ROUND_MASK = generate_num(round_amount)
        self.ROUND_MASK_LOW = generate_alternate_num(round_amount + 1)
        self.ROUND_MASK_FINAL = generate_num(round_amount)
        self.ROUND_MASK_LOW_FINAL = generate_alternate_num(round_amount + 1)
        # self.ROUND_MASK_FINAL = generate_num(round_amount - 5)
        # self.ROUND_MASK_LOW_FINAL = generate_alternate_num(round_amount - 4)
        self.ROUND_QUARTER = self.ROUND_MASK_LOW >> 1
        self.IF_ROUND = round_mode
        self.IF_VERIFY = False
        self.VERIFY_ROUND_LOG_AUTO = None
        self.VERIFY_ROUND_LOG_FORCE = None

    def set_verify_state(self):
        self.IF_VERIFY = True

    def round_to_low_bits(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_as_bits = tensor.view(torch.int64)
        round_bits = tensor_as_bits & self.ROUND_MASK
        round_bit = tensor_as_bits & self.ROUND_MASK_LOW
        exponent_bits = tensor_as_bits & self.EXPONENT_FP64
        sign_bits = tensor_as_bits & self.SIGN_FP64
        rounded_bits64 = torch.where(
            round_bit > 0,
            round_bits + (self.ROUND_MASK_LOW << 1),
            round_bits
        )
        rounded_int64 = torch.where(
            exponent_bits < self.EXPONENT_FP64, 
            sign_bits + exponent_bits + rounded_bits64, 
            sign_bits + exponent_bits + round_bits
        )
        result = rounded_int64.view(torch.float64)

        #log
        if self.IF_ROUND:
            log_result_force, log_result_auto = self.log_to_tensor(result, tensor, rounded_int64)
            self.add_log(log_result_force, log_result_auto)
        else:
            log_result_force, log_result_auto = self.force_round_decision(tensor, rounded_int64)
            rounded_bits64_force = torch.where(
                log_result_force > 0,
                round_bits + (self.ROUND_MASK_LOW << 1),
                round_bits
            )
            rounded_int64_force = torch.where(
                exponent_bits < self.EXPONENT_FP64, 
                sign_bits + exponent_bits + rounded_bits64_force, 
                sign_bits + exponent_bits + round_bits
            )
            rounded_float64_force = rounded_int64_force.view(torch.float64)
            result = torch.where(
                log_result_auto > 0, 
                result, 
                rounded_float64_force
            )

        return result
    
    # def round_to_low_bits(self, tensor: torch.Tensor) -> torch.Tensor:
    #     tensor_as_bits = tensor.view(torch.int64)
    #     round_bits = tensor_as_bits & self.ROUND_MASK
    #     round_bit = tensor_as_bits & self.ROUND_MASK_LOW
    #     exponent_bits = tensor_as_bits & self.EXPONENT_FP64
    #     sign_bits = tensor_as_bits & self.SIGN_FP64
    #     rounded_bits64 = torch.where(
    #         ((round_bit > 0) & (sign_bits == 0)) | ((round_bit == 0) & (sign_bits != 0)),
    #         round_bits + (self.ROUND_MASK_LOW << 1),
    #         round_bits
    #     )
    #     rounded_int64 = sign_bits + exponent_bits + rounded_bits64
    #     result = rounded_int64.view(torch.float64)

    #     #log
    #     if self.IF_ROUND:
    #         log_result_force, log_result_auto = self.log_to_tensor(result, tensor, rounded_int64)
    #         self.add_log(log_result_force, log_result_auto)
    #     else:
    #         log_result_force, log_result_auto = self.force_round_decision(tensor, rounded_int64)
    #         rounded_bits64_force = torch.where(
    #             ((log_result_force > 0) & (sign_bits == 0)) | ((log_result_force == 0) & (sign_bits != 0)),
    #             round_bits + (self.ROUND_MASK_LOW << 1),
    #             round_bits
    #         )
    #         rounded_int64_force = sign_bits + exponent_bits + rounded_bits64_force

    #         rounded_float64_force = rounded_int64_force.view(torch.float64)
    #         result = torch.where(
    #             log_result_auto > 0, 
    #             result, 
    #             rounded_float64_force
    #         )
    #     result = torch.where(
    #         tensor == 0,
    #         0,
    #         result
    #     )
    #     return result
    
    def round_to_low_bits_final(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_as_bits = tensor.view(torch.int64)
        round_bits = tensor_as_bits & self.ROUND_MASK_FINAL
        round_bit = tensor_as_bits & self.ROUND_MASK_LOW_FINAL
        exponent_bits = tensor_as_bits & self.EXPONENT_FP64
        sign_bits = tensor_as_bits & self.SIGN_FP64
        rounded_bits64 = torch.where(
            round_bit > 0,
            round_bits + (self.ROUND_MASK_LOW_FINAL << 1),
            round_bits
        )
        # for exp_i in exponent_bits:
        #     if exp_i == self.EXPONENT_FP64: 
        #         print('Error!')
        rounded_int64 = sign_bits + exponent_bits + rounded_bits64
        result = rounded_int64.view(torch.float64)

        #log
        if self.IF_ROUND:
            log_result_force, log_result_auto = self.log_to_tensor(result, tensor, rounded_int64)
            self.add_log(log_result_force, log_result_auto)
        else:
            log_result_force, log_result_auto = self.force_round_decision(tensor, rounded_int64)
            rounded_bits64_force = torch.where(
                log_result_force > 0,
                round_bits + (self.ROUND_MASK_LOW_FINAL << 1),
                round_bits
            )
            rounded_int64_force = sign_bits + exponent_bits + rounded_bits64_force

            rounded_float64_force = rounded_int64_force.view(torch.float64)
            result = torch.where(
                log_result_auto > 0, 
                result, 
                rounded_float64_force
            )
        result = torch.where(
            tensor == 0,
            0,
            result
        )
        result = torch.where(
            tensor == -torch.finfo(torch.float64).max,
            -torch.finfo(torch.float64).max,
            result
        )
        result = torch.where(
            tensor == torch.finfo(torch.float64).max,
            torch.finfo(torch.float64).max,
            result
        )
        return result

    def force_round_decision(self, origin_tensor, rounded_tensor_int64):
        round_low = (rounded_tensor_int64 - self.ROUND_QUARTER).view(torch.float64)
        round_high = (rounded_tensor_int64  + self.ROUND_QUARTER).view(torch.float64)

        abs_origin_tensor = torch.abs(origin_tensor)

        if_auto = torch.where(
            (
                (torch.abs(round_low) < torch.abs(round_high))
                & (abs_origin_tensor >= torch.abs(round_low))
                & (abs_origin_tensor < torch.abs(round_high))
            ) | (
                (torch.abs(round_low) > torch.abs(round_high))
                & (abs_origin_tensor < torch.abs(round_low))
                & (abs_origin_tensor >= torch.abs(round_high))
            ),
            torch.tensor(1, dtype=torch.uint8),
            torch.tensor(0, dtype=torch.uint8)
        )

        if_auto = if_auto | self.int64_array_to_tensor(self.VERIFY_ROUND_LOG_AUTO, if_auto.shape)
        
        force_decision = self.int64_array_to_tensor(self.VERIFY_ROUND_LOG_FORCE, if_auto.shape)

        if len(self.VERIFY_ROUND_LOG_AUTO)< 1 and len(self.VERIFY_ROUND_LOG_FORCE) < 1:
            self.VERIFY_ROUND_LOG_AUTO = None
            self.VERIFY_ROUND_LOG_FORCE = None

        return force_decision, if_auto

    def add_log(self, log_result_f, log_result_a):
        # self.ROUND_LOG_FORCE.append(log_result_f)
        # self.ROUND_LOG_AUTO.append(log_result_a)
        self.ROUND_LOG_FORCE.append(log_result_f)
        self.ROUND_LOG_AUTO.append(log_result_a)

    def free_log(self):
        self.ROUND_LOG_FORCE = []
        self.ROUND_LOG_AUTO = []

    def log_to_tensor(self, result_tensor, origin_tensor, rounded_tensor_int64):
        round_low = (rounded_tensor_int64 - self.ROUND_QUARTER).view(torch.float64)
        round_high = (rounded_tensor_int64  + self.ROUND_QUARTER).view(torch.float64)

        abs_result_tensor = torch.abs(result_tensor)
        abs_origin_tensor = torch.abs(origin_tensor)

        log_tensor_force = torch.where(
            abs_result_tensor > abs_origin_tensor,
            torch.tensor(1, dtype=torch.uint8),
            torch.tensor(0, dtype=torch.uint8)
        )

        log_tensor_auto = torch.where(
            (
                (torch.abs(round_low) < torch.abs(round_high))
                & (abs_origin_tensor >= torch.abs(round_low))
                & (abs_origin_tensor < torch.abs(round_high))
            ) | (
                (torch.abs(round_low) > torch.abs(round_high))
                & (abs_origin_tensor < torch.abs(round_low))
                & (abs_origin_tensor >= torch.abs(round_high))
            ),
            torch.tensor(1, dtype=torch.uint8),
            torch.tensor(0, dtype=torch.uint8)
        )
        # return log_tensor_force, log_tensor_auto
        return log_tensor_force.flatten(), log_tensor_auto.flatten()

    def int64_array_to_tensor(self, row_data, original_size):
        
        length = int(torch.tensor(original_size).prod().item())
        if length > len(row_data):
            raise ValueError(f"请求取出 {length} 个比特，但当前剩余比特仅 {len(row_data)} 个。")

        # 取出前 length 个比特
        chunk = row_data[:length]
        # 从原 list 中删除这部分
        del row_data[:length]

        # 转回 PyTorch 张量
        original_tensor = torch.tensor(chunk, dtype=torch.uint8)

        if len(original_size) == 0:
            tensor = original_tensor.reshape(())   # 将只有一个元素的张量转换为标量
        else:
            tensor = original_tensor.view(*original_size)

        return tensor

    def add_forward_hooks(self, model):
        hooks = []
        for name, module in model.named_modules():
            if(sum(1 for _ in module.children()) == 0):
                hooks.append(module.register_forward_hook(partial(self.rounding_forward_hook, "forward "+name)))
                # if any(p.requires_grad for p in module.parameters()):
                #     hooks.append(module.register_forward_hook(partial(self.rounding_forward_hook, "forward "+name)))
        return hooks
    
    def add_backward_hooks(self, model):
        hooks = []
        for name, module in model.named_modules():
            if(sum(1 for _ in module.children()) == 0):
                hooks.append(module.register_full_backward_hook(partial(self.full_rounding_backward_hook, "backward "+name)))
                # if any(p.requires_grad for p in module.parameters()):
                #     hooks.append(module.register_full_backward_hook(partial(self.full_rounding_backward_hook, "backward "+name)))
        return hooks
    
    def rounding_forward_hook(self, name, module, input, result_old):
        result = result_old
        result.data = self.round_to_low_bits(result.data)
        return result
    
    def full_rounding_backward_hook(self, name, module, grad_input, grad_output):
        #判断是train还是test
        modified_grads = []
        for grad in grad_input:
            if grad is None:
                modified_grads.append(grad)
                continue
            modified_grad = grad
            modified_grad.data = self.round_to_low_bits(grad.data)
            modified_grads.append(modified_grad)
        return tuple(modified_grads)
    
    def remove_hooks(self, hooks):
        for h in hooks:
            h.remove()

    def rounding_backward_hook_64(self, name, grad):
        new_grad = grad
        new_grad.data = self.round_to_low_bits(grad.data)
        return new_grad

    def save_round_log(self, times, path='./round_log', name = None):
        num_i = str(times)
        # 将所有张量拼接为一个长的一维张量
        concatenated = torch.cat(self.ROUND_LOG_FORCE, dim=0).to(torch.uint8)
        # 转为 numpy 数组
        arr = concatenated.cpu().numpy()
        
        # 使用 packbits 将每 8 个 bit 打包成 1 字节
        packed = np.packbits(arr)
        
        # 写入二进制文件
        with open(path + '/' + num_i + '_' + name + '_force', "wb") as f:
            f.write(packed.tobytes())
                # 将所有张量拼接为一个长的一维张量
                
        concatenated = torch.cat(self.ROUND_LOG_AUTO, dim=0).to(torch.uint8)
        # 转为 numpy 数组
        arr = concatenated.cpu().numpy()
        
        # 使用 packbits 将每 8 个 bit 打包成 1 字节
        packed = np.packbits(arr)
        
        # 写入二进制文件
        with open(path + '/' + num_i + '_' + name + '_auto', "wb") as f:
            f.write(packed.tobytes())
        self.free_log()

    def round_csv_to_arrays(self, times, path='./round_log', name = 'VGG16'):
        num_i = str(times)
        with open(path + '/' + num_i + '_' + name + '_force', "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(arr)  # 得到一维的 0/1 数组
        self.VERIFY_ROUND_LOG_FORCE = bits.tolist()

        with open(path + '/' + num_i + '_' + name + '_auto', "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(arr)  # 得到一维的 0/1 数组
        self.VERIFY_ROUND_LOG_AUTO = bits.tolist()

def find_file_with_keywords_2(directory: str, keyword1: str, keyword2: str):
    """
    在指定目录中查找文件路径同时包含两个特定字段的文件，并返回第一个匹配的完整路径。
    
    :param directory: 要搜索的目录路径
    :param keyword1: 文件路径应包含的第一个关键字
    :param keyword2: 文件路径应包含的第二个关键字
    :return: 第一个匹配文件的完整路径，如果没有找到则返回 None
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} 不是一个有效的目录")
        return None
    
    for root, _, files in os.walk(directory):
        for file in files:
            if keyword1 in file and keyword2 in file:
                return os.path.join(root, file)
    
    return None

def generate_num(bits):
    if bits < 1 or bits > 64:
        raise ValueError("bits must be between 1 and 64, inclusive.")

    high_bits = (1 << (bits - 9)) - 1
    low_bits = 52 - (bits - 9)
    num = (high_bits << low_bits) & ((1 << 52) - 1)
    return num
                    
def generate_alternate_num(bits):
    if bits < 1 or bits > 64:
        raise ValueError("bits must be between 1 and 64, inclusive.")

    num = 1 << (52 - (bits - 9))
    return num
