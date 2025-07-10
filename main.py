from initDevice import *
from model_train import *
from model_verity import *

if __name__ == "__main__":
    initEnvironment()
    rE = roundEnvironment()
    if rE.round_mode and not rE.verify:
        init_write_log(rE.round_path)
        init_checkpoint(rE.ckpt_base_path)
        init_rng_state_log(folder_path=rE.rng_state, round_mode=rE.round_mode)

    device, seed = init_random_seed_and_device(rE.save_random_seeds_flag, rE.seed_path, rE.random_seeds_json, rE.verify)
    if rE.verify:
        verify(device, seed, rE)
    else:
        train(device, seed, rE)
