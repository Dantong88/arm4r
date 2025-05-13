import os



if __name__ == '__main__':
    # for i in range(5, 125, 10):
    #     policy_ckpt = '/scratch/partial_datasets/niudt/project/lff/ckpts/finetune_icrt_meat_off_grill200_vitb_freeze/checkpoint-{}.pth'.format(i)
    #     debug_dir = '/home/niudt/tmp/icrt/freeze_200_vitb/checkpoint-{}'.format(i)
    #     print('python eval.py method.debug={} method.policy_ckpt={}'.format(debug_dir, policy_ckpt))

    for i in [50, 90]:
        policy_ckpt = '/home/niudt/project/arm4r_release/arm4r/output/finetune_rlbench_meat_off_grill/checkpoint-{}.pth'.format(i)
        debug_dir = '/home/niudt/tmp/arm4r_sim/standard_eval_finetune_rlbench_meat_off_grill-checkpoint-{}'.format(i)
        print('python eval.py method.debug={} method.policy_ckpt={}'.format(debug_dir, policy_ckpt))