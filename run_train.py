import os

import sys

BASE_DIR = "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT"

nnunet_raw = f"{BASE_DIR}/nnunet_raw"
nnunet_results = f"{BASE_DIR}/nnunet_results"
nnunet_preprocessed = f"{BASE_DIR}/nnunet_preprocessed"

os.environ['nnUNet_raw'] = nnunet_raw
os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
os.environ['nnUNet_results'] = nnunet_results
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ['TORCH_LOGS'] = "+dynamo"
# os.environ['TORCHDYNAMO_VERBOSE'] = "1"


# parser = ArgumentParser()
# parser.add_argument("--dataset_name", default="Dataset005_mri_fat")
# parser.add_argument("--tr", default="nnUNetTrainer")
# # parser.add_argument("--device", type=int, default=1)
# parser.add_argument("--model", default="2d")
#
# args = parser.parse_args()

# command = f"CUDA_VISIBLE_DEVICES={args.device} nnUNetv2_train {args.dataset_name} {args.model} 0 -tr {args.tr}"
# os.system(command)

dataset_name = "Dataset043_BraTS_MET"
model = "3d_fullres"
tr = "nnUNetTrainerDiceFocal"
pretrained_weights = None # "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_results/Dataset040_masked/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_final.pth"
if __name__ == '__main__':
    from nnunetv2.run.run_training import run_training_entry

    fold = "3"
    sys.argv.extend([dataset_name, model, fold, "-tr", tr, # "--val", # "-device", "cpu",
                     *(["-pretrained_weights", pretrained_weights] if pretrained_weights else [])
                     ])
    print(sys.argv)
    run_training_entry()
