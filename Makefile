CUDA_HOME121=/comp_robot/shock/share/pkgs/cuda-12.1
export NUMEXPR_MAX_THREADS=32
export CUDA_HOME=/comp_robot/shock/share/pkgs/cuda-12.1
export TOKENIZERS_PARALLELISM=True
export HF_ENDPOINT=https://hf-mirror.com
export EDPOSE_COCO_PATH=/comp_robot/zhangyuhong1/code2/ED-Pose/data/coco_dir
export MASTER_ADDR=localhost
export MASTER_PORT=6030
dev = $(shell nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print NR-1,$$1}' | sort -k2 -n | head -1 | awk '{print $$1}' )
host = $(shell hostname)

#export CUDA_VISIBLE_DEVICES=${dev}
MAKEFLAGS2 = $(shell echo ${MAKEFLAGS} | sed 's/^w -- //' | sed 's/^-- //' | sed 's/^w$$//' )



dist_edpose:
	 python -m torch.distributed.launch --nproc_per_node=1  main.py \
			--output_dir "logs/coco_r50" \
			-c config/edpose.cfg.py \
			--options batch_size=1 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
			--dataset_file="coco" \
			--save_log \
			--save_results \
			--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/output3/

dist_edpose_8:
	python -m torch.distributed.launch --nproc_per_node=8  main.py \
			--output_dir "logs/coco_r50" \
			-c config/edpose.cfg.py \
			--options batch_size=1 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
			--dataset_file="coco" \
			--save_log \
			--save_results \
			--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/output3/

debug_edpose:
	python main.py \
		--output_dir "logs/coco_r50" \
		-c config/edpose.cfg.py \
		--options batch_size=1 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
		--dataset_file="coco" \
		--save_log \
		--save_results \
		--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/output3/
		--debug

debug_dino_backbone:
	python main.py \
		--output_dir "logs/coco_r50" \
		-c config/edpose.cfg.py \
		--options batch_size=2 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
		--dataset_file="coco" \
		--save_log \
		--save_results \
		--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/output3/ \
		--debug \
		--dinox_backbone \
		--use_resume False \
		--resume False



dist_dino_8:
	python -m torch.distributed.launch --nproc_per_node=8  main.py \
		--output_dir "logs/coco_r50" \
		-c config/edpose.cfg.py \
		--options batch_size=4 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
		--dataset_file="coco" \
		--save_log \
		--save_results \
		--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/dino_output/ \
		--dinox_backbone \
		--resume True

dist_dino_new_4:
	python -m torch.distributed.launch --nproc_per_node=4  main.py \
		--output_dir "logs/dino_101_test" \
		-c config/edpose.cfg.py \
		--options batch_size=4 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
		--dataset_file="coco" \
		--save_log \
		--save_results \
		--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/dino_101_test/ \
		--dinox_backbone \
		--use_resume False \
		--resume False

dist_dino_new_8:
	python -m torch.distributed.launch --nproc_per_node=8  main.py \
		--output_dir "logs/dino_101_test_8" \
		-c config/edpose.cfg.py \
		--options batch_size=4 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
		--dataset_file="coco" \
		--save_log \
		--save_results \
		--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/dino_101_test_8/ \
		--dinox_backbone \
		--use_resume False \
		--resume False

eval_dino:
	python -m torch.distributed.launch --nproc_per_node=1 main.py \
		--output_dir "logs/coco_r50" \
		-c config/edpose.cfg.py \
		--options batch_size=1 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
		--dataset_file="coco" \
 		--pretrain_model_path "/comp_robot/zhangyuhong1/code2/ED-Pose/dino_101_test_8/checkpoint_best_regular.pth" \
		--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/dino_101_test_8/ \
		--dinox_backbone \
		--eval \
		--debug


eval_dino_resnet50:
	python -m torch.distributed.launch --nproc_per_node=1 main.py \
		--output_dir "logs/coco_r50" \
		-c config/edpose.cfg.py \
		--options batch_size=1 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
		--dataset_file="coco" \
 		--pretrain_model_path "/comp_robot/zhangyuhong1/pretrained_model/EDpose/edpose_r50_coco.pth" \
		--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/dino_eval/ \
		--eval
eval_dino_swint:
	python -m torch.distributed.launch --nproc_per_node=1 main.py \
		--output_dir "logs/coco_r50" \
		-c config/edpose.cfg.py \
		--options batch_size=1 epochs=60 lr_drop=55 num_body_points=17 backbone='swin_L_384_22k' \
		--dataset_file="coco" \
 		--pretrain_model_path "/comp_robot/zhangyuhong1/pretrained_model/EDpose/edpose_swinl_coco.pth" \
		--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/dino_eval/ \
		--eval


eval_dino_debug:
	python -m torch.distributed.launch --nproc_per_node=1 main.py \
		--output_dir "logs/coco_r50" \
		-c config/edpose.cfg.py \
		--options batch_size=1 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
		--dataset_file="coco" \
 		--pretrain_model_path "/comp_robot/zhangyuhong1/code2/ED-Pose/dino_output/checkpoint_best_regular.pth" \
		--output_dir /comp_robot/zhangyuhong1/code2/ED-Pose/dino_eval/ \
		--dinox_backbone \
		--eval \
		--debug 



setup-env-edpose2:
	conda create -n edpose2 python==3.10
	conda activate edpose2
	CUDA_HOME=${CUDA_HOME121} pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

setup-env-edpose2-c:
	CUDA_HOME=${CUDA_HOME121} pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

setup-env-pose2:
	conda create -n pose2 python==3.10
	conda activate pose2
	CUDA_HOME=${CUDA_HOME121} pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	CUDA_HOME=${CUDA_HOME121} pip install -r requirements.txt
	CUDA_HOME=${CUDA_HOME121} pip install openmim
	CUDA_HOME=${CUDA_HOME121} pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html # this will take a long time due to compiling extension
	CUDA_HOME=${CUDA_HOME121} mim install mmdet # mmdet requires mmcv<2.2.0
	CUDA_HOME=${CUDA_HOME121} pip install albumentations

install_script:
	CUDA_HOME=/comp_robot/shock/share/pkgs/cuda-12.1 pip install -r requirements.txt
kill:
	ps aux|grep wandb|grep -v grep | awk '{print $$2}'|xargs kill -9


tb:
	tensorboard --logdir work_dirs --host 0.0.0.0



