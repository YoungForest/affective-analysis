# python main.py --input ./input --video_root /home/data_common/data_yangsen/data --output ./output-resnet-34-kinetics.json --model /home/data_common/data_yangsen/PretrainedModels/resnet-34-kinetics.pth --mode feature
echo '----------------------------break line------------------------------------'
python main.py --input ./input --video_root /home/data_common/data_yangsen/data --output ./output-resnet-101-kinetics.json --model /home/data_common/data_yangsen/PretrainedModels/resnet-101-kinetics.pth --mode feature --model_depth 101
# echo '----------------------------break line------------------------------------'
# python main.py --input ./input --video_root /home/data_common/data_yangsen/data --output ./output-wideresnet-50-kinetics.json --model /home/data_common/data_yangsen/PretrainedModels/wideresnet-50-kinetics.pth --mode feature --model_depth 50
echo '----------------------------break line------------------------------------'
python main.py --input ./input --video_root /home/data_common/data_yangsen/data --output ./output-resnext-101-kinetics.json --model /home/data_common/data_yangsen/PretrainedModels/resnext-101-kinetics.pth --mode feature --model_depth 101
echo '----------------------------break line------------------------------------'
python main.py --input ./input --video_root /home/data_common/data_yangsen/data --output ./output-resnet-18-kinetics.json --model /home/data_common/data_yangsen/PretrainedModels/resnet-18-kinetics.pth --mode feature --model_depth 18
echo '----------------------------break line------------------------------------'
python main.py --input ./input --video_root /home/data_common/data_yangsen/data --output ./output-resnet-34-kinetics-cpu.json --model /home/data_common/data_yangsen/PretrainedModels/resnet-34-kinetics-cpu.pth --mode feature --model_depth 34 --no_cuda
echo '----------------------------break line------------------------------------'
python main.py --input ./input --video_root /home/data_common/data_yangsen/data --output ./output-resnet-50-kinetics.json --model /home/data_common/data_yangsen/PretrainedModels/resnet-50-kinetics.pth --mode feature --model_depth 50
echo '----------------------------break line------------------------------------'
