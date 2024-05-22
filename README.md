## Installation

For nlu and generation, construct the virtual environment by Anaconda or Miniconda3:
```bash
conda env create -f env.yml
```

## Usage


### Subject-driven Generation
1. Similar to the example for [diffusers-dreambooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth), you can run the finetuning using **HRA** with the following command. 
Within the 'subject' folder, run the training script to run the result on the [dreambooth](https://github.com/google/dreambooth) dataset. [Dreambooth](https://github.com/google/dreambooth) dataset consists of 30 subjects, with 25 validation prompts each:
```bash
cd subject
./train_dreambooth_householder.sh
```

1. We also provide the evaluation scripts:
```bash
python true_eval_ablation.py
```

### Controllable Generation

#### Fine-tuning

1. Create the model with additional **HRA** parameters:
```bash
cd control
python tool_add_householder.py \
  --input_path=./models/v1-5-pruned.ckpt \
  --output_path=./models/householder_l_8.ckpt \
  --l=8 
```
2. Specify the control signal and dataset. Train the model specify the same hyperparameters as above:
```bash
python train.py \
  --l=8 \
  --control=canny
```

#### Generation
1. After finetuning with **HRA**, run inference to generate images based on control signal. Because the inference takes some time, to perform large scale evaluation, we split the dataset into different sub-datasets and run inference on multiple gpus:
```bash
python test_householder_parallel.py 
  --l=8 \
  --control=canny
```
1. To evaluate **HRA** results on the three tasks listed in the paper (landmark-to-image (L2I), canny-to-image (C2I) and segmentation map-to-image (S2I)), run the following scripts on the generated images.
```bash
python eval_landmark.py
```
```bash
python eval_canny.py
```
Note, for evaluating the segmentation map-to-image (S2I) task, please install the [Segformer](https://github.com/NVlabs/SegFormer) repository. Run the following testing command on both the original and generated images.
```bash
python tools/test.py local_configs/segformer/B4/segformer.b4.512x512.ade.160k.py ./weights/segformer.b4.512x512.ade.160k.pth
```

### Natural Language Understanding

```bash
cd nlu
experiments/glue/mnli.sh deberta-v3-base
```
