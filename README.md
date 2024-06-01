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

### Mathematical reasoning
We have not yet completed the integration of HRA code into PEFT. Before that, if you want to try using the HRA method to fine-tune large models, you can follow the steps below.

Go to the llama folder
```bash
cd llama
```

#### Environment Setup
We recommend using Python 3.10 for your environment and use the conda to install it.
```bash
conda create -n pytorch python=3.10
```
Then install the required packages with the following command:
```bash
pip install -r requirements.txt
```
Please note that the peft package and transformer package must be downloaded with the versions consistent with those listed in the requirements file. 

After completing the download, please replace the **oft** folder inside the **peft/tuners** within your running environment's **python/site-packages** with the **oft** folder from the current directory.

The path for the oft folder in the environment should be:

```bash
/your_path/anaconda3/envs/pytorch/lib/python3.10/site-packages/peft/tuners/
```
The **layer.py** in the current oft directory is implemented for when λ is not infinity.

If you want to simulate when λ is infinity, please replace **layer.py** with **layer_GS_HRA.py**, and set the hyperparameter λ to 0 during training.


#### Prepare Dataset
The dataset we use for fine-tuning is MetaMathQA-40K, which can be downloaded through this [link](https://huggingface.co/datasets/meta-math/MetaMathQA-40K).
#### Prepare model
The model we use for fine-tuning is llama2. You can choose the model you want to fine-tune.
#### Finetune
Run the following code to complete the fine-tuning:
```bash
bash tune.sh
```
Please note that you need to change the dataset path, the path of the pre-trained model, and you can change the parameters according to your needs in tune.sh. That is:
```bash
BASE_MODEL="YOUR_MODEL_PATH"
DATA_PATH="YOUR_DATA_PATH"
OUTPUT="YOUR_MODEL_SAVED_PATH"
```
#### Evaluation
After the training is complete, you can run the following command to test:
```bash
bash test.sh
```
Please note to change the model path in it:
```bash
BASE_MODEL="YOUR_MODEL_PATH"
OUTPUT="YOUR_MODEL_SAVED_PATH"
```



## Citing our work
```bibtex
@misc{yuan2024bridging,
      title={Bridging The Gap between Low-rank and Orthogonal Adaptation via Householder Reflection Adaptation}, 
      author={Shen Yuan and Haotian Liu and Hongteng Xu},
      year={2024},
      eprint={2405.17484},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


