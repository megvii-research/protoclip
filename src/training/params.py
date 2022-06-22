import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args():
    parser = argparse.ArgumentParser()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Data and Episodic training
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--augmentation",
        choices=[None, "protoclip-light-augmentation"],
        default=None,
        help="Use lighter augmentation for implicit contrast. Choices: [None, protoclip-light-augmentation]",
    ) 
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        default=None,
        help="Path to datasets for evaluation",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="Trunck the number of samples in dataset.",
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--episode-size",
        type=int,
        default=0,
        help="Set episode_size to 0 to disable episodic training",
    )  

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Prototypical contrast
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    parser.add_argument(
        "--external-teacher",
        type=str,
        default=None,
        help="Saved numpy array with shape (dataset_size, feature_dim) as external teacher. leave it as None to disable the external teacher."
    )
    parser.add_argument(
        "--add-projection-head",
        action="store_true",
        default=False,
        help="add two projection heads and leanable temperatures to CLIP",
    ) 
    parser.add_argument(
        "--PBT",
        action="store_true",
        default=False,
        help="enable Prototype Back Translation",
    ) 
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=128,
        help="dimension of projected representations",
    ) 
    parser.add_argument(
        "--projection-hidden-dim",
        type=int,
        default=2048,
        help="dimension of projected representations",
    ) 
    parser.add_argument(
        "--projection-n-layers",
        type=int,
        default=1,
        help="dimension of projected representations",
    ) 
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=-1.0,
        help="target temperature to calculate teacher scroes in proto loss",
    ) 
    parser.add_argument(
        "--clustering-frequency",
        type=int,
        default=-1,
        help="update prototypes, set to -1 for non ProtoCLIP models",
    ) 
    parser.add_argument(
        "--k",
        type=int,
        default=20000,
        help="dimension of projected representations",
    ) 
    parser.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=20,
        help="maximum iterations of K-Means optimization",
    ) 
    parser.add_argument(
        "--kmeans-nredo",
        type=int,
        default=1,
        help="random re-initialize and do K-Means for how many times",
    ) 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Logging and checkpointing
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=True,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--pretrained-text",
        default=None,
        type=str,
        help="Load pretrained language model as text tower via pytorch-transformers.",
    )
    
    # MODELS = [(BertModel,       BertTokenizer,      'bert-base-uncased'),
    #           (OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
    #           (GPT2Model,       GPT2Tokenizer,      'gpt2'),
    #           (TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
    #           (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
    #           (XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024'),
    #           (RobertaModel,    RobertaTokenizer,   'roberta-base')]
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Loss functions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    parser.add_argument("--w-clip", type=float, default=1., help="Loss weight.")
    parser.add_argument("--w-proto", type=float, default=0., help="Loss weight.")
    parser.add_argument("--w-proto-external", type=float, default=0., help="Loss weight.")
    parser.add_argument(
        "--infonce-warmup-epoch",
        default=0,
        type=int,
        help="InfoNCE-only warmup.",
    )
    parser.add_argument(
        "--lit-start-epoch",
        default=-1,
        type=int,
        help="Enable ProtoCLIP asymetric learning rate scheduler. Leave it as negative to skip LiT.",
    )
    parser.add_argument(
        "--text-start-epoch",
        default=0,
        type=int,
        help="Freeze text encoder at the begining of training.",
    )
    parser.add_argument(
        "--text-end-epoch",
        default=-1,
        type=int,
        help="TODO",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Optimization
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--lr-text", type=float, default=-1., help="Seperate learning rate despite visual backbone. Leave it as -1 to use default unified learning rate")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--max-grad-norm",
        default=1e16,
        type=float,
        help="Enable gradient clipping.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Evaluation
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    parser.add_argument("--zeroshot-frequency", type=int, default=0, help="How often to run zero shot.")
    parser.add_argument("--retrieval-frequency", type=int, default=0, help="How often to run coco retrieval.")
    parser.add_argument("--linear-frequency", type=int, default=0, help="How often to run linear eval.")
    parser.add_argument("--visualize-frequency", type=int, default=-1, help="How often to run linear eval.")
    parser.add_argument("--C", type=float, default=3.16, help="inverse regularizer for logistic reg (sklearn implementation).")
    parser.add_argument(
        "--linear-prob-mode",
        choices=["pytorch", "sklearn"],
        default="pytorch",
        help="Use witch implementation for linear evaluaion"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Distributed training
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    args = parser.parse_args()

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
