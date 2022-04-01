# ML_training_evaluation
This is a code framework of a Machine Learning pipeline. It includes the basic data loading, training, validation, testing, logging, checkpoint saving, etc. This pipleline is adpated from my supervisor when I was working as a student assistant. The original framework is from: https://github.com/VisualComputingInstitute/DR-SPAAM-Detector

The purpose of this framework is to build a general code template for machine learning so that users can focus more on the real ML part such as model design, data processing, train/evaluation process, etc, rather than the pipeline itself.

The framework contains a hello-world example that implements a simple classifier trained on cifar10 that runs through this pipeline. Files and scripts with prefix "hw" are files for this example. They are used as reference or template for customization of the pipeline. To start this example, simply run:

    python hw_train.py

Brief explanation of the directories:
1. cfgs: for config files that includes all the necessary ML parameters
2. data: for datasets (create yourself usually)
3. exps: for desigining experiments
4. logs: for saving experiment results, file backup, and experiment logs
5. scripts: for batch jobs (e.g. slurm)
6. src: source code of the pipeline
7. tests: not particularly intended. Just for doing some tests

To customize this pipeline, users are suggested to create the following files (but not limited to):

    1. ./src/model:
        - [YOURMODEL].py: code for model
        - [YOURMODEL]_fn.py: code for model functions (loss computation, output processing)

    2. ./dataset/
        - a script for dataloader. e.g., hw_create_dataloader.py
        - a script for dataset. e.g., hw_dataset.py (pytorch has cifar10 so this script is empty here. but usually it is necessary)

    3. YOUR_OWN_TRAIN_SCRIPT.py

    4. YOUR_OWN_EVAL_SCRIPT.py

To customize this pipeline, users are suggested to put all relevant parameters into a config file. An example is ./cfg/hw_cfg.yaml. Further, the user should create a directory named 'data' and put the dataset in it.

This is a general reminder/suggestion of how to use and customize the pipeline. Please note that machine learning code can be very different depending on specific tasks. In many situations the code of the pipeline will have to be adpated specifically. However, please try making as less change as possible for the general structure and the backbone.


