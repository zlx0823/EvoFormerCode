# EvoFormerCode
This repository provides a reference implementation of *EvoFormer* 

## Dependency Installation

To install all the required dependencies for this project, please use the following command:

```bash
pip install -r requirements.txt
```


## Create Random Walks

As described in our paper, each graph at a specific time step is transformed into a collection of sentences through random walks.

To generate these random walks, you can execute the `create_random_walks.py` script. The following parameters can be configured for random walk generation:
- `p` and `q`: Control the traversal strategy in the graph, as detailed in the node2vec algorithm.
- `walk_length` (L): Determines the maximum length of each sentence.
- `num_walks` (gamma): Specifies the number of walks initiated from each node.

```bash
python create_random_walks.py
```
## Running the Model

To run the EvoFormer model training, use the following command from the `evoformer` directory:

```bash
python train_model.py --dataset_name formula --sample_num 100000 --use_trsns True --use_poc True --epochs 5 --hidden_size 256 --walk_len 32
```

### Command Line Arguments

- `--dataset_name`: Name of the dataset to use (default: 'formula')
- `--sample_num`: Number of samples to use for training (default: 100000)
- `--use_trsns`: Whether to use Temporal Model (True or False, default: False)
- `--use_poc`: Whether to use POC (True or False, default: False)
- `--epochs`: Number of training epochs (default: 5)
- `--hidden_size`: Hidden size of the model (default: 256)
- `--walk_len`: Length of the random walk sequence (default: 32)
