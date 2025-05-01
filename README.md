# GAGRL

This is the implementation of the paper "Optimizing Transit Network Expansion with ​Gated Attentive Graph Reinforcement Learning​."

### Environment
- Python >= 3.8
- PyTorch == 1.12.1
- pip install -r requirements.txt

### Training
- Set configurations in '/transit/cfg/{}.yaml'
- python /transit/train.py

### Testing
- Set 'root_dir' as the folder that saved the trained model
- python /transit/eval.py
