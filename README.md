## Head in Head in Linear Attention(ICML 2026)

Based on Flash-linear-attention.

Using legacy/training to train:

### prepare data:download datasets and using pre_data.sh

### Training:

bash run_one_node.sh


Currently, the trained and tested code includes \flash-linear-attention\legacy\training\fla2\layers\mask_gdn.py and

\flash-linear-attention\legacy\training\fla3\layers\mask_rwkv7.py


At present, the implementation speed of rwkv7 is slow and still needs further optimization.

We provide a training acceleration version of GDN in \flash-linear-attention\legacy\training\fla4\layers\mask_gdn.py
