# Toxic comments classifier

**Problem**: Varying-length textual comments, containing typos, slang, rare words, etc.<br>
**Outcome**: Multi-label classification of 6 types of "toxicity" of comments. Labels slightly correlated, very sparse and imbalanced.
<br>
<br>
This notebook shows how to apply RNN for this classification task, using Bidirectional LSTM and hybrid lstm+conv1D architectures. Several losses are explored (BCE and focal loss). Glove pre-trained embeddings are used in some of the models.
<br>
<br>
**Model eval on val data (Macro F1)**:<br>
- M1: LSTM + fixed class weights (embedding64 + 2 bidirectional layers): **0.43**
- M2: M1 + undersampling 0s + oversampling small cats + embedding128: **0.47**
- M3: LSTM + focal loss (embedding256, lower dropouts + 2 bidirectional layers): **0.35**
- M4: M1 + pre-trained Glove embeddings (200d), back to BCE loss: **0.47**
- M5: Hybrid LSTM+Conv + Glove embeddings (200d + Conv1D*128 + 2 bidirectional: ****

**TO-DOs**:<br>
Plenty of things to improve:<br>
- Larger pre-trained embeddings (currently used 200d only)
- Larger variety of architectures+pre-trained embeddings ensembling
- Did not adequately address class imbalance - class weights/sampling don't work, but different loss functions might
- Train on the entire dataset
- Translation augmentation 
