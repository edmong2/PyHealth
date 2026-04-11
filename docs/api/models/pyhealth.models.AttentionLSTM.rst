pyhealth.models.AttentionLSTM
===================================

.. automodule:: pyhealth.models.attention_lstm
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

AttentionLSTM is an extension of the standard RNN model that replaces
the final hidden state representation with an attention-weighted
aggregation over all time steps.

Instead of relying solely on the last hidden state of the sequence,
this model learns to assign importance weights to each timestep,
allowing it to focus on the most relevant parts of the sequence.

Key Features
------------

- Uses LSTM layers for sequence modeling
- Applies attention over time steps
- Supports masking for variable-length sequences
- Computes attention weights per feature
- Returns attention weights for interpretability

Architecture
------------

For each input feature:

1. The input sequence is embedded using ``EmbeddingModel``
2. The embedded sequence is passed through an LSTM layer
3. Attention scores are computed for each timestep
4. A softmax is applied to obtain attention weights
5. A weighted sum of LSTM outputs produces a context vector

The context vectors from all features are concatenated and passed
through a fully connected layer for prediction.

Outputs
-------

The model returns a dictionary containing:

- ``loss``: training loss
- ``y_prob``: predicted probabilities
- ``y_true``: ground truth labels
- ``logit``: raw output logits
- ``attention_weights``: attention weights for each feature

Use Case
--------

This model is particularly useful for:

- Clinical time-series modeling
- Interpretable sequence classification
- Tasks where identifying important timesteps is valuable

Notes
-----

Attention weights are returned for each feature and can be used for
analysis and interpretability, but may vary across different model
initializations.