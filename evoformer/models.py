import torch.nn as nn
from our_transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from our_transformers import BertModel


def create_mask_matrix(N, blocks):
    """
    Create mask matrix.
    Parameters:
    N: Matrix size (NxN)
    blocks: List where each element is a tuple (start_row, size) representing the start row and size of each red block
    Returns:
    NxN mask matrix with red blocks represented as 1 and others as 0
    """
    # Create an all-zero array representing white cells
    mask_matrix = np.zeros((N, N), dtype=int)

    # Set red blocks
    for start_row, size in blocks:
        mask_matrix[start_row:start_row + size, start_row:start_row + size] = 1
        x = mask_matrix[start_row:start_row + size, start_row:start_row + size]
        mask_matrix[start_row:start_row + size, start_row:start_row + size] = np.tril(x)

    return mask_matrix


    

class Layer(nn.Module):
    """Base layer class for PyTorch. Similar to the TensorFlow-based Layer class."""

    def __init__(self, name=None, logging=False):
        super(Layer, self).__init__()
        self.name = name if name else self.__class__.__name__.lower()
        self.logging = logging
        self.vars = nn.ParameterDict()  # Store learnable parameters

    def forward(self, inputs):
        """Wrapper for forward pass (computation graph)"""
        if self.logging:
            print(f'Logging inputs for layer {self.name}: {inputs}')
        outputs = self._call(inputs)
        if self.logging:
            print(f'Logging outputs for layer {self.name}: {outputs}')
        return outputs

    def _call(self, inputs):
        """To be implemented in subclasses."""
        raise NotImplementedError

    def _log_vars(self):
        """Logging model variables."""
        for name, param in self.vars.items():
            print(f'Variable {self.name}/{name}: {param}')


class TemporalAttentionLayer(Layer):
    def __init__(self, input_dim, n_heads, num_time_steps, attn_drop, residual=False, bias=True,
                 use_position_embedding=True, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.attn_drop = attn_drop
        self.residual = residual
        self.layer_norm1 = nn.LayerNorm(self.input_dim)
        self.layer_norm2 = nn.LayerNorm(self.input_dim)
        params = {"in_channels": self.input_dim, "out_channels": self.input_dim, "kernel_size": 1}
        self.lastcov = nn.Conv1d(**params)
        # Define position embeddings if used
        if use_position_embedding:
            self.vars['position_embeddings'] = nn.Parameter(
                torch.Tensor(num_time_steps, input_dim))
            nn.init.xavier_uniform_(self.vars['position_embeddings'])

        # Define Q, K, V weights for attention
        self.vars['Q_embedding_weights'] = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.vars['K_embedding_weights'] = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.vars['V_embedding_weights'] = nn.Parameter(torch.Tensor(input_dim, input_dim))

        nn.init.xavier_uniform_(self.vars['Q_embedding_weights'])
        nn.init.xavier_uniform_(self.vars['K_embedding_weights'])
        nn.init.xavier_uniform_(self.vars['V_embedding_weights'])

    def forward(self, inputs, seg_list):
        """Forward pass for temporal attention layer.
           inputs: Tensor of shape [N, T, F]
        """
        # Add position embeddings to input
        batch_size, T, C = inputs.size()
        position_inputs = torch.arange(self.num_time_steps, device=inputs.device).unsqueeze(0).repeat(batch_size, 1)
        temporal_inputs = inputs + F.embedding(position_inputs, self.vars['position_embeddings'])  # [N, T, F]

        # Query, Key, Value computation
        q = torch.matmul(temporal_inputs, self.vars['Q_embedding_weights'])  # [N, T, F]
        k = torch.matmul(temporal_inputs, self.vars['K_embedding_weights'])  # [N, T, F]
        v = torch.matmul(temporal_inputs, self.vars['V_embedding_weights'])  # [N, T, F]

        # Split into multiple heads and concatenate
        q_ = torch.cat(q.split(C // self.n_heads, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(k.split(C // self.n_heads, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(v.split(C // self.n_heads, dim=2), dim=0)  # [hN, T, F/h]

        # Scaled dot-product attention
        outputs = torch.matmul(q_, k_.transpose(-1, -2))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
        N = len(seg_list)
        res = partition_intervals(seg_list)
        results = [(start, end - start + 1) for start, end in res]
        mask_matrix = create_mask_matrix(N, results)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mask = torch.from_numpy(mask_matrix).unsqueeze(0).repeat(q_.size(0), 1, 1).to(device)
        outputs = outputs.masked_fill(mask == 0, float('-inf'))
        outputs = F.softmax(outputs, dim=-1)  # Masked attention
        outputs = F.dropout(outputs, p=self.attn_drop, training=self.training)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]

        # Concatenate multiple heads
        split_outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)  # [N, T, F]
        split_outputs = self.feedforward(split_outputs)
        split_outputs = self.feedforward(split_outputs)
        # Optional: Feedforward and residual connection
        if self.residual:
            split_outputs += temporal_inputs
        split_outputs = self.layer_norm2(split_outputs)
        return split_outputs

    def feedforward(self, inputs):
        """Point-wise feedforward layer.
           inputs: A 3d tensor of shape [N, T, C]
        """
        outputs = F.relu(self.lastcov(inputs.transpose(1, 2))).transpose(1, 2)
        return outputs + inputs


class EvoFormer(BertPreTrainedModel):
    '''
        Train a model MLM for node masking and temporal classification.
        Use the temporal_weight to control the tradeoff between the two.

    '''

    def __init__(self, config, batch_size=32, walk_len=32, use_trsns=False, pred_label=None):
        super().__init__(config)
        self.relu = nn.ReLU()
        self.temporal_num_labels = config.temporal_num_labels
        self.vocab_size = config.vocab_size
        self.bert = BertModel(config, batch_size=batch_size, walk_len=walk_len)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.temporal_num_labels)
        self.pred = nn.Linear(2 * config.hidden_size, 2)
        self.mlm = BertOnlyMLMHead(config)
        self.temporal_weight = config.temporal_weight
        self.attentionlayer = TemporalAttentionLayer(input_dim=config.hidden_size, n_heads=4,
                                                     num_time_steps=self.temporal_num_labels, attn_drop=0.2)
        self.poc_fn = nn.Linear(16, config.hidden_size)
        self.init_weights()
        self.use_trsns = use_trsns
        self.pred_label = pred_label

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            temporal_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            poc_emd=None,
            use_poc=False
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            poc_emd=poc_emd,
            use_poc=use_poc
        )

        # mlm part
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

        # temporal classification part
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        temporal_loss = loss_fct(logits.view(-1, self.temporal_num_labels), temporal_labels.view(-1))

        # Calculate classification accuracy
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == temporal_labels).float()

        accuracy = correct.mean()
        pred_loss = None
        classification = None
        if self.use_trsns:
            classification = self.classifier.weight.detach().clone().to(device)
            x = self.classifier.weight.detach().clone().cpu().numpy()
            seg_list,_ = top_down(x, k=8)
            classification = classification.unsqueeze(0)
            classification = self.attentionlayer(classification, seg_list).squeeze(0)
            paired_embeddings = [torch.cat([classification[i], classification[i + 1]], dim=0) for i in
                                 range(classification.shape[0] - 1)]
            paired_embeddings = torch.stack(paired_embeddings)
            pred_logits = self.pred(paired_embeddings).to(device)
            pred_loss_fct = CrossEntropyLoss()
            self.pred_label = self.pred_label.to(device)
            # pred_logits = pred_logits.to(device)
            pred_loss = pred_loss_fct(pred_logits, self.pred_label)

        loss =    masked_lm_loss +  5  * temporal_loss

        if self.use_trsns:
            loss += 5 * pred_loss
        # loss =   masked_lm_loss +    temporal_loss

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (loss, masked_lm_loss, temporal_loss, pred_loss) + outputs

        return outputs, accuracy, classification