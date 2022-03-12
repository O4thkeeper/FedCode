import torch
import torch.nn.functional as F

from torch import nn
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaPreTrainedModel, RobertaModel


class RobertaForSequenceClassification(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.p_head = HyperClassifier(config.hidden_size, 2)
        # self.p_head = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        return sequence_output

    # def forward_local_bias(self, feat, label_weight):
    #     cls_feat = feat[:, 0, :]
    #     clf_w = self.h_linear(label_weight)
    #     x = torch.matmul(cls_feat, clf_w)
    #     return x

    def forward_local_bias(self, feat):
        return self.p_head(feat)

    def forward_global(self, feat):
        return self.classifier(feat)


class BSMLoss(nn.Module):
    """
    Balanced Softmax Loss
    """

    def __init__(self, cls_num_list):
        super(BSMLoss, self).__init__()
        self.sample_per_class = cls_num_list

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()

    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class HyperClassifier(nn.Module):
    # f_size: config.hidden_size  z_dim:2
    def __init__(self, f_size, label_count):
        super(HyperClassifier, self).__init__()
        self.f_size = f_size
        self.label_count = label_count
        self.fc1 = nn.Linear(f_size, f_size)
        self.fc2 = nn.Linear(f_size, f_size * label_count)
        # self.fc1 = nn.Linear(self.label_count, self.f_size)
        # self.fc2 = nn.Linear(self.f_size, self.label_count * self.f_size)

    def forward(self, x):
        # h_in = F.relu(self.fc1(feat))
        # h_final = self.fc2(h_in)
        # h_final = h_final[0, :].view(-1, self.label_count)

        x = x[:, 0, :]
        h_in = F.relu(self.fc1(x))
        h_final = self.fc2(h_in)
        h_final = h_final.view(-1, self.label_count)
        h_final = torch.matmul(x, h_final)

        return h_final

