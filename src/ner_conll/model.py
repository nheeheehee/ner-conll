import torch
import torch.nn as nn
import transformers

import config


def loss_fn(output, target, mask, num_labels):
    criterion = nn.CrossEntropyLoss()
    active_mask = mask.view(-1) == 1  # change all the 0 -> False
    logits = output.view(-1, num_labels)
    labels = torch.where(
        active_mask,
        target.view(-1),
        torch.tensor(criterion.ignore_index).type_as(target),
    )

    loss = criterion(logits, labels)

    return loss


class NERModel(nn.Module):
    def __init__(self, num_pos, num_tags):
        super(NERModel, self).__init__()
        self.num_pos = num_pos
        self.num_tags = num_tags
        self.bert = transformers.BertModel.from_pretrained(
            config.BASE_MODEL, return_dict=False
        )
        self.dropout = nn.Dropout(0.3)
        self.linear_pos = nn.Linear(768, self.num_pos)
        self.linear_tag = nn.Linear(768, self.num_tags)

    def forward(self, ids, mask, token_type_ids, pos_tag, target_tag):

        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        bert_out_tag = self.dropout(o1)
        bert_out_pos = self.dropout(o1)

        bert_out_tag = self.linear_tag(bert_out_tag)
        bert_out_pos = self.linear_pos(bert_out_pos)

        loss_tag = loss_fn(bert_out_tag, target_tag, mask, self.num_tags)
        loss_pos = loss_fn(bert_out_pos, pos_tag, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return bert_out_pos, bert_out_tag, loss


if __name__ == "__main__":
    model = NERModel(8, 8)
    ids = torch.randint(1000, size=(32, config.MAX_LEN)).long()
    mask = torch.randint(0, 1, size=(32, config.MAX_LEN)).long()
    token_type_ids = torch.zeros(
        (
            32,
            config.MAX_LEN,
        )
    ).long()
    pos_tag = torch.randint(0, 9, size=(32, config.MAX_LEN)).long()
    target_tag = torch.randint(0, 9, size=(32, config.MAX_LEN)).long()

    input = {
        "ids": ids,
        "mask": mask,
        "token_type_ids": token_type_ids,
        "pos_tag": pos_tag,
        "target_tag": target_tag,
    }
    output = model(**input)
    print(output[0].size(), output[1].size())
