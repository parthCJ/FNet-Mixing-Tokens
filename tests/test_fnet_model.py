import torch

from fnet_mixing.model import FNetConfig, FNetForSequenceClassification


def test_fnet_forward_shape() -> None:
    config = FNetConfig(
        vocab_size=1000,
        max_position_embeddings=32,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_labels=4,
    )
    model = FNetForSequenceClassification(config)

    input_ids = torch.randint(0, 999, (8, 32))
    attention_mask = torch.ones(8, 32, dtype=torch.long)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    assert logits.shape == (8, 4)


def test_fnet_backward_pass() -> None:
    config = FNetConfig(
        vocab_size=500,
        max_position_embeddings=16,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_labels=3,
    )
    model = FNetForSequenceClassification(config)

    input_ids = torch.randint(0, 499, (4, 16))
    attention_mask = torch.ones(4, 16, dtype=torch.long)
    labels = torch.randint(0, 3, (4,))

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    has_grad = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert has_grad
