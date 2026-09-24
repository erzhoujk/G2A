pytest = __import__("pytest")
torch = pytest.importorskip("torch")

from g2a.guard import guard_huber_loss, guard_refinement_loss


def test_huber_objective() -> None:
    prediction = torch.tensor([0.2, 0.8], requires_grad=True)
    target = torch.tensor([0.1, 0.9])
    loss = guard_huber_loss(prediction, target)
    assert loss.item() > 0
    loss.backward()
    assert prediction.grad is not None


def test_refinement_margin_direction() -> None:
    corrected = torch.tensor([0.2, 0.5])
    original = torch.tensor([0.5, 0.55])
    loss = guard_refinement_loss(corrected, original, margin=0.1)
    assert loss.item() == pytest.approx(0.025)
