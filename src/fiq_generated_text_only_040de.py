"""Main generated caption only, with a frozen or fully trainable CLIP text prefix."""

from torch import nn
from torch.nn import functional as F

from fiq_conflict_context_040 import ordinary_text
from fiq_pixel_free_038 import gallery_kl


class GeneratedCaptionText(nn.Module):
    def __init__(self, base, full_text):
        super().__init__()
        self.clip = base.clip
        self.tokenizer = base.tokenizer
        self.image_fusion = base.image_fusion
        self.full_text = full_text
        self.clip.requires_grad_(False)
        if full_text:
            for part in (self.clip.token_embedding, self.clip.positional_embedding, self.clip.transformer,
                         self.clip.ln_final, self.clip.text_projection):
                part.requires_grad_(True)
        else:
            for part in self.clip.transformer.resblocks[-2:]:
                part.requires_grad_(True)
        self.clip.eval()

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        self.clip.transformer.grad_checkpointing = mode and self.full_text
        return self

    def parameter_groups(self):
        text = [p for n, p in self.clip.named_parameters() if p.requires_grad and not n.startswith('visual.')]
        return [dict(params=text, lr=1e-5, weight_decay=.01, foreach=False),
                dict(params=list(self.image_fusion.parameters()), lr=1e-4, weight_decay=.01, foreach=False)]

    def delta_state(self):
        return dict(blocks={n: p.detach().cpu().clone() for n, p in self.clip.named_parameters() if p.requires_grad},
                    image_fusion={k: p.detach().cpu().clone() for k, p in self.image_fusion.state_dict().items()})

    def gradient_audit(self):
        blocks = self.clip.transformer.resblocks
        probes = {'last_block': blocks[-1].attn.in_proj_weight}
        if self.full_text:
            probes.update(first_block=blocks[0].attn.in_proj_weight, vocabulary=self.clip.token_embedding.weight,
                          positions=self.clip.positional_embedding, projection=self.clip.text_projection)
        result = {name: p.grad is not None and bool(p.grad.count_nonzero()) for name, p in probes.items()}
        if not all(result.values()):
            raise ValueError(f'Caption text gradient did not reach registered parameters: {result}')
        return result


def compose_caption(model, visual, rows, reference, drop, drop_active):
    text = ordinary_text(model, [row['main_context_text'] for row in rows])
    clean, _ = visual(reference, text, drop, drop_active)
    q, alpha = model.image_fusion(text, clean)
    return dict(q=q, alpha=alpha)


def caption_objective(out, gallery, labels, mask):
    query = out['q']
    logits = (query @ gallery.T).masked_fill(~mask, -float('inf')) / .07
    retrieval = F.cross_entropy(logits, labels)
    kl = gallery_kl(query, gallery, labels, mask)
    return dict(total=retrieval + .5 * kl, retrieval=retrieval, gallery_kl=kl)
