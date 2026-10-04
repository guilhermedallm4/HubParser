"""HubParser: multi-task UD parsers (UPOS + DEPREL + HEAD) on BERT-family encoders.

Two head types, as in the dissertation:
  * linear   - three per-token linear classifiers; HEAD is a classification over
               absolute word positions (0 = root, 1..N = 1-based word index).
  * biaffine - linear UPOS + biaffine arc/rel scorers (Dozat & Manning, 2017);
               HEAD is scored over the subtoken positions of the sequence.

Biaffine head targets (`config.head_target`):
  * word_index     - as in the dissertation: the gold column for head word k is
                     sequence position k (misaligned once earlier words split into
                     several subtokens);
  * first_subtoken - corrected: the gold column is the first subtoken of the head
                     word, and the root is [CLS] (position 0).

Label inventories live in the config (`upos_labels`, `deprel_labels`).
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    ModernBertConfig,
    ModernBertModel,
    ModernBertPreTrainedModel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Non-linear projection used before the biaffine layers."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.33):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ELU()
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.dropout(self.norm(self.activation(self.linear(x))))


class Biaffine(nn.Module):
    """score[b, o, i, j] = x[b,i,:] · W[o] · y[b,j,:]  (x = dependent, y = head)."""

    def __init__(self, in_features: int, out_features: int = 1, bias_x: bool = True, bias_y: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.zeros(out_features, in_features + int(bias_x), in_features + int(bias_y)))
        nn.init.normal_(self.weight, std=1.0 / in_features)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], dim=-1)
        return torch.einsum("bih,ohk,bjk->boij", x, self.weight, y)


def _init_biaffine_weights(model, module):
    if isinstance(module, Biaffine):
        nn.init.normal_(module.weight, std=1.0 / module.in_features)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=model.config.initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=model.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper shared by every variant
# ─────────────────────────────────────────────────────────────────────────────
class HubParserMixin:
    """Inference shared by every variant.

    decoding = "greedy" (per-word argmax, as in the dissertation), "eisner" (best
    projective tree) or "mst" (Chu-Liu/Edmonds, best non-projective tree). Tree
    decoders use log P(head | dependent) restricted to the words of the sentence and
    always return a single-rooted tree.
    """

    def _head_columns(self, first_pos):
        """Sequence columns that stand for heads 0..n (0 = root)."""
        if getattr(self.config, "hubparser_head", "linear") == "linear":
            return list(range(len(first_pos) + 1))            # class k = word k
        if getattr(self.config, "head_target", "word_index") == "first_subtoken":
            return [0] + first_pos                               # [CLS] = root
        return list(range(len(first_pos) + 1))                   # dissertation: position k = word k

    @torch.no_grad()
    def _sentence_outputs(self, enc_one):
        inputs = {k: v.to(self.device) for k, v in enc_one.items()}
        if getattr(self.config, "hubparser_head", "linear") == "linear":
            logits_deprel, logits_upos, logits_head = self(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"))
            return logits_upos[0], logits_head[0], logits_deprel[0], None
        seq = self._encode(inputs)
        seq = self.dropout(seq)
        arc = self.arc_biaffine(self.arc_dep_mlp(seq), self.arc_head_mlp(seq)).squeeze(1)[0]
        rel = self.rel_biaffine(self.rel_dep_mlp(seq), self.rel_head_mlp(seq))[0]    # [R, L, L]
        return self.upos_classifier(seq)[0], arc, None, rel

    @torch.no_grad()
    def parse(self, sentences: List[List[str]], tokenizer, decoding: str = "greedy", max_length: int = 512):
        """Parse pre-tokenized sentences. Returns one list per sentence of dicts
        {id, form, upos, head, deprel} (head = 0 is the root)."""
        assert decoding in ("greedy", "eisner", "mst"), decoding
        self.eval()
        biaffine = getattr(self.config, "hubparser_head", "linear") == "biaffine"
        results = []
        for words in sentences:
            enc = tokenizer([words], is_split_into_words=True, truncation=True, max_length=max_length,
                            return_tensors="pt")
            word_ids = enc.word_ids(0)
            first = {}
            for pos, w in enumerate(word_ids):
                if w is not None and w not in first:
                    first[w] = pos
            n = len(first)
            first_pos = [first[i] for i in range(n)]
            cols = self._head_columns(first_pos)
            col2head = {c: h for h, c in enumerate(cols)}
            pos2word = {p: (w + 1 if w is not None else 0) for p, w in enumerate(word_ids)}

            upos_l, head_l, deprel_l, rel = self._sentence_outputs(dict(enc))
            head_rows = head_l[first_pos].float()                        # [n, C]
            if decoding == "greedy":
                pred_cols = head_rows.argmax(-1).tolist()
                if biaffine and getattr(self.config, "head_target", "word_index") == "first_subtoken":
                    heads = [pos2word.get(c, 0) for c in pred_cols]
                else:
                    heads = pred_cols                                    # may fall outside 0..n
            else:
                S = torch.log_softmax(head_rows, -1)[:, cols].double().cpu().numpy()
                heads = eisner(S) if decoding == "eisner" else chu_liu_edmonds(S)
                pred_cols = [cols[h] if h < len(cols) else h for h in heads]
            if biaffine:
                deprel_ids = [int(rel[:, first_pos[d], min(c, rel.shape[-1] - 1)].argmax()) for d, c in enumerate(pred_cols)]
            else:
                deprel_ids = deprel_l[first_pos].argmax(-1).tolist()
            upos_ids = upos_l[first_pos].argmax(-1).tolist()
            results.append([
                {"id": d + 1, "form": words[d], "upos": self.config.upos_labels[upos_ids[d]],
                 "head": int(heads[d]), "deprel": self.config.deprel_labels[deprel_ids[d]]}
                for d in range(n)])
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Linear head (any encoder loadable with AutoModel)
# ─────────────────────────────────────────────────────────────────────────────
class HubParserLinear(HubParserMixin, BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_deprel_labels = len(config.deprel_labels)
        self.num_upos_labels = len(config.upos_labels)
        self.num_head_labels = config.num_head_labels

        self.bert = AutoModel.from_config(config)
        self.deprel_classifier = nn.Linear(config.hidden_size, self.num_deprel_labels)
        self.upos_classifier = nn.Linear(config.hidden_size, self.num_upos_labels)
        self.head_classifier = nn.Linear(config.hidden_size, self.num_head_labels)

        classifier_dropout = getattr(config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(classifier_dropout)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                deprel_label=None, upos_label=None, head_label=None):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.config.model_type != "modernbert":
            encoder_kwargs["token_type_ids"] = token_type_ids
        sequence_output = self.dropout(self.bert(**encoder_kwargs)[0])

        logits_deprel = self.deprel_classifier(sequence_output)
        logits_upos = self.upos_classifier(sequence_output)
        logits_head = self.head_classifier(sequence_output)

        if deprel_label is not None and upos_label is not None and head_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = (
                loss_fct(logits_deprel.view(-1, self.num_deprel_labels), deprel_label.view(-1))
                + loss_fct(logits_upos.view(-1, self.num_upos_labels), upos_label.view(-1))
                + loss_fct(logits_head.view(-1, self.num_head_labels), head_label.view(-1))
            )
            return loss, logits_deprel, logits_upos, logits_head
        return logits_deprel, logits_upos, logits_head


class HubParserLinearModernBert(HubParserLinear):
    config_class = ModernBertConfig


# ─────────────────────────────────────────────────────────────────────────────
# Biaffine head
# ─────────────────────────────────────────────────────────────────────────────
class _BiaffineHeads:
    def _build_heads(self, config):
        self.num_deprel_labels = len(config.deprel_labels)
        self.num_upos_labels = len(config.upos_labels)
        arc_hidden, rel_hidden, mlp_dropout = config.arc_hidden, config.rel_hidden, config.mlp_dropout

        self.upos_classifier = nn.Linear(config.hidden_size, self.num_upos_labels)
        nn.init.xavier_uniform_(self.upos_classifier.weight)
        nn.init.zeros_(self.upos_classifier.bias)

        self.arc_head_mlp = MLP(config.hidden_size, arc_hidden, mlp_dropout)
        self.arc_dep_mlp = MLP(config.hidden_size, arc_hidden, mlp_dropout)
        self.rel_head_mlp = MLP(config.hidden_size, rel_hidden, mlp_dropout)
        self.rel_dep_mlp = MLP(config.hidden_size, rel_hidden, mlp_dropout)

        self.arc_biaffine = Biaffine(arc_hidden, out_features=1, bias_x=True, bias_y=False)
        self.rel_biaffine = Biaffine(rel_hidden, out_features=self.num_deprel_labels, bias_x=True, bias_y=True)

    def _heads_forward(self, seq, attention_mask, deprel_label, upos_label, head_label):
        seq = self.dropout(seq)
        B, L, _ = seq.shape

        logits_upos = self.upos_classifier(seq)

        logits_head = self.arc_biaffine(self.arc_dep_mlp(seq), self.arc_head_mlp(seq)).squeeze(1)
        if self.training and logits_head.abs().max() > 1e4:
            raise RuntimeError(f"logits_head.abs().max()={logits_head.abs().max():.3e}")
        if attention_mask is not None:
            logits_head = logits_head.masked_fill((attention_mask == 0).unsqueeze(1), -1e4)

        logits_rel = self.rel_biaffine(self.rel_dep_mlp(seq), self.rel_head_mlp(seq))  # [B, R, L, L]
        logits_rel_t = logits_rel.permute(0, 2, 3, 1).contiguous()                    # [B, L, L, R]

        # DEPREL evaluated at the predicted head
        arc_preds = logits_head.argmax(-1).clamp(0, L - 1)
        idx_pred = arc_preds.unsqueeze(-1).unsqueeze(-1).expand(B, L, 1, self.num_deprel_labels)
        logits_deprel_out = logits_rel_t.gather(2, idx_pred).squeeze(2)

        if head_label is None or deprel_label is None or upos_label is None:
            return logits_deprel_out, logits_upos, logits_head

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        oob_mask = (head_label != -100) & (head_label >= L)
        head_label_clean = head_label.clone()
        deprel_label_clean = deprel_label.clone()
        head_label_clean[oob_mask] = -100
        deprel_label_clean[oob_mask] = -100

        loss_head = loss_fct(logits_head.reshape(B * L, L), head_label_clean.reshape(-1))

        # DEPREL loss with teacher forcing on the gold head
        safe_heads = head_label_clean.clamp(0, L - 1)
        idx_gold = safe_heads.unsqueeze(-1).unsqueeze(-1).expand(B, L, 1, self.num_deprel_labels)
        logits_deprel_gold = logits_rel_t.gather(2, idx_gold).squeeze(2)
        loss_deprel = loss_fct(logits_deprel_gold.reshape(B * L, self.num_deprel_labels), deprel_label_clean.reshape(-1))

        loss_upos = loss_fct(logits_upos.reshape(B * L, self.num_upos_labels), upos_label.reshape(-1))

        for name, val in [("loss_head", loss_head), ("loss_deprel", loss_deprel), ("loss_upos", loss_upos)]:
            if torch.isnan(val) or torch.isinf(val):
                raise RuntimeError(f"{name}={val.item():.4e}")

        return loss_head + loss_deprel + loss_upos, logits_deprel_out, logits_upos, logits_head


class HubParserBiaffine(HubParserMixin, _BiaffineHeads, BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        encoder_dropout = (
            config.classifier_dropout if getattr(config, "classifier_dropout", None) is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(encoder_dropout)
        self._build_heads(config)
        self.post_init()

    def _init_weights(self, module):
        _init_biaffine_weights(self, module)

    def _encode(self, inputs):
        return self.bert(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                         token_type_ids=inputs.get("token_type_ids")).last_hidden_state

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                deprel_label=None, upos_label=None, head_label=None):
        seq = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids).last_hidden_state
        return self._heads_forward(seq, attention_mask, deprel_label, upos_label, head_label)


class HubParserBiaffineModernBert(HubParserMixin, _BiaffineHeads, ModernBertPreTrainedModel):
    config_class = ModernBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = ModernBertModel(config)
        encoder_dropout = (
            getattr(config, "classifier_dropout", None)
            or getattr(config, "hidden_dropout_prob", None)
            or getattr(config, "embedding_dropout", 0.1)
        )
        self.dropout = nn.Dropout(encoder_dropout)
        self._build_heads(config)
        self.post_init()

    def _init_weights(self, module):
        _init_biaffine_weights(self, module)

    def _encode(self, inputs):
        return self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                deprel_label=None, upos_label=None, head_label=None):
        seq = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self._heads_forward(seq, attention_mask, deprel_label, upos_label, head_label)


def hubparser_class(head: str, model_type: str):
    modern = model_type == "modernbert"
    if head == "linear":
        return HubParserLinearModernBert if modern else HubParserLinear
    return HubParserBiaffineModernBert if modern else HubParserBiaffine

# ═════════════════════════════ tree decoders ═════════════════════════════
NEG = -1e9


def _arc_matrix(S):
    """(n+1, n+1) matrix A[h, d] with A[h, 0] = -inf (nothing heads the root)."""
    n = S.shape[0]
    A = np.full((n + 1, n + 1), NEG)
    A[:, 1:] = S.T
    A[np.arange(n + 1), np.arange(n + 1)] = NEG  # no self loops
    return A


def _connected_to_root(heads):
    """heads[d-1] in 0..n or None (detached). Returns ok[v] = v reaches the root."""
    n = len(heads)
    ok = [False] * (n + 1)
    ok[0] = True
    for start in range(1, n + 1):
        path, cur = [], start
        while cur is not None and cur != 0 and not ok[cur] and cur not in path:
            path.append(cur)
            cur = heads[cur - 1]
        if cur is not None and ok[cur]:
            for p in path:
                ok[p] = True
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 4. Eisner (projective, exactly one root child)
# ─────────────────────────────────────────────────────────────────────────────
def _eisner_tables(A):
    """Inside tables over words 1..n. A[h, d] arc scores. Returns C, I, backpointers."""
    n = A.shape[0] - 1
    # C[s,t,0]: complete span, head t (left-pointing); C[s,t,1]: head s (right-pointing)
    C = np.full((n + 2, n + 2, 2), NEG)
    I = np.full((n + 2, n + 2, 2), NEG)
    bC = np.zeros((n + 2, n + 2, 2), dtype=int)
    bI = np.zeros((n + 2, n + 2, 2), dtype=int)
    for s in range(1, n + 1):
        C[s, s, 0] = C[s, s, 1] = 0.0
    for k in range(1, n):
        for s in range(1, n - k + 1):
            t = s + k
            vals = C[s, s:t, 1] + C[s + 1:t + 1, t, 0]
            r = int(np.argmax(vals))
            I[s, t, 0] = vals[r] + A[t, s]
            I[s, t, 1] = vals[r] + A[s, t]
            bI[s, t, 0] = bI[s, t, 1] = s + r
            vals = C[s, s:t, 0] + I[s:t, t, 0]
            r = int(np.argmax(vals))
            C[s, t, 0] = vals[r]
            bC[s, t, 0] = s + r
            vals = I[s, s + 1:t + 1, 1] + C[s + 1:t + 1, t, 1]
            r = int(np.argmax(vals))
            C[s, t, 1] = vals[r]
            bC[s, t, 1] = s + 1 + r
    return C, I, bC, bI


def eisner(S):
    n = S.shape[0]
    A = _arc_matrix(S)
    C, I, bC, bI = _eisner_tables(A)
    root_scores = [A[0, r] + C[1, r, 0] + C[r, n, 1] for r in range(1, n + 1)]
    r = 1 + int(np.argmax(root_scores))
    heads = [0] * n
    heads[r - 1] = 0

    def back_C(s, t, d):
        if s == t:
            return
        m = bC[s, t, d]
        if d == 0:
            back_C(s, m, 0)
            back_I(m, t, 0)
        else:
            back_I(s, m, 1)
            back_C(m, t, 1)

    def back_I(s, t, d):
        m = bI[s, t, d]
        if d == 0:
            heads[s - 1] = t
        else:
            heads[t - 1] = s
        back_C(s, m, 1)
        back_C(m + 1, t, 0)

    back_C(1, r, 0)
    back_C(r, n, 1)
    return heads


# ─────────────────────────────────────────────────────────────────────────────
# 5. Chu-Liu/Edmonds (non-projective MST, exactly one root child)
# ─────────────────────────────────────────────────────────────────────────────
def _find_cycle(heads):
    n = len(heads) - 1  # heads[0] unused
    color = [0] * (n + 1)
    for start in range(1, n + 1):
        path, cur = [], start
        while cur != 0 and color[cur] == 0:
            color[cur] = 1
            path.append(cur)
            cur = heads[cur]
        if cur != 0 and color[cur] == 1:
            return path[path.index(cur):]
        for p in path:
            color[p] = 2
    return None


def _cle(A):
    """A[h, d], nodes 0..n with 0 = root. Returns heads[d] for d=1..n (index 0 unused)."""
    n = A.shape[0] - 1
    heads = [0] + [int(np.argmax(A[:, d])) for d in range(1, n + 1)]
    cycle = _find_cycle(heads)
    if cycle is None:
        return heads
    cyc = set(cycle)
    rest = [v for v in range(n + 1) if v not in cyc]          # rest[0] == 0
    idx = {v: i for i, v in enumerate(rest)}
    c = len(rest)                                              # contracted node id
    B = np.full((c + 1, c + 1), NEG)
    enter_from, leave_to = {}, {}
    cyc_score = sum(A[heads[v], v] for v in cycle)
    for u in rest:
        for v in rest:
            if u != v:
                B[idx[u], idx[v]] = A[u, v]
        # arc entering the cycle from u: replace one cycle arc
        gains = [(A[u, v] - A[heads[v], v], v) for v in cycle]
        g, v = max(gains)
        B[idx[u], c] = cyc_score + g
        enter_from[u] = v
        # arc leaving the cycle to u
        if u != 0:
            s, w = max((A[w, u], w) for w in cycle)
            B[c, idx[u]] = s
            leave_to[u] = w
    sub = _cle(B)
    out = list(heads)
    for u in rest[1:]:
        h = sub[idx[u]]
        out[u] = leave_to[u] if h == c else rest[h]
    u = rest[sub[c]]
    out[enter_from[u]] = u
    return out


def chu_liu_edmonds(S):
    n = S.shape[0]
    A = _arc_matrix(S)
    heads = _cle(A)[1:]
    roots = [d for d in range(1, n + 1) if heads[d - 1] == 0]
    if len(roots) == 1:
        return heads
    # one-root constraint: try each root child, keep the best tree
    best, best_score = None, -np.inf
    for r in range(1, n + 1):
        B = A.copy()
        B[0, :] = NEG
        B[0, r] = A[0, r]
        h = _cle(B)[1:]
        sc = sum(A[h[d - 1], d] for d in range(1, n + 1))
        if sc > best_score:
            best, best_score = h, sc
    return best


