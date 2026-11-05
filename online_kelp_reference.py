"""Paired online Kelp replay; reference implementation, not a latency benchmark.

Uses the public Kelp head's attention/cfc modules with the paper's inference
dt=1/2048. It deliberately differs from the length-dependent public forward().
Requires Qwen3 and a head checkpoint trained for the same hidden-state index.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def online_generate(model, tokenizer, prompt_ids, head=None, layer=20,
                    threshold=0.5, max_new_tokens=1024, seed=20,
                    temperature=0.7, top_p=0.9, top_k=20):
    """Return only approved output tokens; also retain trigger information."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = model.device
    prompt_ids = prompt_ids.to(device)
    out = model(input_ids=prompt_ids, use_cache=True,
                output_hidden_states=head is not None, return_dict=True)
    cache, next_logits = out.past_key_values, out.logits[:, -1, :]
    risks, accepted = [], []
    header_tokens = 0
    blocked = False
    if head is not None:
        # Use the last structural assistant marker, not one inside user text.
        marker = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
        ids = prompt_ids[0].tolist()
        starts = [i + len(marker) for i in range(len(ids) - len(marker) + 1)
                  if ids[i:i + len(marker)] == marker]
        if not starts:
            raise ValueError('Qwen assistant marker missing')
        assistant_start = starts[-1]
        attn = head.attention
        hidden = out.hidden_states[layer].to(next(head.parameters()).dtype)
        keys, values = attn.key(hidden), attn.value(hidden)
        prefix = attn(hidden[:, :assistant_start])[0]
        weights = torch.softmax(head.prefix_scorer(prefix).squeeze(-1), dim=1)
        memory = head.prefix_to_h(torch.bmm(weights.unsqueeze(1), prefix).squeeze(1))
        # Qwen non-thinking inserts an empty think block into the assistant prefix.
        # Feed it through the recurrence, but never treat it as user-visible text.
        header_tokens = len(ids) - assistant_start
        for pos in range(assistant_start, len(ids)):
            query = attn.query(hidden[:, pos:pos + 1])
            scores = query @ keys[:, :pos + 1].transpose(-1, -2)
            scores = scores / ((keys.size(-1) + 1e-6) ** 0.5)
            feature = (torch.softmax(scores, dim=-1) @ values[:, :pos + 1]).squeeze(1)
            memory = head.cfc(feature, memory, 1.0 / 2048)
            risk = torch.softmax(head.mem_head(memory).float(), dim=-1)[0, 1].item()
            risks.append(risk)
            if threshold is not None and risk >= threshold:
                blocked = True
                break
    del out
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    eos_ids.add(tokenizer.convert_tokens_to_ids('<|im_end|>'))
    for _ in range(max_new_tokens):
        if blocked:
            break
        logits = next_logits.float() / temperature
        if top_k:
            cutoff = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, -float('inf'))
        sorted_logits, indices = logits.sort(descending=True, dim=-1)
        remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.scatter(1, indices, sorted_logits.masked_fill(remove, -float('inf')))
        token = torch.multinomial(logits.softmax(dim=-1), 1)
        if token.item() in eos_ids:
            break
        # Process the selected token BEFORE releasing it. The previous hidden
        # state belongs to the preceding input token, not this candidate token.
        out = model(input_ids=token, past_key_values=cache, use_cache=True,
                    output_hidden_states=head is not None, return_dict=True)
        cache, next_logits = out.past_key_values, out.logits[:, -1, :]
        if head is not None:
            hidden = out.hidden_states[layer][:, -1:, :].to(keys.dtype)
            keys = torch.cat([keys, attn.key(hidden)], dim=1)
            values = torch.cat([values, attn.value(hidden)], dim=1)
            scores = attn.query(hidden) @ keys.transpose(-1, -2)
            scores = scores / ((keys.size(-1) + 1e-6) ** 0.5)
            feature = (torch.softmax(scores, dim=-1) @ values).squeeze(1)
            memory = head.cfc(feature, memory, 1.0 / 2048)
            risk = torch.softmax(head.mem_head(memory).float(), dim=-1)[0, 1].item()
            risks.append(risk)
            if threshold is not None and risk >= threshold:
                blocked = True
                del out
                break
        accepted.append(token.item())
        del out
    return {
        'text': tokenizer.decode(accepted, skip_special_tokens=True),
        'released_token_ids': accepted,
        'released_tokens': len(accepted),
        'blocked': blocked,
        'trigger_assistant_index': len(risks) - 1 if blocked else None,
        'assistant_header_tokens': header_tokens,
        'max_risk': max(risks) if risks else None,
        'risk_scores': risks,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--autodan_json', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--kelp_dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen3-8B')
    p.add_argument('--layer', type=int, default=20)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--observe_only', action='store_true', help='Score full responses without blocking, for calibration')
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--seeds', type=int, nargs='+', default=[20])
    args = p.parse_args()
    if not 0 < args.threshold < 1 or args.max_new_tokens < 1:
        p.error('threshold must be in (0,1) and max_new_tokens must be positive')
    spec = importlib.util.spec_from_file_location('kelp_head', args.kelp_dir / 'models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to('cuda:0').eval()
    head = module.StreamingSafetyHead(
        input_dim=AutoConfig.from_pretrained(args.model).hidden_size,
        proj_dim=1024, mem_dim=1024, num_labels=2, use_dt=True)
    head.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    head.to(device=model.device, dtype=torch.bfloat16).eval()
    records = json.loads(args.autodan_json.read_text(encoding='utf-8'))
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sample_id, item in records.items():
        prompt = item['final_suffix'].replace('[REPLACE]', item['goal'].lower())
        prefix = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_tensors='pt')
        if prefix.size(1) + args.max_new_tokens > 4096:
            raise ValueError(f'{sample_id}: exceeds the 4096-token experiment limit; do not silently truncate')
        for seed in args.seeds:
            kwargs = dict(layer=args.layer, threshold=None if args.observe_only else args.threshold,
                          max_new_tokens=args.max_new_tokens, seed=seed)
            bare = online_generate(model, tokenizer, prefix, **kwargs)
            guarded = online_generate(model, tokenizer, prefix, head=head, **kwargs)
            released = guarded['released_token_ids']
            if released != bare['released_token_ids'][:len(released)]:
                raise RuntimeError(f'{sample_id}: paired generated prefixes diverged')
            results.append({'id': sample_id, 'goal': item['goal'], 'seed': seed,
                            'bare': bare, 'guarded': guarded})
            args.output.write_text(json.dumps({
                'protocol': 'online fixed-dt Kelp reference; dt=1/2048',
                'threshold': None if args.observe_only else args.threshold, 'layer': args.layer,
                'max_new_tokens': args.max_new_tokens,
                'temperature': 0.7, 'top_p': 0.9, 'top_k': 20,
                'records': results}, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{sample_id} seed={seed} blocked={guarded["blocked"]}')


if __name__ == '__main__':
    main()
