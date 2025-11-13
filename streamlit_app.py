# -------------------------
# Markov fallback generator (filtered to real words)
# -------------------------
class Markov:
    def __init__(self):
        # map: (w1,w2) -> { next_word: count, ... }
        self.map: Dict[Tuple[str,str], Dict[str,int]] = {}
        # list of starting bigrams observed
        self.starts: List[Tuple[str,str]] = []

    def train(self, text: str):
        toks = tokenize(text)
        if len(toks) < 3:
            return
        self.starts.append((toks[0].lower(), toks[1].lower()))
        for i in range(len(toks)-2):
            key = (toks[i].lower(), toks[i+1].lower())
            nxt = toks[i+2].lower()
            self.map.setdefault(key, {})
            self.map[key][nxt] = self.map[key].get(nxt, 0) + 1

    def _best_choice(self, choices: Dict[str,int]) -> Optional[str]:
        """Return the most likely next token (argmax)."""
        if not choices:
            return None
        best = max(sorted(choices.items()), key=lambda kv: kv[1])
        return best[0]

    def _valid_tokens_set(self) -> set:
        """Return a set of 'real' tokens to prefer: merged dictionary tokens + vocab.
           Ensure vocab exists by calling build_vocab() if needed."""
        # prefer using global VOCAB if present; otherwise build one
        try:
            source_vocab = set(VOCAB) if VOCAB else set(build_vocab())
        except Exception:
            source_vocab = set(build_vocab())
        # also include tokens from merged dictionary keys (multi-word keys will be tokenized)
        md = merged_dictionary()
        dict_tokens = set()
        for k,v in md.items():
            dict_tokens.update(tokenize(k))
            dict_tokens.update(tokenize(v.get("definition","")))
            for ex in v.get("examples",[]):
                dict_tokens.update(tokenize(ex))
        return source_vocab | dict_tokens

    def _is_real_word(self, tok: str, allowed_set: set) -> bool:
        # allow single-letter 'a' and 'i'
        if not re.fullmatch(r"[a-zA-Z']+", tok):
            return False
        if len(tok) == 1 and tok.lower() not in ("a","i"):
            return False
        # require presence in allowed_set when possible
        if allowed_set and tok not in allowed_set:
            return False
        return True

    def _best_choice_filtered(self, choices: Dict[str,int]) -> Optional[str]:
        """Pick the highest-count choice that also passes the 'real word' checks.
           Fall back to unfiltered best choice if none pass."""
        if not choices:
            return None
        allowed = self._valid_tokens_set()
        # create list of (token,count) filtered by validity
        filtered = [(w,c) for w,c in choices.items() if self._is_real_word(w, allowed)]
        if filtered:
            # deterministic tie-break by sorted token then count
            best = max(sorted(filtered), key=lambda kv: kv[1])
            return best[0]
        # fallback to any best (unfiltered)
        return self._best_choice(choices)

    def _best_unigram_after(self, token: str) -> Optional[str]:
        """Backoff: look for bigrams that start with token and choose most common follower overall,
           but prefer 'real' words."""
        candidates: Dict[str,int] = {}
        for (a,b), nxts in self.map.items():
            if a == token:
                for w,cnt in nxts.items():
                    candidates[w] = candidates.get(w,0) + cnt
        if not candidates:
            return None
        # prefer filtered candidate
        filt = [(w,c) for w,c in candidates.items() if self._is_real_word(w, self._valid_tokens_set())]
        if filt:
            best = max(sorted(filt), key=lambda kv: kv[1])
            return best[0]
        return self._best_choice(candidates)

    def generate(self, seed: str=None, max_words:int=40) -> str:
        """
        Deterministic greedy generation with 'real word' filtering:
         - If seed has >=2 tokens and we have a matching bigram, greedily pick the most likely next real word
           and repeat using the last two words each step, returning only the NEW words (continuation).
         - Back off to a unigram-after-last-token heuristic if exact bigram missing.
         - If no seed or nothing found, produce a full sentence from a random start (still preferring real words).
        """
        if seed:
            toks = tokenize(seed)
            if len(toks) >= 2:
                key = (toks[-2].lower(), toks[-1].lower())
                continuation: List[str] = []
                for _ in range(max_words):
                    if key in self.map:
                        nxt = self._best_choice_filtered(self.map[key])
                    else:
                        nxt = self._best_unigram_after(key[1])
                    if not nxt:
                        break
                    # safety: avoid weird short-token loops
                    if len(continuation) >= 2 and continuation[-1] == nxt and len(nxt) <= 2:
                        break
                    continuation.append(nxt)
                    key = (key[1], nxt)
                if continuation:
                    return " ".join(continuation)
        # No seed or couldn't continue: generate full sentence preferring real words
        if not self.starts:
            return ""
        key = random.choice(self.starts)
        out = [key[0], key[1]]
        for _ in range(max_words-2):
            choices = self.map.get((out[-2], out[-1]))
            if not choices:
                break
            nxt = self._best_choice_filtered(choices)
            if not nxt:
                break
            out.append(nxt)
            if len(out) >= 3 and out[-1] == out[-2] == out[-3]:
                break
        return " ".join(out)

MARKOV = Markov()
