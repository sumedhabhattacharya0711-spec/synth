"""
orchestrator.py — keyphrase extraction (YAKE + spaCy + regex), arXiv
vocabulary bootstrap, structural PS parsing, combinatorial hypothesis
generation, LLM polish.
Corresponds to notebook Cell 4.
"""

from .config import re, time, defaultdict, Counter, arxiv, safe_parse_json

import yake

# ---------- Utilities ----------

def clean_term(t):
    t = re.sub(r"^(for|to|the|a|an|of|in|on|with|by|as|and|or|towards?|"
               r"can|will|should|could|would|may|might|must|do|does|did|"
               r"how|what|which|where|when|why|is|are|be|been|being|"
               r"has|have|had|make|made|more|while|that|this|these|"
               r"its|it|they|their|our|your|we|also|very|most|"
               r"some|any|each|every|all|both|many|much|such)\s+",
               "", t.strip(), flags=re.I)
    t = re.sub(r"\s+(for|to|the|a|an|of|in|on|with|by|as|and|or|towards?|"
               r"can|will|should|could|would|may|might|must|"
               r"how|what|which|where|when|why|is|are|be|been|"
               r"has|have|had|make|made|more|while|that|this|these|"
               r"its|it|they|their|our|your|we|also)$",
               "", t.strip(), flags=re.I)
    return t.strip().strip(".,;:()\"'")

def extract_domain_anchor(ps):
    """Extract domain from YAKE's top non-generic keyphrase."""
    kw = yake.KeywordExtractor(lan="en", n=2, top=5, dedupLim=0.5)
    keywords = kw.extract_keywords(ps)
    
    generic = {"design", "build", "system", "solution", "approach",
               "challenge", "problem", "method", "technique", "tool",
               "building", "understanding", "development", "application",
               "create", "demonstrate", "implement", "prototype"}
    
    for phrase, score in keywords:
        words = set(phrase.lower().split())
        if not words <= generic and len(phrase) >= 4:
            return phrase.lower()
    
    return ""

# ---------- Stage 1: Keyphrase extraction (YAKE + spaCy + regex) ----------

def extract_keyphrases(ps, top_n=20):
    phrases = []
    
    # ===== PRIORITY 1: Structural regex (highest confidence) =====
    
    # 1a. ALL-CAPS and CamelCase
    named = re.findall(
        r"\b(?:[A-Z][A-Za-z]*[A-Z]+[a-z]*|[A-Z]{2,})\b(?:\s+[\d.]+)?", ps)
    noise = {"THE","AND","FOR","WITH","YOUR","THIS","THAT","CAN",
             "NOT","BUT","ARE","WAS","HAS","HOW","KEY","AI","UI","UX",
             "BY","OR","IN","ON","AN","IT","IF","SO","DO","NO","UP"}
    for n in named:
        n = n.strip()
        if n not in noise and len(n) >= 2:
            phrases.append((n, 1.0))
    
    # 1b. "using/via/such as X" tool names
    for t in re.findall(
        r"(?:using|via|with|like|e\.g\.,?\s*|such as|including|"
        r"based on|powered by)\s+"
        r"([A-Z][\w\s\-.]+?)(?=[,.\);\s]+(?:and|or|for|to|with|while|that|\n)|$)", ps):
        for p in re.split(r"\s+(?:and|or)\s+", t):
            clean = clean_term(p)
            if len(clean) >= 3:
                phrases.append((clean, 0.95))
    
    # 1c. Listed items — "such as X, Y, Z"
    for item in re.findall(
        r"(?:such as|like|e\.g\.,?|including)\s+(.+?)(?:\)|\.|\n)", ps, re.I):
        for p in re.split(r",\s*(?:and\s+)?|,?\s+and\s+|,?\s+or\s+", item):
            clean = clean_term(p)
            if 1 <= len(clean.split()) <= 5 and len(clean) >= 4:
                phrases.append((clean, 0.9))
    
    # 1d. Parenthetical items
    for match in re.findall(r"\(([^)]+)\)", ps):
        for item in re.split(r"[,/]\s*|\s+and\s+|\s+or\s+", match):
            clean = clean_term(item.strip())
            if len(clean) >= 4 and len(clean.split()) <= 4:
                phrases.append((clean.lower(), 0.85))
    
    # 1e. Constraint phrases
    for c in re.findall(
        r"((?:lightweight|mobile.first|energy.efficient|low.compute|"
        r"real.time|on.device|compute.efficient|resource.constrained|"
        r"large.scale|scalable|noisy|sparse|"
        r"interpretab\w+|evidence.based|query.driven|data.driven|"
        r"end.to.end|multi.turn|context.aware)\w*"
        r"(?:\s+\w+)?)", ps, re.I):
        clean = clean_term(c)
        if len(clean) >= 5:
            phrases.append((clean.lower(), 0.8))
    
    # ===== PRIORITY 2: YAKE (statistical — fills gaps) =====
    
    kw_extractor = yake.KeywordExtractor(
        lan="en", n=3, top=30, dedupLim=0.5)
    yake_keywords = kw_extractor.extract_keywords(ps)
    
    yake_stopwords = {"design", "build", "demonstrate", "combine", "system",
                      "solution", "challenge", "approach", "work", "use",
                      "show", "enable", "provide", "require", "support",
                      "called", "based", "first", "novel", "new", "key",
                      "also", "well", "just", "even", "still"}
    
    existing = {p.lower() for p, s in phrases}
    
    for phrase, score in yake_keywords:
        clean = clean_term(phrase)
        if len(clean) < 4 or len(clean.split()) > 3:
            continue
        if clean.lower() in existing:
            continue
        words = set(clean.lower().split())
        if words <= yake_stopwords:
            continue
        if any(clean.lower() in ex for ex in existing):
            continue
        importance = max(0.3, min(0.7, 1.0 - score))
        phrases.append((clean.lower(), importance))
    
    # deduplicate — keep highest score
    seen = {}
    for phrase, score in phrases:
        if phrase and (phrase not in seen or score > seen[phrase]):
            seen[phrase] = score
    
    # remove substrings — keep longer phrase
    final = {}
    sorted_phrases = sorted(seen.items(), key=lambda x: -len(x[0]))
    for phrase, score in sorted_phrases:
        if not any(phrase in other and phrase != other
                   for other in final):
            final[phrase] = score
    
    return sorted(final.items(), key=lambda x: -x[1])[:top_n]

# ---------- Shared term relevance scoring ----------
#
# parse_ps_universal's category lists (models/techniques/constraints/tasks/
# modalities) are built via list(set(...)), whose iteration order isn't
# stable across runs (str hash randomization) — so every downstream [:N]
# slice and cartesian pairing was silently nondeterministic. score_terms
# gives those lists a real, reproducible order; _terms_compatible uses that
# same grounding info to stop pairing two terms that never actually relate
# to each other in the source text.

METHOD_KEYWORDS = {"attention", "augmented", "graph", "neural", "transformer",
                   "embedding", "retrieval", "generation", "extraction",
                   "classification", "detection", "clustering", "prediction",
                   "inference", "attribution", "contrastive", "supervised",
                   "reinforced", "adaptive", "hierarchical", "sparse", "dense",
                   "hybrid", "recurrent", "convolutional", "masked", "gated",
                   "dynamic", "iterative"}

def score_terms(terms, ps_lower, keyphrases=None, arxiv_vocab=None):
    """Rank extracted terms by relevance (keyphrase importance + arXiv
    vocab frequency + literal PS presence). Ties break alphabetically so
    the result is fully deterministic regardless of input order."""
    kp_scores = {p.lower(): s for p, s in (keyphrases or [])}
    vocab_counts = {t.lower(): c for t, c in (arxiv_vocab or [])}

    scored = []
    for term in terms:
        t_lower = term.lower()
        score = 0.0
        if t_lower in kp_scores:
            score += kp_scores[t_lower] * 2.0
        else:
            term_words = set(t_lower.split())
            best_overlap = max(
                (kp_scores[kp] for kp in kp_scores
                 if term_words & set(kp.split())), default=0)
            score += best_overlap * 0.5
        if t_lower in vocab_counts:
            score += min(vocab_counts[t_lower] / 5.0, 1.0)
        if t_lower in ps_lower:
            score += 0.5
        scored.append((term, round(score, 3)))

    return sorted(scored, key=lambda x: (-x[1], x[0]))

def _terms_compatible(term_a, term_b, ps_lower, keyphrase_words, window=300):
    """Gate a cartesian pairing: allow it if both terms co-occur near each
    other in the PS text, or at least one side is grounded in the PS
    (literal substring or shares a keyphrase word). Blocks pairing two
    purely speculative bootstrapped terms that have no textual link."""
    a, b = term_a.lower(), term_b.lower()
    a_in_ps, b_in_ps = a in ps_lower, b in ps_lower
    if a_in_ps and b_in_ps:
        return abs(ps_lower.find(a) - ps_lower.find(b)) <= window
    a_core = a_in_ps or bool(set(a.split()) & keyphrase_words)
    b_core = b_in_ps or bool(set(b.split()) & keyphrase_words)
    return a_core or b_core

# ---------- Stage 2: arXiv vocabulary bootstrap ----------

METHOD_PATTERN = re.compile(
    r"(\w+(?:-\w+)?\s+(?:attention|attribution|augmented|based|driven|"
    r"enhanced|guided|aware|informed|grounded|gated|masked|contrastive|"
    r"adversarial|supervised|unsupervised|reinforced|pretrained|"
    r"distilled|pruned|quantized|sparse|dense|hybrid|"
    r"hierarchical|recursive|iterative|adaptive|dynamic|"
    r"neural|graph|transformer|convolutional|recurrent|"
    r"embedding|encoding|decoding|generation|extraction|"
    r"classification|detection|segmentation|retrieval|ranking|"
    r"clustering|summarization|prediction|inference)s?\b"
    r"|\b(?:RAG|GNN|BERT|GPT|LLM|NER|NLI|QA|IR|NLP|ASR|TTS|CNN|RNN|"
    r"LSTM|GRU|VAE|GAN|RL|RLHF|DPO|SFT|ICL|CoT)\b)", re.I)

def bootstrap_vocabulary(keyphrases, papers_per_query=5):
    title_ngrams = []
    paper_titles = []
    for phrase, score in keyphrases[:8]:
        try:
            results = list(arxiv.Client().results(
                arxiv.Search(query=phrase, max_results=papers_per_query)))
            for r in results:
                paper_titles.append(r.title)
                words = [w.strip(".,;:()[]") for w in r.title.lower().split()]
                for i in range(len(words)):
                    if i + 1 < len(words):
                        title_ngrams.append(f"{words[i]} {words[i+1]}")
                    if i + 2 < len(words):
                        title_ngrams.append(
                            f"{words[i]} {words[i+1]} {words[i+2]}")
                methods_found = METHOD_PATTERN.findall(r.title)
                for m in methods_found:
                    clean = clean_term(m.strip())
                    if len(clean) >= 5:
                        title_ngrams.append(clean.lower())
            time.sleep(1.0)
        except Exception as e:
            print(f"  bootstrap skip: {e}")
    
    generic = {"of the","in the","for the","and the","with the",
               "based on","a novel","we propose","this paper",
               "et al","such as","as well","can be","is a",
               "we present","in this","has been"}
    common = Counter(title_ngrams).most_common(40)
    vocab = []
    for phrase, count in common:
        clean = clean_term(phrase)
        if count >= 2 and clean not in generic and len(clean) > 5:
            vocab.append((clean, count))
    
    # filter bootstrap terms against keyphrases to avoid domain pollution
    kp_words = set()
    for phrase, score in keyphrases:
        kp_words.update(phrase.lower().split())
    
    grounded_vocab = []
    for term, count in vocab:
        term_words = set(term.lower().split())
        if term_words & kp_words or count >= 3:
            grounded_vocab.append((term, count))
    
    return grounded_vocab, paper_titles

# ---------- Stage 3: Structural parsing ----------

def parse_ps_universal(ps, arxiv_vocab, keyphrases=None):
    acronyms = list(set(re.findall(
        r"\b[A-Z][A-Za-z]*[A-Z]+[a-z]*\b|\b[A-Z]{2,}\b", ps)))
    noise = {"THE","AND","FOR","WITH","YOUR","THIS","THAT","CAN",
             "NOT","BUT","ARE","WAS","HAS","HOW","KEY","AI","UI","UX",
             "BY","OR","IN","ON","AN","IT","IF","SO","DO","NO","UP"}
    acronyms = [a for a in acronyms if a not in noise and len(a) >= 2]
    
    proper_nouns = list(set(re.findall(
        r"(?:[A-Z][a-z]+(?:\s+|-)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", ps)))
    
    tools = [clean_term(t) for t in re.findall(
        r"(?:using|via|with|like|e\.g\.,?|such as|including)\s+"
        r"([A-Z][\w\s\-]+?)(?:\s*[,.\)]|\s+(?:and|or|for|to))", ps)
        if len(clean_term(t)) >= 2]
    
    tech_compounds = [clean_term(t).lower() for t in re.findall(
        r"(\w+(?:\s+\w+)?)\s+(?:model|architecture|mechanism|framework|"
        r"network|algorithm|method|technique|approach|pipeline|backbone|"
        r"adapter)s?", ps, re.I) if len(clean_term(t)) >= 4]
    
    constraints = list(set(c.lower() for c in re.findall(
        r"(?:lightweight|mobile|edge|real.?time|low.?compute|"
        r"energy.?efficient|efficient|compact|limited|"
        r"mobile.first|on.?device|resource.?constrained|"
        r"large.scale|scalable|noisy|sparse|"
        r"interpretab\w+|robust)", ps, re.I)
        if len(c) >= 5))
    
    tasks = list(set(clean_term(t).lower() for t in re.findall(
        r"(\w+[-]?\w*\s+(?:editing|generation|detection|segmentation|"
        r"classification|removal|enhancement|correction|"
        r"processing|synthesis|recognition|prediction|"
        r"optimization|compression|inference|training|"
        r"selection|transfer|retrieval|extraction|"
        r"inpainting|relighting|stylization|quantization|"
        r"explanation|analysis|querying|ranking|monitoring|"
        r"clustering|summarization|coaching|intervention|"
        r"learning|reasoning|adaptation|understanding))", ps, re.I)
        if len(clean_term(t)) >= 5))
    
    modalities = list(set(m.lower() for m in re.findall(
        r"(?:voice|gesture|stylus|text|image|video|speech|"
        r"conversational|context.?aware|natural\s*language|"
        r"touch|tap|prompt|multimodal|visual|verbal|"
        r"dialogue|transcript|utterance)", ps, re.I)))
    
    arxiv_terms = [clean_term(phrase) for phrase, count in arxiv_vocab[:15]]
    arxiv_terms = [t for t in arxiv_terms if len(t) >= 4]
    
    models = list(set([a.strip() for a in acronyms] +
                      [p.strip() for p in proper_nouns] +
                      [t.strip() for t in tools]))
    techniques = list(set(tech_compounds + arxiv_terms))
    
    # filter tasks in negative context
    ps_lower = ps.lower()
    negative_markers = ["not ", "aren't ", "don't ", "cannot ", "isn't ",
                        "no ", "never ", "lack ", "without ", "not real"]
    filtered_tasks = []
    for t in tasks:
        t_pos = ps_lower.find(t.lower())
        if t_pos >= 0:
            context_before = ps_lower[max(0, t_pos-40):t_pos]
            if any(neg in context_before for neg in negative_markers):
                continue
        filtered_tasks.append(t)
    tasks = filtered_tasks

    def ranked(term_list):
        return [t for t, s in score_terms(term_list, ps_lower, keyphrases, arxiv_vocab)]

    return {
        "models": ranked([m for m in models if len(m) >= 2]),
        "techniques": ranked([t for t in techniques if len(t) >= 4]),
        "constraints": ranked(constraints),
        "tasks": ranked(tasks),
        "modalities": ranked(modalities),
        "arxiv_vocabulary": arxiv_terms,
    }

# ---------- Stage 4: Combinatorial generation ----------

def generate_combinatorial(parsed, keyphrases, domain_anchor, ps=""):
    hypotheses = []
    ps_lower = ps.lower()
    keyphrase_words = set()
    for phrase, score in keyphrases:
        keyphrase_words.update(phrase.lower().split())

    def compatible(a, b):
        # no PS text to check co-occurrence against — don't gate
        return True if not ps_lower else _terms_compatible(
            a, b, ps_lower, keyphrase_words)

    for tech in parsed["techniques"][:8]:
        for constraint in parsed["constraints"][:4]:
            if tech.lower() == constraint.lower():
                continue
            if not compatible(tech, constraint):
                continue
            hypotheses.append({
                "hypothesis": f"Applying {tech} under {constraint} "
                             f"constraints for {domain_anchor}",
                "search_queries": [
                    f"{tech} {constraint} {domain_anchor}",
                    f"efficient {tech} {constraint}",
                    f"{tech} {domain_anchor} optimization",
                ],
                "section_type": "feasibility",
                "source_terms": [tech, constraint],
            })

    for model in parsed["models"][:8]:
        candidate_tasks = [t for t in parsed["tasks"][:6]
                           if compatible(model, t)][:2]
        for task in candidate_tasks:
            hypotheses.append({
                "hypothesis": f"Using {model} for {task}",
                "search_queries": [
                    f"{model} {task}",
                    f"{model} {task} {domain_anchor}",
                    f"{model} based {task}",
                ],
                "section_type": "method",
                "source_terms": [model, task],
            })

    for mod in parsed["modalities"][:5]:
        for task in parsed["tasks"][:3]:
            if mod.lower() in task.lower():
                continue
            if not compatible(mod, task):
                continue
            hypotheses.append({
                "hypothesis": f"Enabling {task} through {mod} input "
                             f"for {domain_anchor}",
                "search_queries": [
                    f"{mod} {task} {domain_anchor}",
                    f"{mod} driven {task}",
                ],
                "section_type": "method",
                "source_terms": [mod, task],
            })

    techs = parsed["techniques"][:8]
    for i in range(len(techs)):
        for j in range(i + 1, min(i + 3, len(techs))):
            if not compatible(techs[i], techs[j]):
                continue
            hypotheses.append({
                "hypothesis": f"Comparing {techs[i]} and {techs[j]} "
                             f"for {domain_anchor}",
                "search_queries": [
                    f"{techs[i]} vs {techs[j]}",
                    f"{techs[i]} {techs[j]} {domain_anchor}",
                ],
                "section_type": "sota",
                "source_terms": [techs[i], techs[j]],
            })

    for phrase, score in keyphrases[:6]:
        hypotheses.append({
            "hypothesis": f"State of the art in {phrase} for {domain_anchor}",
            "search_queries": [
                f"{phrase} {domain_anchor}",
                f"{phrase} survey",
                f"recent {phrase}",
            ],
            "section_type": "sota",
            "source_terms": [phrase],
            "keyphrase_score": score,
        })

    arxiv_methods = [t for t in parsed.get("arxiv_vocabulary", [])
                     if any(kw in t.lower() for kw in METHOD_KEYWORDS)]

    for method in arxiv_methods[:6]:
        for task in parsed["tasks"][:4]:
            if not compatible(method, task):
                continue
            hypotheses.append({
                "hypothesis": f"Applying {method} to {task} in {domain_anchor}",
                "search_queries": [
                    f"{method} {task}",
                    f"{method} {domain_anchor}",
                    f"{method} for {task}",
                ],
                "section_type": "method",
                "source_terms": [method, task],
            })

    return hypotheses

# ---------- Dedup + coverage ----------

def deduplicate(hypotheses):
    def terms(h):
        return set(" ".join(h["search_queries"]).lower().split())
    unique, seen = [], []
    for h in hypotheses:
        t = terms(h)
        if not any(len(t & s) / max(len(t | s), 1) > 0.5 for s in seen):
            unique.append(h)
            seen.append(t)
    return unique

def cap_raw_hypotheses(hypotheses, max_per_type=8):
    """Bound how many candidates reach the network-bound validation stage.
    Generation order is now deterministic and relevance-ranked (see
    score_terms), so keeping the first N per section_type keeps the
    highest-ranked candidates instead of an unbounded cartesian product."""
    counts = defaultdict(int)
    capped = []
    for h in hypotheses:
        t = h["section_type"]
        if counts[t] < max_per_type:
            capped.append(h)
            counts[t] += 1
    return capped

def check_coverage(hypotheses, parsed):
    covered = set()
    for h in hypotheses:
        covered.update(t.lower() for t in h["source_terms"])
    all_terms = set()
    for key in ["models", "techniques", "constraints", "tasks", "modalities"]:
        all_terms.update(t.lower() for t in parsed.get(key, []))
    missing = [t for t in (all_terms - covered) if len(t) >= 3]
    if missing:
        print(f"  Coverage gap — adding: {missing[:5]}")
        for term in missing[:5]:
            hypotheses.append({
                "hypothesis": f"State of the art in {term}",
                "search_queries": [f"{term} survey", f"{term} recent methods"],
                "section_type": "sota",
                "source_terms": [term],
            })
    return hypotheses

# ---------- arXiv validation ----------
#
# Two-tier check, not one binary gate:
#   - hard filter (terms_grounded): every individual source_term must have
#     at least one arXiv hit on its own. This catches ungrounded/junk
#     extraction artifacts — it does NOT require the *combination* to have
#     prior art, so a novel pairing of two real things survives.
#   - soft signal (arxiv_hits / title_match_rate / novel): the combined
#     query is still searched and scored, but only recorded, not used to
#     exclude. select_top's scoring already weights hits, so well-
#     precedented hypotheses still rank higher without novel ones being
#     deleted before they're ever scored.

def validate_arxiv(hypotheses, min_hits=2, min_match=0.3):
    term_cache = {}

    def term_grounded(term):
        key = term.lower()
        if key in term_cache:
            return term_cache[key]
        try:
            results = list(arxiv.Client().results(
                arxiv.Search(query=term, max_results=2)))
            grounded = len(results) >= 1
            time.sleep(0.5)
        except Exception:
            grounded = True  # network hiccup — don't punish the term for it
        term_cache[key] = grounded
        return grounded

    validated = []
    for h in hypotheses:
        terms_ok = all(term_grounded(t) for t in h.get("source_terms", []))

        query = h["search_queries"][0]
        query_words = set(re.findall(r"[a-z]+", query.lower()))
        query_words -= {"the","a","an","of","in","for","and","or","to",
                        "with","by","on","vs","based","efficient","recent",
                        "survey","optimization","editor","system","pipeline",
                        "analysis","approach","method"}
        try:
            results = list(arxiv.Client().results(
                arxiv.Search(query=query, max_results=5)))
            hits = len(results)
            if query_words and results:
                matches = sum(1 for r in results
                    if len(query_words & set(re.findall(r"[a-z]+",
                    r.title.lower()))) / len(query_words) >= 0.3)
                match_rate = matches / len(results)
            else:
                match_rate = 0
            time.sleep(1.0)
        except Exception as e:
            print(f"    arxiv validation error: {e}")
            hits, match_rate = 0, 0

        h["arxiv_hits"] = hits
        h["title_match_rate"] = round(match_rate, 2)
        h["novel"] = hits < min_hits or match_rate < min_match
        h["terms_grounded"] = terms_ok

        if terms_ok:
            validated.append(h)
            tag = "novel/low prior-art" if h["novel"] else "established"
            print(f"  ✓ [{hits} hits, {match_rate:.0%} match, {tag}] {query[:50]}")
        else:
            print(f"  ✗ [ungrounded source terms] {query[:50]}")

    n_novel = sum(1 for h in validated if h["novel"])
    print(f"\n  {len(validated)} grounded out of {len(hypotheses)} "
          f"({n_novel} novel, {len(validated) - n_novel} established)")
    return validated

# ---------- Selection ----------

def select_top(hypotheses, top_n=8):
    cat_counts = defaultdict(int)
    def score(h):
        hits = h.get("arxiv_hits", 3)
        kp = h.get("keyphrase_score", 0.3)
        match = h.get("title_match_rate", 0.5)
        penalty = max(0, cat_counts[h["section_type"]] - 1) * 0.3
        return (hits * 0.3) + (kp * 0.2) + (match * 0.5) - penalty
    ranked = sorted(hypotheses, key=score, reverse=True)
    selected = []
    for h in ranked:
        if len(selected) >= top_n:
            break
        cat = h["section_type"]
        if cat_counts[cat] >= 3:
            continue
        selected.append(h)
        cat_counts[cat] += 1
    return selected

# ---------- Stage 6: Unified LLM call ----------

def _llm_hypothesize(ps, parsed, candidates, llm, top_n, mode, 
                     weights=None, deliverables=None):
    
    ps_lower = ps.lower()

    # parsed["techniques"]/["tasks"] mix PS-verbatim terms with terms
    # bootstrapped from arXiv titles (Stage 2/3). Both are legitimate
    # "extracted terms" per the RULES below — split them into labeled
    # buckets instead of silently dropping the literature-derived half,
    # and rely on score_terms' ranking (already applied in
    # parse_ps_universal) rather than re-filtering by raw substring match.
    terms = []
    if parsed.get("models"):
        terms.append(f"Named models/tools: {', '.join(parsed['models'][:10])}")
    if parsed.get("techniques"):
        ps_techniques = [t for t in parsed["techniques"] if t.lower() in ps_lower]
        lit_techniques = [t for t in parsed["techniques"] if t.lower() not in ps_lower]
        if ps_techniques:
            terms.append(f"Techniques mentioned in the PS: {', '.join(ps_techniques[:10])}")
        if lit_techniques:
            terms.append(f"Related techniques from the literature (may extend "
                        f"beyond the PS): {', '.join(lit_techniques[:8])}")
    if parsed.get("constraints"):
        terms.append(f"Constraints: {', '.join(parsed['constraints'])}")
    if parsed.get("tasks"):
        ps_tasks = [t for t in parsed["tasks"] if t.lower() in ps_lower]
        lit_tasks = [t for t in parsed["tasks"] if t.lower() not in ps_lower]
        if ps_tasks:
            terms.append(f"Tasks mentioned in the PS: {', '.join(ps_tasks[:10])}")
        if lit_tasks:
            terms.append(f"Related tasks from the literature: {', '.join(lit_tasks[:6])}")
    if parsed.get("modalities"):
        terms.append(f"Input modalities: {', '.join(parsed['modalities'][:8])}")

    arxiv_methods = [t for t in parsed.get("arxiv_vocabulary", [])
                     if any(kw in t.lower() for kw in METHOD_KEYWORDS)]
    if arxiv_methods:
        terms.append(f"Research methods from arXiv: {', '.join(arxiv_methods[:8])}")
    
    term_str = "\n".join(terms) if terms else "No specific terms extracted."
    
    # evaluation weights section
    weight_section = ""
    if weights:
        weight_lines = [f"  {w}% — {c}" for c, w in
                       sorted(weights.items(), key=lambda x: -x[1])]
        weight_section = f"""
EVALUATION CRITERIA (generate more hypotheses for higher-weighted areas):
{chr(10).join(weight_lines)}"""
    
    # deliverables section
    deliv_section = ""
    if deliverables:
        deliv_lines = [f"  • {d}" for d in deliverables[:8]]
        deliv_section = f"""
REQUIRED DELIVERABLES (hypotheses should help produce these):
{chr(10).join(deliv_lines)}"""
    
    if candidates and mode == "polish":
        def novelty_tag(h):
            if "novel" not in h:
                return ""
            return " [novel/low prior-art]" if h["novel"] else " [established in literature]"
        cand_text = "\n".join(
            f"  - {h['hypothesis'][:80]}{novelty_tag(h)} (query: {h['search_queries'][0]})"
            for h in candidates[:8])
        candidate_section = f"""
Auto-generated candidates (use as inspiration but rewrite completely):
{cand_text}"""
    else:
        candidate_section = ""
    
    prompt = f"""You are a research strategist. Read this problem statement carefully
and generate {top_n} research hypotheses as clear STATEMENTS (never questions).

FULL Problem Statement:
{ps[:2000]}

Extracted terms from the PS:
{term_str}
{weight_section}
{deliv_section}
{candidate_section}

RULES:
1. Write each hypothesis as a STATEMENT — NEVER as a question
2. Each hypothesis must name a SPECIFIC research technique and connect it to a PS requirement
3. ONLY use terms from the PS or the extracted terms — do NOT introduce your own terms
4. Each hypothesis needs 2-3 arXiv search queries that would match real paper titles
5. Cover different aspects of the PS — do not repeat the same topic
6. If evaluation weights are provided, allocate more hypotheses to higher-weighted areas
7. section_type must be one of: "sota", "method", "feasibility", "dataset"

Return ONLY JSON: {{"hypotheses": [
  {{"hypothesis": "clear statement naming a technique", "search_queries": ["q1", "q2"],
    "section_type": "sota"}},
]}}"""
    
    try:
        out = llm.invoke(prompt)
        result = safe_parse_json(out.content)
        if result and "hypotheses" in result:
            hypotheses = result["hypotheses"][:top_n]
            valid_types = {"sota", "method", "feasibility", "dataset"}
            for h in hypotheses:
                if h.get("section_type", "").lower() not in valid_types:
                    h["section_type"] = "method"
                else:
                    h["section_type"] = h["section_type"].lower()
            print(f"  LLM {mode}: {len(hypotheses)} hypotheses ✓")
            return hypotheses
    except Exception as e:
        print(f"  LLM {mode} failed: {e}")
    
    if candidates:
        return [{"hypothesis": h["hypothesis"],
                 "search_queries": h["search_queries"],
                 "section_type": h.get("section_type", "method")}
                for h in candidates[:top_n]]
    return []


def orchestrate_final(ps, llm, top_n=5, weights=None, deliverables=None):
    
    print("=" * 60)
    print("STAGE 1-3: Extraction + Bootstrap + Parsing")
    print("=" * 60, "\n")
    
    keyphrases = extract_keyphrases(ps)
    for phrase, score in keyphrases:
        print(f"  {score:.3f}  {phrase}")
    
    domain_anchor = extract_domain_anchor(ps)
    print(f"\n  Domain anchor: '{domain_anchor}'")
    
    arxiv_vocab, titles = bootstrap_vocabulary(keyphrases)
    print(f"\n  arXiv bootstrap: {len(titles)} titles → {len(arxiv_vocab)} terms")
    for term, count in arxiv_vocab[:5]:
        print(f"    [{count}x] {term}")
    
    parsed = parse_ps_universal(ps, arxiv_vocab, keyphrases)
    for k, v in parsed.items():
        if v:
            print(f"  {k}: {v[:8]}{'...' if len(v) > 8 else ''}")
    
    if weights:
        print(f"\n  Evaluation weights:")
        for c, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"    {w}% — {c}")
    
    print(f"\n{'=' * 60}")
    print("STAGE 4-5: Combinatorial + arXiv Validation")
    print("=" * 60, "\n")
    
    raw = generate_combinatorial(parsed, keyphrases, domain_anchor, ps)
    print(f"  {len(raw)} raw hypotheses")
    deduped = deduplicate(raw)
    print(f"  {len(deduped)} after dedup")
    deduped = cap_raw_hypotheses(deduped)
    print(f"  {len(deduped)} after capping (bounds load on arXiv validation)")
    deduped = check_coverage(deduped, parsed)
    print(f"  {len(deduped)} after coverage check")
    
    validated = validate_arxiv(deduped)
    
    extraction_richness = (
        len(parsed.get("models", [])) +
        len(parsed.get("techniques", [])) +
        len(parsed.get("tasks", []))
    )
    
    print(f"\n  Extraction richness: {extraction_richness} terms")
    print(f"  Validated hypotheses: {len(validated)}")
    
    if len(validated) >= top_n:
        path = "polish"
        print(f"\n{'=' * 60}")
        print(f"STAGE 6: LLM Polish — {len(validated)} validated (1 call)")
        print("=" * 60, "\n")
        candidates = select_top(validated, top_n=max(top_n + 3, 8))
        hypotheses = _llm_hypothesize(ps, parsed, candidates, llm, top_n,
                                      "polish", weights=weights,
                                      deliverables=deliverables)
    
    elif extraction_richness >= 8:
        path = "polish-raw"
        print(f"\n{'=' * 60}")
        print(f"STAGE 6: LLM Polish on raw — rich but niche (1 call)")
        print("=" * 60, "\n")
        candidates = select_top(deduped, top_n=max(top_n + 3, 8))
        hypotheses = _llm_hypothesize(ps, parsed, candidates, llm, top_n,
                                      "polish", weights=weights,
                                      deliverables=deliverables)
    
    else:
        path = "generate"
        print(f"\n{'=' * 60}")
        print(f"STAGE 6: LLM Generate — sparse extraction (1 call)")
        print("=" * 60, "\n")
        hypotheses = _llm_hypothesize(ps, parsed, None, llm, top_n,
                                      "generate", weights=weights,
                                      deliverables=deliverables)
    
    for i, h in enumerate(hypotheses):
        print(f"\nH{i+1} [{h.get('section_type', '?')}]: {h.get('hypothesis', '')}")
        for q in h.get("search_queries", []):
            print(f"    → {q}")
    
    print(f"\n{'=' * 60}")
    print(f"  DONE: {len(hypotheses)} hypotheses, 1 LLM call")
    print(f"  Path: {path}")
    print(f"  Domain: '{domain_anchor}'")
    print("=" * 60)
    
    return hypotheses