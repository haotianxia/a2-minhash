import difflib
import os
import re
import time

from spartacus.tools import Tool, core_tools

from minhash import LSH, MinHash, similarity

THRESHOLD = 0.5
MIN_CHARS = 400
MAX_RATIO = 0.75
HEADER_CHARS = 160
WRAPPED = ("read_file", "grep", "list_files", "bash")
NUMBERED = re.compile(r"\d+\t")
SAME = "[reuse] identical to %s shown earlier (%d lines)."
DIFF = ("[reuse] same as %s shown earlier except for the lines in this unified diff "
        "(- earlier, + here); every other line is unchanged:\n%s")


def call_label(name, args):
    inner = ", ".join("%s=%s" % (k, v) for k, v in args.items())
    return "%s(%s)" % (name, inner if len(inner) <= 100 else inner[:100] + "...")


class ContentReuse:
    def __init__(self, threshold=THRESHOLD, min_chars=MIN_CHARS, max_ratio=MAX_RATIO):
        self.threshold, self.min_chars, self.max_ratio = threshold, min_chars, max_ratio
        self.mh = MinHash()
        self.lsh = LSH(bands=32, rows=4)
        self.refs, self.exact, self.names = [], {}, set()
        self.stats = {"outputs": 0, "folded": 0, "identical": 0, "chars_in": 0, "chars_out": 0, "seconds": 0.0}

    def tools(self, root):
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if base != root:
                for name in files:
                    self.names.update((os.path.relpath(os.path.join(base, name), root), name))
        return [Tool(t.name, t.spec, self._runner(t)) for t in core_tools(root) if t.name in WRAPPED]

    def _runner(self, tool):
        def run(**args):
            return self.fold(call_label(tool.name, args), str(tool.run(**args)))
        return run

    def fold(self, label, text):
        start = time.perf_counter()
        out = self._fold(label, text) if len(text) >= self.min_chars else text
        self.stats["outputs"] += 1
        self.stats["chars_in"] += len(text)
        self.stats["chars_out"] += len(out)
        self.stats["seconds"] += time.perf_counter() - start
        return out

    def _fold(self, label, text):
        lines = text.splitlines()
        if all(NUMBERED.match(line) for line in lines):
            parts = [(None, [line.split("\t", 1)[1] for line in lines])]
        else:
            parts = self._split(lines)
        out, changed = [], False
        for header, body in parts:
            name = label if header is None else "the part of %s headed %r" % (label, header)
            folded = self._fold_part(name, body)
            changed |= folded is not None
            out += ([header] if header is not None else []) + [folded if folded is not None else "\n".join(body)]
        return "\n".join(out) if changed else text

    def _split(self, lines):
        parts, header, body = [], None, []
        for line in lines:
            if len(line) <= HEADER_CHARS and any(name in line for name in self.names):
                if header is not None or body:
                    parts.append((header, body))
                header, body = line, []
            else:
                body.append(line)
        parts.append((header, body))
        if all(len("\n".join(b)) < self.min_chars for _, b in parts):
            return [(None, lines)]
        return parts

    def _fold_part(self, name, body):
        text = "\n".join(body)
        if len(text) < self.min_chars:
            return None
        if text in self.exact:
            self.stats["identical"] += 1
            self.stats["folded"] += 1
            return SAME % (self.exact[text], len(body))
        sig = self.mh.text(text, mask=True)
        scored = sorted(((similarity(sig, self.refs[i][2]), i) for i in self.lsh.query(sig)), reverse=True)
        folds = [self._diff(self.refs[i], body) for sim, i in scored[:3] if sim >= self.threshold]
        best = min(folds, key=len, default=None)
        if best is not None and len(best) <= self.max_ratio * len(text):
            self.stats["folded"] += 1
            return best
        self.exact[text] = name
        self.lsh.add(len(self.refs), sig)
        self.refs.append((name, body, sig))
        return None

    def _diff(self, ref, body):
        return DIFF % (ref[0], "\n".join(list(difflib.unified_diff(ref[1], body, lineterm="", n=0))[2:]))
