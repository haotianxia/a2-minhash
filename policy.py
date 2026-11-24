import json
import os
import re
import time

from assignment2.datasets import common
from assignment2.datasets.corpus import norm_answer_value
from minhash import LSH, MinHash, similarity

FLOOR = 0.3
MARGIN = 0.1
META = {"tid", "hid", "seq", "opened", "received", "reported", "filed"}
KINDS = {"scenario": "corpus", "resolution_id": "tickets", "root_cause": "bugs", "category": "docs", "verdict": "citations"}
ANSWERS = {"tickets": ("resolution_id",), "bugs": ("component", "root_cause"), "docs": ("category",),
           "citations": ("verdict",)}
IDENT = re.compile(r"\b(CHG|INC|POL)-\d{3,4}\b|\bREL-\d+\.\d+\b", re.IGNORECASE)
STATUS = [("approved", ("approv", "authoris")), ("rejected", ("reject", "declin")),
          ("request", ("request", "pending")), ("resolved", ("resolv", "closed")),
          ("open", ("ongoing", "under investigation")), ("final", ("effective", "in force")),
          ("draft", ("draft", "for comment"))]


def dataset_name(data_dir):
    return os.path.basename(os.path.normpath(data_dir)).split("-")[0]


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flat(value):
    if isinstance(value, dict):
        return " ".join(flat(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(flat(v) for v in value)
    return str(value)


def label(answer):
    return norm_answer_value({k: v for k, v in answer.items() if k != "duplicate_of"})


def status_class(text):
    text = common.norm_text(text)
    return next((name for name, stems in STATUS if any(stem in text for stem in stems)), "none")


class TaskReuse:
    def __init__(self, data_dir, floor=FLOOR, margin=MARGIN, validate=True, calibrate=True):
        env = os.path.join(data_dir, "env")
        history = load_jsonl(os.path.join(env, "history.jsonl"))
        self.name = next(kind for field, kind in KINDS.items() if field in history[0])
        self.answer_fields = ("answer",) if self.name == "corpus" else ANSWERS[self.name]
        self.venues = {}
        if self.name == "citations":
            with open(os.path.join(env, "db.json"), encoding="utf-8") as handle:
                for venue in json.load(handle)["venues"]:
                    for alias in [venue["name"]] + venue["abbrev"]:
                        self.venues[common.norm_text(alias)] = venue["abbrev"][0]
        self.tau, self.margin, self.validate, self.calibrate = floor, margin, validate, calibrate
        self.mh = MinHash()
        self.lsh = LSH(bands=64, rows=2)
        self.sig, self.key, self.answer, self.by_key = {}, {}, {}, {}
        self.seconds = 0.0
        for record in history:
            self.add(record["hid"], record, self.history_answer(record))

    def history_answer(self, record):
        if self.name == "docs":
            return {"category": record["category"], "duplicate_of": record["hid"]}
        if self.answer_fields[0] in ("answer", "verdict"):
            return dict(record[self.answer_fields[0]])
        return {field: record[field] for field in self.answer_fields}

    def text(self, record):
        return "\n".join(flat(v) for k, v in record.items() if k not in META and k not in self.answer_fields)

    def key_of(self, r):
        if self.name == "tickets":
            return common.norm_text(r["product"]), common.norm_code(r["error_code"]), common.norm_text(r["platform"])
        if self.name == "bugs":
            return (str(r["exception"]).split(".")[-1].lower(), common.norm_frame(r["top_frame"]),
                    common.norm_version(r["version"]))
        if self.name == "docs":
            ident = IDENT.search(r["title"])
            return ident.group(0).upper() if ident else None, status_class(r["title"] + " " + r["body"])
        if self.name == "citations":
            c = r["citation"]
            venue = common.norm_text(c["venue"])
            return (common.norm_doi(c.get("doi", "")), common.norm_title(c["title"]),
                    tuple(common.norm_author(a) for a in c["authors"]), self.venues.get(venue, venue), str(c["year"]))
        return common.norm_code(r["scenario"]), common.norm_text(r["kind"]), common.norm_code(r.get("target") or "")

    def lookup(self, task):
        start = time.perf_counter()
        sig, key = self.mh.text(self.text(task)), self.key_of(task)
        best, labels = None, set()
        for rid in self.lsh.query(sig):
            sim = similarity(sig, self.sig[rid])
            if sim < self.tau or (self.validate and self.key[rid] != key):
                continue
            labels.add(label(self.answer[rid]))
            if best is None or sim > best[1]:
                best = (rid, sim)
        self.seconds += time.perf_counter() - start
        if best is None or len(labels) > 1:
            return None
        return {"match": best[0], "sim": best[1], "tau": self.tau, "answer": self.answer[best[0]]}

    def add(self, rid, record, answer):
        start = time.perf_counter()
        sig, key = self.mh.text(self.text(record)), self.key_of(record)
        if self.calibrate:
            for other in self.by_key.get(key, []):
                if label(self.answer[other]) != label(answer):
                    self.tau = max(self.tau, similarity(sig, self.sig[other]) + self.margin)
        self.sig[rid], self.key[rid], self.answer[rid] = sig, key, answer
        self.by_key.setdefault(key, []).append(rid)
        self.lsh.add(rid, sig)
        self.seconds += time.perf_counter() - start
