import json, itertools
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from groq import Groq

import config

CTX_LIMIT = 8192
MARGIN    = 64


class Experiment2:
    # ─────────────────────────── setup ──────────────────────────
    def __init__(self, *, model_id: str, seed: int):
        self.model_id = model_id
        code_dir      = Path.home() / "UCL" / "FYP" / "code"

        self.data_dir = code_dir / "data"
        prompts_dir   = code_dir / "pre-prompts"

        api_key       = (code_dir / "groq-access-token.txt").read_text().strip()
        self.client   = Groq(api_key=api_key)
        self.rng      = np.random.default_rng(seed)

        with open(self.data_dir / "subreddit-selection.json") as f:
            self.sub_selection = json.load(f)

        self.pp1 = (prompts_dir / "expt1-zero-shot.txt").read_text()
        self.pp2 = (prompts_dir / "expt2-stage-2.txt").read_text()
        self.pp3 = (prompts_dir / "expt2-stage-3.txt").read_text()

        config.output(f"✔  Groq client ready  •  model = {model_id}")

    # ───────────────────────── data helpers ─────────────────────
    def load_df(self, path: Path) -> pd.DataFrame:
        config.output(f"↪  Loading dataframe from {path}")
        df = pd.read_parquet(path)
        df["date_posted"] = pd.to_datetime(df["created_utc"], unit="s")
        config.output(f"   …loaded {len(df):,} rows")
        return df

    def batch_iter(self, seq, size):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    # ───────────────────────── stage-2 helpers ───────────────────
    def _batch_to_prompt(self, batch: pd.DataFrame, sub: str) -> str:
        start, end = batch.iloc[0]["date_posted"], batch.iloc[-1]["date_posted"]
        header = f"All supplied posts will be from r/{sub}. They were posted between {start} and {end}\n"
        body   = []
        for idx, row in batch.iterrows():
            meta = [f"- {c}: {row[c]}" for c in row.index if c not in ("text",)]
            body.append(
                f"[POST {idx+1}]\n- date: {row['date_posted'].strftime('%Y-%m-%d')}\n"
                + "\n".join(meta)
                + f"\n- post contents:\n'{row['text']}'\n"
            )
        return header + "\n".join(body)

    # ───────────────────────────── stage-2 ───────────────────────
    @config.debug_function
    def stage2(self,
               df: pd.DataFrame,
               batch_size: int      = 20,
               max_new_tokens: int  = 200) -> Dict[Tuple[str,str,str], List[Dict]]:

        config.output("▶  Stage-2  (mini-reports)")
        demographics = list(itertools.product(
            df.subreddit.unique(),
            df.predicted_age.unique(),
            df.predicted_gender.unique())
        )
        config.output(f"   {len(demographics)} demographic slices to process")

        results: Dict[Tuple[str,str,str], List[Dict]] = {}
        processed = 0
        for sub, age, gender in demographics:
            slice_df = df[(df.subreddit==sub)&(df.predicted_age==age)&(df.predicted_gender==gender)]
            if slice_df.empty:
                continue

            buckets: List[Dict] = []
            for batch in self.batch_iter(slice_df.sort_values("date_posted"), batch_size):
                processed += 1
                if processed % 50 == 0:
                    config.debug(f"      …processed {processed} batches so far")

                prompt = self._batch_to_prompt(batch, sub)
                rsp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self.pp2},
                        {"role": "user",   "content": prompt}],
                    temperature=0.2,
                    max_completion_tokens=max_new_tokens
                )
                try:
                    buckets.extend(json.loads(rsp.choices[0].message.content))
                except json.JSONDecodeError:
                    config.debug("         ⚠ could not parse JSON → skipping batch")

            results[(sub, age, gender)] = (
                buckets if buckets else
                [{"summary":"no trends detected","evidence":[],"reasoning":""}]
            )

        config.output(f"   Stage-2 complete  •  {processed} chat calls")
        return results

    # ───────────────────────────── stage-3 ───────────────────────
    def _rep_to_str(self, rep: Dict, idx: int) -> str:
        s = f"[REPORT {idx+1}]\n - trend summary: {rep['summary']}"
        if rep.get("reasoning"):
            s += f"\n - reasoning: {rep['reasoning']}"
        return s + f"\n - evidence: {rep['evidence']}\n"

    @config.debug_function
    def stage3(self,
               group: Tuple[str, str],
               reports: List[Dict],
               max_new_tokens: int = 400) -> Tuple[str, int]:

        age, gender = group
        header = f"METADATA\n - age range: {age}\n - gender: {gender}\n\n"
        error = 0
        layer = 0
        current = reports[:]

        len_current = len(current)
        len_prev = 2 * len(current)

        while 0 < len_current < len_prev:
            layer += 1
            config.debug(f"      Layer {layer}: compressing {len(current)} reports")

            # chunk until prompt fits
            chunks, chunk, tok_est = [], [], len(header.split())
            for rep in current:
                rep_txt  = self._rep_to_str(rep, 0)
                rep_toks = len(rep_txt.split())
                if tok_est + rep_toks + MARGIN > CTX_LIMIT - max_new_tokens:
                    chunks.append(chunk); chunk=[]; tok_est=len(header.split())
                chunk.append(rep); tok_est += rep_toks
            if chunk:
                chunks.append(chunk)

            new_reports = []
            for ch in chunks:
                prompt = header + "".join(self._rep_to_str(r, i) for i, r in enumerate(ch))
                rsp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self.pp3},
                        {"role": "user",   "content": prompt}],
                    temperature=0.2,
                    max_completion_tokens=max_new_tokens
                )
                try:
                    new_reports.extend(json.loads(rsp.choices[0].message.content))
                except json.JSONDecodeError:
                    error += 1
                    config.debug("         ⚠ JSON parse error on compression chunk")

            if not new_reports:
                return json.dumps({"summary":"no data","evidence":[],"reasoning":""}, indent=2), error
            current = new_reports

            len_prev = len_current
            len_current = len(current)

        config.debug(f"      Final layer produced 1 report")
        return json.dumps(current[0], ensure_ascii=False, indent=2), error

    # ─────────────────────────── master run ──────────────────────
    def run(self,
            df: pd.DataFrame,
            chunk_size: int      = 20,
            stage2_tokens: int   = 200,
            stage3_tokens: int   = 400):

        # -------- Stage-2
        trends = self.stage2(df, batch_size=chunk_size,
                             max_new_tokens=stage2_tokens)

        # -------- Stage-3
        config.output("▶  Stage-3  (hierarchical compression)")
        final_inputs: Dict[Tuple[str, str], List[Dict]] = {}
        for (sub, age, gender), reps in trends.items():
            final_inputs.setdefault((age, gender), []).extend(reps)

        summaries = {}
        for demog, reps in final_inputs.items():
            config.output(f"   Compressing demographic {demog} with {len(reps)} mini-reports")
            summaries[demog], err = self.stage3(demog, reps,
                                                max_new_tokens=stage3_tokens)
            if err:
                config.debug(f"      → {err} parse errors")

        config.output("✓  Experiment finished")
        return trends, summaries


# ─────────────────────────────── CLI ───────────────────────────
if __name__ == "__main__":
    experiment = Experiment2(model_id="llama-3.3-70b-versatile", seed=42)

    for group in ["relevant", "irrelevant"]:
        for period in ["current"]:
            df_path = (
                Path.home() / "UCL" / "FYP" / "results"
                / "overall-results-historical-and-current-data"
                / f"expt2-{group}-subreddits-{period}-data" / "mistral"
                / "demographic-inferences.parquet"
            )
            df = experiment.load_df(df_path)

            # sanity-check: Filter 2025 data
            df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")

            trends, summaries = experiment.run(df)

            with open(f"{group}-{period}-overall-stage-2-results.json", "w") as f:
                json.dump({str(k): v for k, v in trends.items()}, f, indent=2)

            with open(f"{group}-{period}-overall-stage-3-results.json", "w") as f:
                json.dump({str(k): v for k, v in summaries.items()}, f, indent=2)

            config.output("✓  Results written to disk")
