"""Generate the full 109-locus Tool 8 appendix table from the committed
results/riskflag_panel.json - plain-LaTeX tabular blocks (no longtable),
36 rows per block with repeated headers. Numbers are machine-extracted,
never hand-transcribed."""
import json

p = json.load(open("results/riskflag_panel.json"))
recs = sorted((r for r in p["records"] if r.get("risk_mean") is not None),
              key=lambda r: -r["risk_mean"])
assert len(recs) == 109

def esc(s):
    return str(s).replace("_", "\\_")

CHUNK = 36
out = [r"""\subsection{Tool 8 panel: full per-locus table}
\label{app:panel-table}
All 109 scored loci of the Tool 8 panel, ranked by mean locus risk,
machine-extracted from \texttt{results/riskflag\_panel.json} by
\texttt{scripts/make\_panel\_table.py}. Columns: accession, organism
(H = human, M = mouse), NGG cut sites scored in the $\pm250$ bp window,
flag counts (L/M/H = LOW/MODERATE/HIGH), mean and max site risk. The six
requested loci that failed accession resolution or sequence fetch (KRAS,
Rosa26, Alb, Hbb, TRAC, TRBC1) are recorded in the JSON and excluded
here; no locus is silently dropped."""]
for i in range(0, len(recs), CHUNK):
    block = recs[i:i + CHUNK]
    cont = " (continued)" if i else ""
    out.append("\\begin{table}[h]\n\\centering\\footnotesize")
    out.append(f"\\caption{{Tool 8 panel, loci {i+1}--{i+len(block)} of 109 by mean risk{cont}.}}")
    out.append("\\begin{tabular}{rllcrrrcc}")
    out.append("\\hline")
    out.append("rank & gene & accession & org & sites & L/M/H & mean & max \\\\")
    out.append("\\hline")
    for j, r in enumerate(block, i + 1):
        fc = r["flag_counts"]
        out.append(f"{j} & {esc(r['gene'])} & {esc(r['accession'])} & "
                   f"{r['organism'][0].upper()} & {r['n_cutsites_scored']} & "
                   f"{fc.get('LOW',0)}/{fc.get('MODERATE',0)}/{fc.get('HIGH',0)} & "
                   f"{r['risk_mean']:.3f} & {r['risk_max']:.3f} \\\\")
    out.append("\\hline\n\\end{tabular}\n\\end{table}")
open("paper/appendix_panel.tex", "w").write("\n".join(out) + "\n")
print("appendix_panel.tex written:", len(recs), "rows in",
      (len(recs) + CHUNK - 1) // CHUNK, "blocks")
