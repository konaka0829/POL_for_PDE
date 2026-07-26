from __future__ import annotations
from pathlib import Path
import json


def create_e1_plots(
    out: Path,
    tables: dict,
    *,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 160,
) -> list[dict]:
    import matplotlib.pyplot as plt
    made = []
    modes, selected, diag, noise = (tables[k] for k in ("mode_comparison","selected_results","readout_diagnostics","noise_summary"))
    for regime in ("stable", "unstable"):
        rows = [r for r in modes if r["regime"] == regime and r["q"] == max(x["q"] for x in modes if x["regime"] == regime)]
        fig, ax = plt.subplots(); ax.plot([r["coefficient_index"] for r in rows],[r["theoretical_multiplier"] for r in rows],label="theory"); ax.plot([r["coefficient_index"] for r in rows],[r["learned_effective_diagonal"] for r in rows],"o",ms=3,label="learned"); ax.set_yscale("log"); ax.legend(); ax.set_xlabel("coefficient index"); ax.set_ylabel("multiplier")
        for fmt in formats:
            name=f"multiplier_{regime}.{fmt}"; fig.savefig(out/name,dpi=dpi,bbox_inches="tight"); made.append({"relative_path":name,"status":"created","source":"mode_comparison.csv","format":fmt})
        plt.close(fig)
    fig, ax=plt.subplots()
    for case in sorted(set(r["case_name"] for r in selected)):
        rr=[r for r in selected if r["case_name"]==case]; ax.plot([r["q"] for r in rr],[r["full_reference_field_relative_l2_mean"] for r in rr],"o-",label=case); ax.plot([r["q"] for r in rr],[r["output_representation_floor_mean"] for r in rr],"--",label=f"{case} floor")
    ax.set_yscale("log"); ax.legend(); ax.set_xlabel("q"); ax.set_ylabel("relative L2")
    for fmt in formats:
        name=f"bandwidth_error.{fmt}"; fig.savefig(out/name,dpi=dpi,bbox_inches="tight"); made.append({"relative_path":name,"status":"created","source":"selected_results.csv","format":fmt})
    plt.close(fig)
    fig, ax=plt.subplots()
    for case in sorted(set(r["case_name"] for r in diag)):
        rr=[r for r in diag if r["case_name"]==case]; ax.plot([r["q"] for r in rr],[r["learned_operator_norm"] for r in rr],"o-",label=f"{case} learned"); ax.plot([r["q"] for r in rr],[r["ideal_operator_norm"] for r in rr],"--",label=f"{case} ideal")
    ax.set_yscale("log"); ax.legend(); ax.set_xlabel("q"); ax.set_ylabel("operator norm")
    for fmt in formats:
        name=f"readout_operator_norm.{fmt}"; fig.savefig(out/name,dpi=dpi,bbox_inches="tight"); made.append({"relative_path":name,"status":"created","source":"readout_diagnostics.csv","format":fmt})
    plt.close(fig)
    fig, ax=plt.subplots()
    qmax=max(r["q"] for r in noise)
    for case in sorted(set(r["case_name"] for r in noise)):
        rr=[r for r in noise if r["case_name"]==case and r["q"]==qmax]; ax.plot([r["noise_level"] for r in rr],[r["output_perturbation_rms_mean"] for r in rr],"o-",label=case)
    ax.legend(); ax.set_xlabel("noise level"); ax.set_ylabel("output perturbation RMS")
    for fmt in formats:
        name=f"noise_sensitivity.{fmt}"; fig.savefig(out/name,dpi=dpi,bbox_inches="tight"); made.append({"relative_path":name,"status":"created","source":"noise_summary.csv","format":fmt})
    plt.close(fig)
    return made
