"""Audit report collection and rendering for LLMSmartAuditTool (SmartContractTA).

Why this module exists
----------------------
The per-detector phases in ``chatdev/phase.py`` each overwrite
``chat_env.env_dict['ideas']`` with their own ``<INFO>`` conclusion, so only the
**last** detector's result survives there; nothing aggregates the individual
findings. Without this collector a 41-detector run reports the verdict of
detector 41 and silently discards the other 40.

The collector hooks the single ``Phase.execute`` chokepoint, captures every
detector's conclusion as it happens, and renders two artefacts per run: a
Markdown report for a human reader and a JSON document with a stable schema
(:meth:`AuditReportCollector.render_data`) that the SmartBugs parser consumes.

``DETECTORS`` is the machine-readable source of truth for which categories
SmartContractTA actually probes --- one entry per phase in
``CompanyConfig/SmartContractTA/ChatChainConfig.json``. ``docs/coverage.md`` is
the human-facing companion and is kept consistent with this table.

Reporting coverage, not just findings
-------------------------------------
Both artefacts carry ``categories_checked`` alongside the findings. An empty
finding list means "clean" only if the checked count is the full 41; after a
backend rejection it is 0, and the run must not be readable as a clean audit.
The SmartBugs parser turns ``checked == 0`` into an error for exactly that
reason.
"""

import json
import os
from collections import OrderedDict

# phase_name -> (human title, classification, doc-assigned severity)
# classification: SWC/DASP where there is a clean mapping; otherwise an honest "logic/DeFi"
# label. Severity is a DOC-ASSIGNED default (the tool's prompts do not emit severities).
DETECTORS = OrderedDict([
    ("ArithmeticDetector", ("Integer Overflow / Underflow", "SWC-101, DASP-3", "High")),
    ("ReentrancyDetector", ("Reentrancy", "SWC-107, DASP-1", "High")),
    ("UncheckedSendDetector", ("Unchecked External Call (send/transfer/ERC20 return)", "SWC-104, DASP-4", "Medium")),
    ("UncheckedLowLevelCallDetector", ("Unchecked Low-level Call (call/delegatecall/send)", "SWC-104, DASP-4", "Medium")),
    ("TODDetector", ("Transaction Order Dependence (front-running)", "SWC-114, DASP-7", "Medium")),
    ("TimeStampManipulationDetector", ("Timestamp Dependence", "SWC-116, DASP-8", "Low")),
    ("PredictableRandDetector", ("Predictable / Weak Randomness", "SWC-120, DASP-6", "Medium")),
    ("TXRelianceDetector", ("Authorization via tx.origin", "SWC-115, DASP-2", "Medium")),
    ("SuicideDetector", ("Unprotected SELFDESTRUCT", "SWC-106, DASP-2", "High")),
    ("GasLimitDetector", ("Gas-Limit / Unbounded Operation", "SWC-128, DASP-5", "Medium")),
    ("PriceManipulationDetector", ("Price Manipulation (centralized pricing)", "logic/DeFi (no standard SWC)", "High")),
    ("DataCorruptionDetector", ("Assumption of Fixed-Length Array", "logic (no standard SWC)", "Medium")),
    ("WithdrawalFunctionDetector", ("Missing Withdrawal Function (locked funds)", "logic (locked funds)", "Medium")),
    ("LackAuthorizationDetector", ("Missing Access Control / Authorization", "SWC-105 (approx.), DASP-2", "High")),
    ("DataInconsistencyDetector", ("Storage/Memory Data Inconsistency", "logic (no standard SWC)", "Medium")),
    ("HashCollisionDetector", ("Hash Collision via abi.encodePacked", "SWC-133", "Medium")),
    ("UninitializedReturnVariableDetector", ("Uninitialized Return Variable", "logic (SWC-109 family)", "Medium")),
    ("MisdeclaredConstructorDetector", ("Incorrect Constructor Name (legacy syntax)", "SWC-118", "High")),
    ("MissingOnlyOwnerDetector", ("Missing onlyOwner Modifier", "DASP-2 (no specific SWC)", "High")),
    ("MisuseMsgValueDetector", ("Misuse of msg.value in a Loop", "logic (no standard SWC)", "High")),
    ("PrecisionLossDetector", ("Precision Loss / Integer-Division Truncation", "logic (arithmetic)", "Medium")),
    ("RedundantConditionalDetector", ("Redundant Conditional Check", "code-quality (no standard SWC)", "Low")),
    ("OracleDependencyDetector", ("Single External Oracle Dependency", "logic/DeFi (no standard SWC)", "Medium")),
    ("OwnershipHijackingDetector", ("Unprotected Ownership Change", "DASP-2 (no specific SWC)", "High")),
    ("CentralizationRiskDetector", ("Centralization Risk", "DASP-2 / governance (no standard SWC)", "Medium")),
    ("FundingCalculationDetector", ("Funding-Rate Calculation Precision", "logic/DeFi (no standard SWC)", "Medium")),
    ("FlashLoanDetector", ("Flash-Loan Fee Manipulation", "logic/DeFi (no standard SWC)", "High")),
    ("MappingGetterDetector", ("Misuse of Mapping Getter", "code-quality (no standard SWC)", "Low")),
    ("GetterFunctionDetector", ("Unimplemented Interface Function", "logic/compile (no standard SWC)", "Low")),
    ("UnnecessaryComparisonDetector", ("Unnecessary Boolean Comparison", "code-quality (no standard SWC)", "Low")),
    ("InconsistentInitializationDetector", ("Inconsistent State Initialization Order", "logic (no standard SWC)", "Medium")),
    ("SourceSwappingDetector", ("Yield-Source Swap Token Mismatch", "logic/DeFi (no standard SWC)", "Medium")),
    ("SignatureVerificationDetector", ("Incorrect Signature / Signer Verification", "SWC-122 (approx.)", "High")),
    ("OrderInitializationDetector", ("Inheritance Initialization Order / Diamond", "logic (SWC-125 family)", "Medium")),
    ("ImpracticalityMatchDetector", ("Strict Balance Equality (this.balance ==)", "SWC-132", "Low")),
    ("InconsistentTokensDetector", ("Inconsistent Base Tokens on Migration", "logic/DeFi (no standard SWC)", "Medium")),
    ("PartialWithdrawalsDetector", ("Unhandled Partial Withdrawals on Migration", "logic/DeFi (no standard SWC)", "Medium")),
    ("FallbackFunctionDetector", ("Missing Fallback Function (rejected/locked ether)", "logic (no standard SWC)", "Low")),
    ("UnlimitedTokenDetector", ("Unlimited Token Approval", "logic/DeFi (no standard SWC)", "Medium")),
    ("InputValidationDetector", ("Lack of Input Validation", "SWC-123 (approx.)", "Medium")),
    ("DoSDetector", ("Denial of Service", "SWC-113 / SWC-128, DASP-5", "Medium")),
])

# Phases whose conclusion is a consolidated narrative summary (not a single category).
SUMMARY_PHASES = ("BugsSummary", "TestBugsSummary")

# Where to drop a running snapshot of the report, and the context it needs.
# Set by the SmartBugs entrypoint; unset outside SmartBugs, where the tool keeps
# its original behaviour of writing only once, at the end.
ENV_PARTIAL = "SB_PARTIAL_REPORT"
ENV_CONTRACT = "SB_CONTRACT_NAME"


def _is_identified(conclusion: str) -> bool:
    """Heuristic: a detector reports a finding when its <INFO> conclusion signals
    '... Identified.' / '... Identify.' and is not a templated negative ('NO ...').

    The detector prompts standardize on '<INFO> X Identified.' for positives and
    '<INFO> NO X.' for negatives, so this is reliable for prompt-compliant output."""
    text = conclusion.split("<INFO>")[-1].strip().lower()
    if not text:
        return False
    if text.startswith("no "):
        return False
    return "identif" in text  # matches 'identified' and the one 'identify' typo


class AuditReportCollector:
    """Accumulates one analysis run's detector conclusions and renders them.

    Kept separate from the module-level singleton below so it can be
    instantiated directly in a test without touching global state.
    """

    def __init__(self):
        """Start out empty; :meth:`reset` defines the fields in one place."""
        self.reset()

    def reset(self):
        """Discard everything collected so far, ready for the next contract.

        Called at the start of every run (``chat_chain.py``) because the
        singleton outlives a single analysis when several run in one process.
        """
        self._findings = []          # list of dicts for detectors that fired
        self._checked = []           # all detector phase_names that ran (coverage proof)
        self._summary = ""           # consolidated BugsSummary/TestBugsSummary text

    def record(self, phase_name, conclusion):
        """Take note of one finished phase.

        A phase counts as *checked* as soon as it ran, whether or not it found
        anything --- that is what makes the coverage figure meaningful. Phases
        that are not detectors at all are either the consolidated summary or
        ignored outright, so unrelated chat phases cannot inflate the count.
        """
        if conclusion is None:
            return
        if phase_name in DETECTORS:
            self._checked.append(phase_name)
            if _is_identified(conclusion):
                title, classification, severity = DETECTORS[phase_name]
                self._findings.append({
                    "phase": phase_name,
                    "title": title,
                    "classification": classification,
                    "severity": severity,
                    "detail": conclusion.split("<INFO>")[-1].strip(),
                })
            # A detector finished: put the new state on disk before the next one
            # starts, so a hard stop costs at most the detector in flight.
            self.flush_partial()
        elif phase_name in SUMMARY_PHASES:
            self._summary = (conclusion or "").split("<INFO>")[-1].strip()

    @property
    def findings(self):
        """The detectors that fired, as a copy so callers cannot mutate state."""
        return list(self._findings)

    @property
    def checked(self):
        """Phase names of every detector that ran, as a copy."""
        return list(self._checked)

    def render_markdown(self, contract_name, provider, model, timestamp, config="SmartContractTA"):
        """The human-readable report: header, findings, summary, coverage list.

        Findings are ordered by severity so the worst appears first. The
        coverage list at the end names *every* known detector with a checkbox,
        not only the ones that ran --- a category missing from a run is then
        visible instead of merely absent.
        """
        severity_rank = {"High": 0, "Medium": 1, "Low": 2}
        findings = sorted(self._findings, key=lambda f: severity_rank.get(f["severity"], 3))

        lines = []
        lines.append("# LLMSmartAudit Report — {}".format(contract_name))
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append("| Contract | `{}` |".format(contract_name))
        lines.append("| Analysis mode (config) | {} |".format(config))
        lines.append("| Provider | {} |".format(provider))
        lines.append("| Model | {} |".format(model))
        lines.append("| Timestamp | {} |".format(timestamp))
        lines.append("| Categories checked | {} / {} |".format(len(self._checked), len(DETECTORS)))
        lines.append("| Findings | {} |".format(len(findings)))
        lines.append("")
        lines.append("> Severities are documentation-assigned defaults per category "
                     "(see `docs/coverage.md`); the model itself does not emit a severity. "
                     "Results are non-deterministic and may vary between runs and providers.")
        lines.append("")

        lines.append("## Findings")
        lines.append("")
        if not findings:
            lines.append("_No vulnerabilities were reported by the detectors for this contract._")
        else:
            lines.append("| # | Severity | Category | Classification |")
            lines.append("|---|---|---|---|")
            for i, f in enumerate(findings, 1):
                lines.append("| {} | {} | {} | {} |".format(
                    i, f["severity"], f["title"], f["classification"]))
            lines.append("")
            for i, f in enumerate(findings, 1):
                lines.append("### {}. {} ({})".format(i, f["title"], f["severity"]))
                lines.append("")
                lines.append("- **Category (detector):** {}".format(f["phase"]))
                lines.append("- **Classification:** {}".format(f["classification"]))
                lines.append("")
                lines.append("**Model output:**")
                lines.append("")
                lines.append("```")
                lines.append(f["detail"])
                lines.append("```")
                lines.append("")

        if self._summary:
            lines.append("## Consolidated summary (model)")
            lines.append("")
            lines.append(self._summary)
            lines.append("")

        lines.append("## Coverage (categories checked this run)")
        lines.append("")
        lines.append("<details><summary>{} of {} SmartContractTA categories</summary>".format(
            len(self._checked), len(DETECTORS)))
        lines.append("")
        for phase in DETECTORS:
            mark = "x" if phase in self._checked else " "
            hit = " — **finding**" if any(f["phase"] == phase for f in self._findings) else ""
            lines.append("- [{}] {}{}".format(mark, DETECTORS[phase][0], hit))
        lines.append("")
        lines.append("</details>")
        lines.append("")
        return "\n".join(lines)

    def write_markdown(self, out_path, contract_name, provider, model, timestamp,
                       config="SmartContractTA"):
        """Render the Markdown report to *out_path*; returns the path written."""
        content = self.render_markdown(contract_name, provider, model, timestamp, config)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        return out_path

    def render_data(self, contract_name, provider, model, timestamp, config="SmartContractTA"):
        """Machine-readable raw data backing the Markdown report (consumed by the SmartBugs
        parser). Stable schema: meta + findings[] + summary + checked[]."""
        return {
            "tool": "LLMSmartAudit",
            "config": config,
            "contract": contract_name,
            "provider": provider,
            "model": model,
            "timestamp": timestamp,
            "categories_total": len(DETECTORS),
            "categories_checked": len(self._checked),
            "checked": list(self._checked),
            "summary": self._summary,
            "findings": [
                {
                    "name": f["title"],
                    "detector": f["phase"],
                    "classification": f["classification"],
                    "severity": f["severity"],
                    "description": f["detail"],
                }
                for f in self._findings
            ],
        }

    def write_json(self, out_path, contract_name, provider, model, timestamp,
                   config="SmartContractTA"):
        """Write the machine-readable report to *out_path*; returns the path.

        This is the file the container hands to SmartBugs as ``/output.json``.
        """
        data = self.render_data(contract_name, provider, model, timestamp, config)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return out_path

    def flush_partial(self):
        """Write a snapshot of everything collected so far, if configured.

        Called after every detector phase. The tool aggregates in memory and
        writes only once at the very end, so without this a run stopped by a
        timeout loses every detector it had already finished.

        The context (contract, provider, model) is read from the environment
        because record() has none: it sees a phase name and a conclusion, and
        nothing else. The entrypoint knows all three and exports them.

        The write goes to a temporary file next to the target and is moved into
        place with os.replace(), which is atomic on POSIX and Windows alike.
        Writing in place would truncate the previous snapshot before the new one
        is complete --- and a kill landing in that window would destroy exactly
        the partial result this method exists to preserve.

        Failure is swallowed on purpose. This runs inside a paid analysis; an
        unwritable snapshot path must cost the snapshot, never the run.
        """
        out_path = os.environ.get(ENV_PARTIAL)
        if not out_path:
            return None
        tmp_path = out_path + ".tmp"
        try:
            self.write_json(
                out_path=tmp_path,
                contract_name=os.environ.get(ENV_CONTRACT, "unknown"),
                provider=os.environ.get("LLM_PROVIDER", "unknown"),
                model=os.environ.get("LLM_MODEL_ID")
                      or os.environ.get("LLM_MODEL", "unknown"),
                timestamp="partial",
            )
            os.replace(tmp_path, out_path)
            return out_path
        except Exception as e:  # noqa: BLE001 - see docstring
            print(f"audit_report: partial snapshot failed: {e}")
            return None


# Module-level singleton: one analysis run == one process, so a single collector is safe.
_collector = AuditReportCollector()


def reset():
    """Clear the singleton --- call once at the start of every analysis run."""
    _collector.reset()


def record(phase_name, conclusion):
    """Hand one finished phase to the singleton (called from ``Phase.execute``)."""
    _collector.record(phase_name, conclusion)


def write_markdown(out_path, contract_name, provider, model, timestamp, config="SmartContractTA"):
    """Module-level shortcut for :meth:`AuditReportCollector.write_markdown`."""
    return _collector.write_markdown(out_path, contract_name, provider, model, timestamp, config)


def write_json(out_path, contract_name, provider, model, timestamp, config="SmartContractTA"):
    """Module-level shortcut for :meth:`AuditReportCollector.write_json`."""
    return _collector.write_json(out_path, contract_name, provider, model, timestamp, config)


def get_collector():
    """The singleton itself, for callers that need the collected data directly."""
    return _collector
