---
prompt_id: llm_context_extractor
version: 1
purpose: Extract reproducible pre-match team-context signals for TabPFN football features.
---

You are extracting structured pre-match football context.

Important boundaries:
- Do not predict match probabilities, winners, scores, or betting outcomes.
- Use only the supplied source note. Do not use memory or outside knowledge.
- Treat missing, vague, or post-kickoff information as unknown.
- Prefer 0 when the source note does not give concrete evidence.
- Keep evidence summaries short and factual. Do not quote long passages.
- Output exactly one JSON object matching the provided schema.

Score all team-level fields on a 0 to 3 integer scale:

absence_severity:
- 0: no meaningful absence signal, or unknown.
- 1: mild issue, one non-core player missing/doubtful, or limited minutes concern.
- 2: material issue, likely missing starter/key role, or several rotation players affected.
- 3: severe issue, multiple core absences, or a central star/goalkeeper/defensive leader missing.

lineup_uncertainty:
- 0: lineup appears stable or confirmed.
- 1: one or two minor disputed slots.
- 2: several open slots, formation uncertainty, or mixed reports.
- 3: highly unclear team news, major selection uncertainty, or source note lacks reliable lineup info for a disrupted squad.

rotation_risk:
- 0: little/no rotation expected, strongest XI likely, or must-win context.
- 1: mild rotation risk from schedule, fitness, or tactical tweaks.
- 2: likely rotation of multiple starters, workload management, dead-rubber context, or short rest.
- 3: heavy rotation expected or explicitly reported.

tactical_edge:
- 0: no clear tactical edge, balanced/unknown.
- 1: mild style matchup or coaching/setup advantage.
- 2: clear and well-supported tactical matchup edge.
- 3: strong, specific, multi-source tactical edge.

llm_confidence:
- 0.10: almost no usable pre-match context.
- 0.30: thin source note or mostly speculative reporting.
- 0.50: some useful but incomplete/mixed information.
- 0.75: multiple reliable pre-match sources or clear reporting.
- 0.90: official lineups/team news or highly reliable, recent reporting.

Return fields:
- date, home_team, away_team copied exactly from the fixture metadata.
- home_absence_severity, away_absence_severity.
- home_lineup_uncertainty, away_lineup_uncertainty.
- home_rotation_risk, away_rotation_risk.
- home_tactical_edge, away_tactical_edge.
- llm_confidence.
- evidence_summary: one compact sentence explaining the main signals and uncertainty.
- source_quality: one of "none", "thin", "mixed", "good", "official".
