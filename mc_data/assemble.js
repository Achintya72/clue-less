#!/usr/bin/env node
// Assemble raw per-puzzle JSON into a clean benchmark dataset.
const fs = require("fs");
const path = require("path");

const RAW = path.join(__dirname, "raw_json");
const files = fs.readdirSync(RAW).filter(f => f.endsWith(".json")).sort();

// par endpoint gives rounded community-average par for every date; use to
// backfill averagePar on older puzzles that lack a parDetails block.
const avgParByDate = {};
for (const line of fs.readFileSync(path.join(__dirname, "dates_par.txt"), "utf8").trim().split("\n")) {
  const [d, p] = line.split(/\s+/);
  if (d) avgParByDate[d] = Number(p);
}

const puzzles = [];
const problems = [];

for (const f of files) {
  let d;
  try { d = JSON.parse(fs.readFileSync(path.join(RAW, f), "utf8")); }
  catch (e) { problems.push(`${f}: parse error`); continue; }
  if (!d.answer || !Array.isArray(d.clue)) { problems.push(`${f}: missing answer/clue`); continue; }
  // Skip non-cryptic promotional entries (no progressive hints, e.g. the launch teaser).
  if (!Array.isArray(d.hints) || d.hints.length === 0) {
    problems.push(`${f}: excluded (no hints / non-cryptic) answer=${d.answer}`);
    continue;
  }

  // Full clue text (segments are space-delimited tokens/phrases)
  const clueText = d.clue.map(s => s.text).join(" ").replace(/\s+/g, " ").trim();

  // Definition / wordplay split when the segments are typed
  const defSegs = d.clue.filter(s => s.type === "definition").map(s => s.text);
  const wpSegs = d.clue.filter(s => s.type === "wordplay").map(s => s.text);

  // Enumeration, e.g. [7] -> "(7)", [4,3] -> "(4,3)"
  const enumeration = Array.isArray(d.config) ? "(" + d.config.join(",") + ")" : null;

  // Ordered progressive hints
  const hints = (d.hints || []).map((h, i) => ({
    order: i + 1,
    type: h.type,            // e.g. indicators, fodder, definition
    text: h.text,
    colour: h.colour ?? null,
    highlighting: h.highlighting ?? null, // char ranges into clueText
  }));

  puzzles.push({
    date: d.date,
    puzzleId: d.puzzleId,
    clue: clueText,
    enumeration,
    answer: d.answer.toUpperCase().trim(),
    answerLength: d.answer.replace(/[^A-Za-z]/g, "").length,
    wordLengths: d.config ?? null,
    letterRevealOrder: Array.isArray(d.letterRevealOrder) ? d.letterRevealOrder : null,
    par: d.par ?? null,                                   // official fixed par for the clue
    communityAveragePar: d.parDetails?.averagePar ?? avgParByDate[d.date] ?? null, // rounded avg of all solvers
    numHints: hints.length,
    hints,
    definition: defSegs.length ? defSegs.join(" ") : null,
    wordplay: wpSegs.length ? wpSegs.join(" ") : null,
    clueSegments: d.clue.map(s => ({ text: s.text, type: s.type })),
    setterName: d.setterName ?? null,
    explainerVideo: d.explainerVideo ?? null,
    solveCount: d.parDetails?.solveCount ?? null,
  });
}

puzzles.sort((a, b) => a.date.localeCompare(b.date));

// JSON array
fs.writeFileSync(path.join(__dirname, "minute_cryptic_dataset.json"),
  JSON.stringify(puzzles, null, 2));
// JSONL (one puzzle per line) for streaming eval harnesses
fs.writeFileSync(path.join(__dirname, "minute_cryptic_dataset.jsonl"),
  puzzles.map(p => JSON.stringify(p)).join("\n") + "\n");

// Summary
const parDist = {};
for (const p of puzzles) parDist[p.par] = (parDist[p.par] || 0) + 1;
const hintDist = {};
for (const p of puzzles) hintDist[p.numHints] = (hintDist[p.numHints] || 0) + 1;
console.log("puzzles:", puzzles.length);
console.log("date range:", puzzles[0]?.date, "->", puzzles[puzzles.length-1]?.date);
console.log("par distribution:", JSON.stringify(parDist));
console.log("numHints distribution:", JSON.stringify(hintDist));
console.log("with definition tag:", puzzles.filter(p=>p.definition).length);
console.log("problems:", problems.length);
if (problems.length) console.log(problems.slice(0,20).join("\n"));
