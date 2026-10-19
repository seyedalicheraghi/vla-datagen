export const meta = {
  name: 'fleet-research',
  description: 'Survey 2024-2026 publications on agent fleets, graphics agents, evaluator loops, realism metrics and Claude Code features; verify each report',
  phases: [
    { title: 'Research', detail: 'five parallel literature/doc sweeps (read + web only)', model: 'sonnet' },
    { title: 'Verify', detail: 'one skeptic per report checks sources and numbers', model: 'sonnet' },
  ],
}

const RULES = `HARD RULES (from the owner): you are on a cloud machine. Do NOT run any code, tests, scripts, Python or Isaac Sim, and do not install anything. Only read files (Read/Grep/Glob) and use the web: load WebSearch and WebFetch with ToolSearch (query "select:WebSearch,WebFetch"). If the proxy blocks a site, do not try to get around it; use another source (arXiv abs pages, OpenReview, project pages, Semantic Scholar, GitHub READMEs). Today is 2026-10-18: prefer the newest work (2025-2026) and label older work with its year. Never invent a paper, number or URL; if you cannot confirm something, say "unconfirmed".`

const CONTEXT = `Context: the owner's repo (local clone at /home/user/vla-datagen, private GitHub repo seyedalicheraghi/IsaacSim, an Isaac Lab fork) has a drone scene (scripts/drone_scene/) rendered in NVIDIA Isaac Sim 5.1 RTX: a quadrotor carrying two simulated LUCID machine-vision cameras (front, back) flies 2 km at 40 m over a real place (west Ames, Iowa) built from open geodata with PBR materials, an HDR sky and a calibrated sun; a GPU sensor model turns HDR renders into camera images. We are designing a FLEET of Claude Code agents that will run on the owner's own Ubuntu PC with one NVIDIA GPU (never on a cloud machine). The fleet iterates: render a few probe frames (minutes per render, one GPU, renders must be serialized) -> critique realism -> propose and implement code/material/lighting edits (possibly in parallel git worktrees) -> re-render -> keep only verified improvements. Requirements: faster and better than one agent; cheaper models for most helpers, the strongest model only where it matters.`

const FINDINGS = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' }, authors_or_org: { type: 'string' }, date: { type: 'string' }, venue: { type: 'string' },
      url: { type: 'string' }, claim: { type: 'string' }, numbers: { type: 'string', description: 'exact quantitative results, or "none"' },
      design_implication: { type: 'string' }, confirmed_from_primary_source: { type: 'boolean' },
    }, required: ['title', 'url', 'claim', 'design_implication', 'confirmed_from_primary_source'] } },
    design_rules: { type: 'array', items: { type: 'string' }, description: 'concrete, actionable rules for our fleet, each citing the finding titles it rests on' },
    open_questions: { type: 'array', items: { type: 'string' } },
  },
  required: ['findings', 'design_rules'],
}

const VERDICTS = {
  type: 'object',
  properties: {
    checked: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' }, source_exists: { type: 'boolean' },
      claim_supported: { type: 'string', enum: ['yes', 'partly', 'no', 'unverifiable'] },
      correction: { type: 'string', description: 'what the source actually says, with the correct number/URL; empty if fully supported' },
    }, required: ['title', 'source_exists', 'claim_supported', 'correction'] } },
    rules_to_drop_or_fix: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['checked', 'summary'],
}

const TOPICS = [
  { key: 'multi-agent-evidence', prompt: `Survey the most recent evidence (newest first) on WHEN and HOW multi-agent LLM systems beat a single agent, and how they fail. Cover at minimum, each from its primary source: Anthropic "How we built our multi-agent research system" (2025) and any later Anthropic engineering posts on subagents / agent teams / long-running agents; Google Research / DeepMind work quantifying when multi-agent helps or hurts (e.g. "Towards a Science of Scaling Agent Systems", coordination topologies, error amplification, capability saturation); Cognition "Don't Build Multi-Agents" (2025); "Why Do Multi-Agent LLM Systems Fail?" (MAST taxonomy, 2025); Microsoft Magentic-One (task/progress ledgers); Mixture-of-Agents; multi-agent debate and later papers questioning it; 2025-2026 evidence on parallel coding agents (worktrees, best-of-N with verifiers, SWE-bench style results), and any 2026 survey or benchmark of agent orchestration. Then search for anything newer in 2026. For each finding give the exact numbers and the implication for a fleet that edits one codebase and judges rendered images. End with design rules on: topology (central orchestrator vs peer), what to parallelize (read-only exploration, independent candidates) vs keep single-threaded, number of agents, single-writer vs many-writers, handoff/context format, verification and termination criteria, model tiering and token cost.` },
  { key: 'graphics-agents', prompt: `Survey (newest first, 2024-2026) agent systems that create or edit 3D scenes, materials, lighting or renderer settings in a loop with visual feedback from a vision-language model. Cover at minimum: SceneCraft (ICML 2024), BlenderAlchemy (ECCV 2024), BlenderGym (CVPR 2025, what it found about generator vs verifier compute / inference scaling), LL3M (2025), Make-it-Real (NeurIPS 2024), VLMaterial (2025), 3D-GPT, Holodeck / SceneX, and 2025-2026 work on: agents for NVIDIA Omniverse / Isaac Sim / OpenUSD (e.g. NVIDIA USD Code / ChatUSD / Omniverse agent toolkits), VLM-guided procedural material or shader editing, photoreal digital twins of real places from geodata, and sim-to-real appearance transfer (e.g. NVIDIA Cosmos Transfer) including whether a generative post-process fits a physically based camera-sensor dataset. For each: loop structure (proposer, renderer, critic, selection), pairwise vs score comparison, candidates per round, regression avoidance (revert, tree search), memory or library learning, measured gains. End with design rules for a render -> critique -> edit loop on ONE local GPU where each render takes minutes.` },
  { key: 'evaluator-loops', prompt: `Survey (2024-2026) evaluator-driven program improvement and self-improving agent loops, and LLM-as-judge reliability. Cover: AlphaEvolve (DeepMind 2025) and open re-implementations (OpenEvolve, ShinkaEvolve), Google AI co-scientist (generate-debate-evolve, Elo tournaments), Darwin Godel Machine, ADAS and AFlow (automated agent design), Reflexion and Self-Refine, Agentic Context Engineering (ACE, evolving playbooks), skill/library learning, and judge reliability (position bias, self-preference, verbosity bias; pairwise vs pointwise scoring; juries/ensembles such as "Replacing Judges with Juries" PoLL; MLLM-as-a-judge studies for images). Also any 2026 work. Extract concrete numbers and mechanisms: population size, selection and elitism, diversity, evaluation cascades (cheap checks before expensive evaluation), memory of failed attempts, judge-bias mitigations (randomized A/B order, hidden labels, several judges, ties allowed). End with design rules for a loop where one evaluation = rendering about 6 probe images on one GPU (minutes) and judging them with a vision-language model plus simple image statistics.` },
  { key: 'realism-metrics', prompt: `Find the best practical ways (2023-2026 literature and tools) to measure how photorealistic a rendered aerial/drone image is and how close it is to real footage, in a loop on ONE NVIDIA GPU with only a handful of images per round. Cover: distribution metrics (FID, KID, CMMD from "Rethinking FID" 2024, FD-DINOv2) and their minimum sample sizes; perceptual similarity (LPIPS, DreamSim); no-reference IQA (NIQE, BRISQUE, MUSIQ, CLIP-IQA, Q-Align, 2025-2026 VLM-based IQA); natural-image statistics (power spectrum ~1/f^alpha, gradient statistics, colour statistics, contrast and haze falloff with distance / aerial perspective); multimodal LLMs as realism or AI-generated-image detection judges and how reliable they are; sim-to-real appearance gap studies for drones and robots. Then: which real reference imagery is legally usable for west Ames, Iowa (about 42.03 N, 93.68 W): USDA NAIP orthoimagery (resolution, recent years for Iowa, licence, how to read a small area from public cloud-optimized GeoTIFFs via Microsoft Planetary Computer STAC or AWS), Iowa state orthophotos, Mapillary (licence, API token), and public drone video datasets with licences (VisDrone, UAVid, others) for about 40 m oblique views. Recommend a minimal metric set implementable with numpy, PIL and torch (name any optional pip packages and their licences) and say what each metric catches.` },
]

const CC_PROMPT = `${RULES}\n\n${CONTEXT}\n\nTask: give me the exact CURRENT file formats and behaviour, from the official Claude Code documentation (code.claude.com/docs or docs.anthropic.com), for building this fleet so the owner can use it with their own local Claude Code (VS Code extension and CLI on Ubuntu) right after git pull. For each item give a minimal working example and the doc URL; mark anything not confirmed in the docs as unconfirmed.\n1. Custom subagents in .claude/agents/*.md: every supported frontmatter field (name, description, tools, disallowedTools, model and its allowed values/aliases such as sonnet, opus, haiku, inherit; effort; color; permissionMode; maxTurns; skills; memory; isolation such as worktree; hooks; mcpServers; anything else), how description drives automatic delegation, whether a subagent can spawn subagents, and whether subagents can read images (PNG) with Read.\n2. Skills in .claude/skills/<name>/SKILL.md: frontmatter (name, description, allowed-tools, model, context: fork, agent, argument-hint, disable-model-invocation, user-invocable, ...), invoking with /<name>, $ARGUMENTS, supporting files.\n3. Saved workflows in .claude/workflows/ for a "Workflow" tool (multi-agent orchestration scripts): is this documented publicly, file format, invocation by name, opt-in rules. If not public, say so.\n4. Agent teams (experimental; env var such as CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS): what they are, how to enable, when to use them instead of subagents, limitations.\n5. Running the main session as a custom agent (claude --agent <name>, or the "agent" setting) and output styles (.claude/output-styles/*.md) for how the main assistant talks to the owner.\n6. Git worktrees: claude --worktree, subagent isolation: worktree, cleanup.\n7. Hooks that could enforce owner rules: PreToolUse on Bash to require confirmation for git push (permissionDecision "ask") and to block commands; exact JSON in .claude/settings.json and the hook script contract (stdin JSON, exit code 2, JSON output).\n8. Project .claude/settings.json permissions: allow / ask / deny rule syntax (e.g. "Bash(git push:*)"), and how it combines with settings.local.json.\n9. Headless runs: claude -p with --agents / --max-turns / --output-format for unattended overnight loops, and model aliases currently valid.`

phase('Research')
const reports = await pipeline(
  [...TOPICS, { key: 'claude-code-features', prompt: null }],
  t => agent(t.prompt ? `${RULES}\n\n${CONTEXT}\n\nTask: ${t.prompt}` : CC_PROMPT, {
    label: `research:${t.key}`, phase: 'Research', schema: FINDINGS, model: 'sonnet',
    ...(t.prompt ? {} : { agentType: 'claude-code-guide' }),
  }),
  (report, t) => {
    if (!report) return null
    return agent(`${RULES}\n\nYou are a skeptical fact-checker. Below is a research report (JSON) on "${t.key}". For EVERY finding, open the source (URL, or search the exact title) and check (a) it exists with that title/author/date and (b) the claim and numbers match. Default to claim_supported="unverifiable" if you cannot open the source; do not trust the report. Also list design rules that rest on wrong or unverifiable findings.\n\nREPORT:\n${JSON.stringify(report)}`, {
      label: `verify:${t.key}`, phase: 'Verify', schema: VERDICTS, model: 'sonnet',
    }).then(v => ({ key: t.key, report, verdict: v }))
  },
)
return reports.filter(Boolean)
