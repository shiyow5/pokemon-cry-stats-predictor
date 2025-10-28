# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PokémonCryML is a machine learning project for working with Pokémon cries (vocalizations). This is an early-stage project with minimal source code currently implemented.

## Development Environment

This project uses Python with a virtual environment located at `.venv/`.

**Activating the virtual environment:**
```bash
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

**Installing dependencies (when requirements.txt exists):**
```bash
pip install -r requirements.txt
```

## Executable Specifications (ExecPlans)

This repository follows a rigorous specification-driven development workflow defined in `.agent/PLANS.md`. When creating or implementing features:

1. **Read `.agent/PLANS.md` thoroughly** - It contains comprehensive guidelines for writing executable specifications that enable any contributor to implement features from scratch.

2. **ExecPlans are living documents** - They must contain:
   - `Progress`: Checkbox list with timestamps tracking granular steps
   - `Surprises & Discoveries`: Unexpected behaviors or insights discovered during implementation
   - `Decision Log`: All design decisions with rationale and dates
   - `Outcomes & Retrospective`: Summary of achievements, gaps, and lessons learned

3. **Key principles from PLANS.md:**
   - **Self-contained**: Include all knowledge needed for a novice to succeed
   - **Observable outcomes**: Define what users can do after implementation and how to verify it
   - **Plain language**: Define all technical terms; avoid undefined jargon
   - **Idempotent**: Steps should be safe to run multiple times
   - **Validation-focused**: Include exact test commands and expected outputs

4. **When implementing from an ExecPlan:**
   - Do not prompt for "next steps" - proceed autonomously to the next milestone
   - Keep all sections up to date as you progress
   - Resolve ambiguities independently
   - Commit frequently
   - Update the Decision Log when making significant choices

5. **Format**: ExecPlans are written as single markdown documents with specific sections (see skeleton in PLANS.md). When writing to a file, omit triple backticks; when including inline, use a single fenced code block.

## Project Structure

As the project develops, source code should be organized logically (e.g., separate directories for data processing, model training, inference, utilities). Update this section as the architecture emerges.
