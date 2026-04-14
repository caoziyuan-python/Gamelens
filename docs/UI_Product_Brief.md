# GameLens UI Product Brief

## 1. Product Positioning
GameLens is a lightweight game insight dashboard for product managers and analysts.
Its core job is to turn App Store review data into:
- clear product conclusions
- prioritized action items
- evidence-backed theme and sentiment analysis
- fast AI-assisted follow-up answers

Target users:
- game product managers
- user research / insight teams
- publishing and growth teams

## 2. Core User Goal
After selecting a game, the user should be able to answer three questions in under 3 minutes:
- What is the biggest user problem right now?
- What should we prioritize first?
- How does this game compare with similar games?

## 3. Information Architecture
### Sidebar
- Game selector
- Start analysis button
- Competitor search
- Add searched games into custom analysis list

### Main Tabs
1. AI Overview
Main decision layer. Shows KPI, top conclusion, priority issues, and one AI recommendation.

2. Data Support
Shows topic cards, representative user quotes, and model validation results.

3. Deep Dive
Supports single-game charts and multi-game comparison mode.

4. Actionable Insights
Shows all insight cards with filtering, voting, and PRD export.

5. Multi-game Compare
Currently acts as a navigation / summary entry, because full comparison is already inside Deep Dive.

6. Learning Feedback
Shows feedback volume and usefulness rate by source.

7. AI Assistant
Chat-based layer for follow-up product questions with tool trace visibility.

## 4. Key Screens
### AI Overview
Purpose:
- give the answer first
- reduce cognitive load

Must-have modules:
- KPI row
- core conclusion banner
- left column: priority issue cards
- right column: short AI recommendation
- footer: data source / confidence description

### Data Support
Purpose:
- explain why the conclusion is trustworthy

Must-have modules:
- topic cards
- representative review quote
- validation table
- sentiment model agreement indicator

### Deep Dive
Purpose:
- support exploration after the top-level decision is understood

Single-game mode:
- rating distribution
- sentiment by rating
- sentiment pie
- topic heatmap
- keyword cloud

Multi-game mode:
- comparison table
- best vs worst summary
- positive sentiment comparison chart
- manual revenue efficiency comparison

### Actionable Insights
Purpose:
- provide an operational task list

Each card should include:
- priority
- impact count / ratio
- impact metric
- action recommendation
- evidence / trigger
- helpful / not helpful actions
- export PRD

### AI Assistant
Purpose:
- allow natural-language follow-up without leaving the dashboard

Must-have:
- example prompts
- chat input
- answer area
- optional tool trace
- chat history

## 5. UX Principles
- Answer first, evidence second
- Keep one clear primary action per area
- Make priority visually obvious
- Use confidence and source labels to build trust
- Avoid forcing users to interpret raw data before seeing conclusions

## 6. Visual Direction
- Dark dashboard style
- Strong contrast between “decision”, “evidence”, and “exploration”
- Use status colors carefully:
  - red: critical / P0
  - yellow: caution / P1
  - green: success / key conclusion / completed
  - blue: AI recommendation / informational panels

Typography guidance:
- large numeric KPI
- medium-weight section headers
- short card titles
- concise supporting copy

## 7. Interaction Notes
- Competitor search should feel lightweight and fast
- Added games should immediately appear in the selector
- AI recommendation should be cached to avoid repeat LLM calls on tab switch
- Feedback actions should be instant and low friction
- Export should produce a PM-friendly markdown PRD snippet

## 8. Design Priorities
Priority 1:
- polish AI Overview
- make issue cards more scannable
- strengthen hierarchy between conclusion and evidence

Priority 2:
- improve multi-game comparison readability
- reduce duplication between tabs

Priority 3:
- refine AI Assistant chat experience
- improve empty-state and loading-state visuals

## 9. Current Constraints
- Some insight text may be long because it comes from LLM output
- LLM availability may fail due to network / Azure issues
- The app must degrade gracefully to rule-based results

## 10. Success Criteria
A good UI redesign should make users feel:
- “I know the main problem immediately”
- “I trust where this conclusion comes from”
- “I know what to do next”
- “I can compare products without exporting data elsewhere”
