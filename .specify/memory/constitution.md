# PhysicalAI-Book Constitution

## Core Principles

### I. Spec-Driven Development (NON-NEGOTIABLE)
All implementation must be traceable to the feature specification; No code, documentation or tests without corresponding requirements in spec.md; Any requirement changes must go through spec update first

### II. Executable Examples
Every code example in the book must be tested and runnable; All examples must execute within 30 seconds on target hardware; Examples must be validated with automated tests

### III. Progressive Complexity
Modules build sequentially; Foundational concepts before applications; Reader must be able to understand Module N before proceeding to Module N+1; Cross-references between modules are encouraged

### IV. Performance Standards
- Capstone project inference latency must be under 2 seconds per spoken command
- Book build process must complete in under 5 minutes
- All code examples must execute in under 30 seconds
- Simulations must run at 30+ Hz on target hardware
- Book must build and serve without errors

### V. Pedagogical Coherence
All content must serve the learning objectives defined in user stories; Theory must connect to practical examples; Each module must have clear success criteria and acceptance scenarios

### VI. Open Source First
Prefer open-source tools, libraries and models where possible; Commercial dependencies should have free tiers sufficient for learning; Document alternatives for proprietary tools

## Development Workflow

### Quality Gates
- All code examples pass pytest validation before merging
- All internal links validated during build process
- All diagrams properly referenced and rendered
- Requirements traceability matrix maintained
- Success criteria verifiable through tests

### Review Process
- All PRs verify compliance with constitution principles
- Cross-artifact consistency checked (spec, plan, tasks alignment)
- Performance requirements validated
- Learning objectives verified through acceptance scenarios

## Governance
- Constitution supersedes all other practices
- All implementation must comply with these principles
- Amendments require explicit documentation and approval
- All PRs must verify constitution compliance

**Version**: 1.0.0 | **Ratified**: 2025-12-14 | **Last Amended**: 2025-12-16
