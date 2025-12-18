# Physical AI & Humanoid Robotics Textbook

A comprehensive, spec-driven technical textbook on Physical AI and humanoid robotics built with Docusaurus.

## 📚 Book Contents

- **Module 1: Physical AI Foundations** — Embodied intelligence and why robots are different
- **Module 2: Digital Twins & Gazebo** — Simulation and robot modeling with URDF
- **Module 3: Perception & NVIDIA Isaac** — Sensors, SLAM, and sim-to-real transfer
- **Module 4: Vision-Language-Action Systems** — LLMs, language understanding, and robot control
- **Module 5: Capstone Project** — Complete autonomous humanoid system

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ (for Docusaurus)
- **npm** or **yarn** (package manager)
- **Python** 3.9+ (for code examples and testing)
- **Git**

### Installation & Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/asad/PhysicalAI-Book.git
   cd PhysicalAI-Book
   ```

2. **Install Docusaurus dependencies**:
   ```bash
   cd book
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```
   The book will be available at `http://localhost:3000`

4. **Build the static site** (for production):
   ```bash
   npm run build
   ```
   Output will be in `book/build/`

## 📖 Project Structure

```
book/
├── docs/                    # Book content (Markdown)
│   ├── 01-foundations/      # Module 1
│   ├── 02-simulation/       # Module 2
│   ├── 03-perception/       # Module 3
│   ├── 04-vla-systems/      # Module 4
│   ├── 05-capstone/         # Module 5
│   ├── glossary.md
│   ├── references.md
│   └── troubleshooting.md
├── static/                  # Static assets
│   ├── diagrams/            # Architecture diagrams (SVG/PNG)
│   ├── code-examples/       # Python code snippets
│   └── media/               # Images, videos
├── src/
│   ├── css/custom.css       # Custom styling
│   └── pages/               # Custom pages (if needed)
├── docusaurus.config.js     # Docusaurus configuration
├── sidebars.js              # Navigation structure
└── package.json             # NPM dependencies

examples/                     # Capstone project code
├── humanoid-sim/            # Main capstone
│   ├── ros2_nodes/
│   ├── gazebo_models/
│   ├── perception/
│   ├── planning/
│   └── vla/
├── requirements.txt
└── setup.sh

tests/                        # Testing
├── unit/                    # Unit tests for code examples
├── integration/             # Integration tests
├── capstone/                # Capstone system tests
└── diagrams/                # Diagram validation

specs/                        # Specification documents
└── 001-physical-ai-book/
    ├── spec.md              # Feature specification
    ├── plan.md              # Implementation plan
    ├── tasks.md             # Task breakdown
    └── research.md          # Technical research
```

## 🔧 Configuration

### Edit Site Metadata

Modify `book/docusaurus.config.js`:
- `title`: Site title
- `tagline`: Tagline
- `url`: Site URL (for GitHub Pages: `https://username.github.io`)
- `baseUrl`: Base path (for GitHub Pages: `/PhysicalAI-Book/`)

### Update Navigation

Modify `book/sidebars.js` to change the sidebar structure.

## 📝 Writing Content

### Add a New Page

1. Create a `.md` file in the appropriate module folder:
   ```bash
   book/docs/02-simulation/new-section.md
   ```

2. Add it to `sidebars.js`:
   ```javascript
   items: [
     '02-simulation/intro',
     '02-simulation/new-section',  // Add here
     '02-simulation/exercises',
   ]
   ```

3. Link to it from other pages:
   ```markdown
   [New Section](../02-simulation/new-section.md)
   ```

### Code Block Syntax

Use GitHub-flavored markdown code blocks:

````markdown
```python
import rclpy

def main():
    print("Hello, ROS 2!")

if __name__ == '__main__':
    main()
```
````

Supported languages: `python`, `bash`, `yaml`, `xml`, `cpp`, `java`, etc.

### Diagrams

Include SVG or PNG diagrams:

```markdown
![Architecture Diagram](../static/diagrams/architecture.svg)
```

## 🧪 Testing

### Run Code Example Tests

```bash
cd /path/to/repo
python -m pytest tests/unit/ -v
```

### Test Specific Module

```bash
pytest tests/unit/test_module1_examples.py -v
```

### Coverage Report

```bash
pytest tests/ --cov --cov-report=html
# Open htmlcov/index.html
```

## 🚀 Deployment

### Local Testing

1. Build the site:
   ```bash
   cd book
   npm run build
   ```

2. Serve locally:
   ```bash
   npm run serve
   ```

3. Open `http://localhost:3000` and verify all pages

### Deploy to GitHub Pages

1. **Enable GitHub Pages** in repository settings:
   - Go to Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages` / root

2. **GitHub Actions will automatically deploy** when you push to `main` or `001-physical-ai-book` branches

3. **View deployed site** at:
   ```
   https://username.github.io/PhysicalAI-Book/
   ```

## 📚 Writing Guidelines

### Structure

- **Clear introductions** — Start each module with "why" questions
- **Progressive complexity** — Build from basics to advanced
- **Concrete examples** — Every concept needs code
- **Practical exercises** — Let readers apply knowledge
- **Visual aids** — Diagrams for complex systems

### Code Examples

- Keep examples **<30 seconds execution time**
- Include **comments** explaining each step
- Provide **expected output**
- Use **Python 3.9+** for compatibility with ROS 2
- All examples must **run without errors** (SC-009)

### Links

- Use **relative links** between modules
- Link to **external resources** (ROS 2 docs, Gazebo, Isaac)
- Include **references section** at end of chapters

## 🐛 Troubleshooting

### Build Errors

```bash
# Clear Docusaurus cache
npm run clear
npm run build
```

### Node version issues

```bash
# Check Node version
node --version  # Should be 18+

# Update Node
nvm install 18
nvm use 18
```

### Port already in use

```bash
# Use a different port
npm start -- --port 3001
```

## 📋 Checklist: Adding New Content

- [ ] Created markdown file in appropriate module folder
- [ ] Added to `sidebars.js` navigation
- [ ] Added internal links between related pages
- [ ] Included code examples (if applicable)
- [ ] Code examples run and produce expected output
- [ ] Added exercises (if module introduction)
- [ ] Proofread for clarity and correctness
- [ ] Built locally and verified in browser
- [ ] Pushed to branch

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add your content or fixes
4. Test locally (build, run code examples)
5. Commit with clear messages
6. Create a pull request

## 📄 License

This textbook is open-source and available under the MIT License.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/asad/PhysicalAI-Book/issues)
- **Discussions**: [GitHub Discussions](https://github.com/asad/PhysicalAI-Book/discussions)

## 🎯 Project Goals

- ✅ Teach Physical AI through a comprehensive curriculum
- ✅ Provide working code examples for every concept
- ✅ Build a complete autonomous humanoid capstone
- ✅ Bridge theory and practice
- ✅ Make robotics accessible to CS students

---

**Ready to get started?** Open `http://localhost:3000` after running `npm start`.
