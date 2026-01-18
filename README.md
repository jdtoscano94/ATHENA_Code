# ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms
Progress in computational science depends on complex numerical workflows that must faithfully encode physical laws, yet translating conceptual insight into reliable code remains a major bottleneck. Although large language models can generate isolated code fragments, they lack the structured reasoning required to design, verify, and iteratively refine complete scientific pipelines. Here we introduce ATHENA, an agentic framework explicitly designed to emulate scientific research modeled as a knowledge-driven contextual bandit process. Its core loop separates conceptual policy from numerical realization through expert-derived conceptual scaffolding, enabling principled diagnosis, reformulation, and repair of computational strategies. Across scientific computing and scientific machine learning tasks, ATHENA autonomously discovers exact analytical solutions, constructs stable numerical solvers, diagnoses ill-posed formulations, and orchestrates hybrid symbolic–numeric workflows. Quantitatively, ATHENA exceeds expert-designed workflows, achieving superhuman accuracy and robustness on canonical benchmarks. By reframing computation as an object of agentic reasoning, the framework enables autonomous orchestration of heterogeneous algorithms across scientific domains.

### Citation

If you use ATHENA in your research, please cite the following preprint:

```bibtex
@article{toscano2025athena,
  title={ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms},
  author={Toscano, Juan Diego and Chen, Daniel T and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2512.03476},
  year={2025}
}
```
### Environment Setup

To replicate the environment used in the ATHENA framework, ensure you have Python 3.9+ installed. The framework relies on several foundation model APIs (e.g., Google Gemini, OpenAI, Anthropic) and standard scientific computing libraries.

**Prerequisites**

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/ATHENA.git](https://github.com/your-username/ATHENA.git)
cd ATHENA
pip install -r requirements.txt
```

**API Configuration**

ATHENA orchestrates multiple Foundation Models. You must configure your API keys in a `.env` file or export them as environment variables:

```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export GOOGLE_API_KEY="your_key"
# Add other provider keys as necessary
```

### Execution Workflow

ATHENA is designed to operate autonomously based on a formalized **User Request**. The execution flow is initiated via the Coordinator agent.

**1. Define the Problem**
Create a configuration file (e.g., `config.json`) or a natural language prompt defining the PDE, boundary conditions, and physical parameters.

**2. Initiate the HENA Loop**
Run the main orchestration script. This triggers the **Gatekeeper**, which routes the request to the Strategy Team.

```bash
python main.py --task "Solve the 1D Viscous Burgers equation with nu=0.01/pi" --mode autonomous
```

**3. Monitor Evolution**
The system will iteratively generate code, execute it, and refine the strategy. Logs and intermediate results (plots, code snapshots) are saved in the `experiments/` directory.

**4. Review Results**
Upon convergence or completion of the allocated budget, the final solution, code state $S_{final}$, and a comprehensive report summarizing the evolutionary trajectory are generated in the output folder.

