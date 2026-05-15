# Optimal Workforce Management: Finding the Sweet Spot Between Cost and Effectiveness in Expert Networks

## Idea

The main goal of this research is to find the sweet spot between cost and effectiveness in expert networks. 
While big and costly LLMs (such as GPT-4) are able to solve most of the problems by themselves (so-called "monolithic" approach), 
multi-agent systems offer the benefit of specialization and cooperation. Just like the similar situations occur in real-world teams
where each member has a specific skill set and is responsible for a specific task.

This study aims to find the sweet spot in managing a team of AI agents. We focus on optimization problems (e.g., planning, resource allocation) 
that require multi-stage reasoning, mathematical modeling, and code generation.

## Used Architectures

During our research we will use the following architectures:

### 2-Agent Architecture

- agent roles: Modeler + Solver (implementation + validation)
- agent communication:

![2-Agent Architecture](assets/architectures/2-agent.png)

The modeler receives a textual problem, formalizes it, and passes the specification to the coder. The coder writes the code, executes it, and returns the result. 
If the code fails, the coder attempts to fix it themselves in a simple loop.

Expected results:

The expected result in this situation would be that it is the most cost-effective solution, but it may not perform well on more complex problems.

### 3-Agent Architecture

- agent roles: Modeler + Coder + Reviewer
- agent communication:

![3-Agent Architecture](assets/architectures/3-agent.png)

It introduces an explicit ‘code review’ loop. Modeller -> Coder -> Reviewer. The reviewer tests the code for execution errors (does it work?) and logical errors (does it solve the right problem?). 
If errors are found, the code is sent back to the Coder with specific feedback.

Expected results:
We expect that the solution will be more costly than the 2-Agent architecture (more tokens are used), but it may perform better on more complex problems.

### 4-Agent Architecture

- agent roles: Planner + Modeler + Coder + Reviewer
- agent communication:

![4-Agent Architecture](assets/architectures/4-agent.png)

The planner is responsible for planning the solution. 
The modeler receives a textual problem, formalizes it, and passes the specification to the coder. 
The coder writes the code and passes it to the planner for validation. The planner then returns the result.


Expected results:

It is suspected that this approach will be best for multi-stage or ambiguously defined problems. 
However, it can generate excessive coordination costs. 
This mainly refers to too many iterations for more challenging tasks or ‘unnecessary’ running of the entire 
pipeline for simpler tasks, which can reduce performance compared to a 3-agent architecture.

## Used Datasets

CardinalOperations/NL4OPT, available on the Hugging Face platform, will serve as our main dataset (benchmark).

This dataset comes from the NL4Opt competition (NeurIPS 2022) and consists of approximately 1,100 linear programming problems described entirely in natural language.

Example dataset entry:

| en_question | en_answer |
|--------------|------------|
| A lab has 1000 units of medicinal ingredients to make two pills, a large pill and a small pill. A large pill requires 3 units of medicinal ingredients and 2 units of filler. A small pill requires 2 units of medicinal ingredients and 1 unit of filler. The lab has to make at least 100 large pills. However, since small pills are more popular at least 60% of the total number of pills must be small. How many of each should be made to minimize the total number of filler material needed? | 350.0 |
| An accounting firm employs part time workers and full time workers. Full time workers work 8 hours per shift while part time workers work 4 hours per shift. In addition, full time workers are paid $300 per shift while part time workers are paid $100 per shift. Currently, the accounting firm has a project requiring 500 hours of labor. If the firm has a budget of $15000, how many of each type of worker should be scheduled to minimize the total number of workers. | 100.0 |

Data description:
- **en_question**: The description of the optimization problem in natural language (English).
- **en_answer**: The numerical answer being the optimal value of the objective function or the solution, such as the minimum number of workers, minimum amount of material, etc.

## Experiments

### Experiment 1: Sweet Spot Analysis (Number of Agents)

Objective: To find the optimal number of agents for a fixed base model.

Method: We use the most powerful model available (e.g., GPT-4o) as the ‘brain’ for all agents. We run our entire dataset through 2-, 3-, and 4-agent architectures.

Research question: Does adding a Reviewer agent (in 3-A) or Planner agent (in 4-A) significantly improve success rates? At what cost (tokens, time)? Is 4-A worse than 3-A for simple problems?

### Experiment 2: Cost vs. Quality Trade-off (Base Models)
Objective: To investigate whether a ‘team’ of smaller models can match an ‘expert’.

Method: We select one optimal architecture (e.g., 3-agent, based on the results of Experiment 1). Then we run it using different base models. We can also try to find the best strategy, e.g., a small model for modelling and evaluation and a large one for programming.

Used models:
- Large/Expensive (High-end): GPT-4o
- Medium/Efficient (Mid-range): Llama-3 70B 
- Small/Low-end: Llama-3 8B

Research question: Can a team of 3 agents based on Llama-3 8B (low cost) achieve similar effectiveness as 3 agents based on GPT-4o (high cost)? Where is the optimal point on the cost/effectiveness curve?

## Evaluation Metrics

Set of metrics to evaluate the performance of the agents.

### Cost Metrics
- **Token Cost**: The total number of tokens (both input and output) consumed by all agents in the system to solve a single problem.
- **Latency**: The total time (in seconds) from submitting the problem to receiving the final analysis.

### Solution Quality Metrics
- **Solved problems**: The number of problems solved.
- **Accuracy**: The percentage of correct answers.
- **Precision**: The percentage of correct answers that are relevant.
- **Recall**: The percentage of relevant answers that are found.

### Process Metrics
- **Incorrect answers**: The number of incorrect answers.
- **Iterations**: The number of iterations in the loop (may help with identifying that the problem was too hard for model, or that the solution was too complex).