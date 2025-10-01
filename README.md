<div id="top">

<!-- HEADER STYLE: MODERN -->
<div align="left" style="position: relative; width: 100%; height: 100%; ">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: absolute; top: 0; right: 0;" alt="Project Logo"/>

# AGENT-API

<em><em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Typer-000000.svg?style=flat-square&logo=Typer&logoColor=white" alt="Typer">
<img src="https://img.shields.io/badge/TOML-9C4121.svg?style=flat-square&logo=TOML&logoColor=white" alt="TOML">
<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=flat-square&logo=FastAPI&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/LangChain-1C3C3C.svg?style=flat-square&logo=LangChain&logoColor=white" alt="LangChain">
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat-square&logo=Docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
<br>
<img src="https://img.shields.io/badge/GitHub%20Actions-2088FF.svg?style=flat-square&logo=GitHub-Actions&logoColor=white" alt="GitHub%20Actions">
<img src="https://img.shields.io/badge/Google-4285F4.svg?style=flat-square&logo=Google&logoColor=white" alt="Google">
<img src="https://img.shields.io/badge/uv-DE5FE9.svg?style=flat-square&logo=uv&logoColor=white" alt="uv">
<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat-square&logo=Pydantic&logoColor=white" alt="Pydantic">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat-square&logo=YAML&logoColor=white" alt="YAML">

</div>
</div>
<br clear="right">

---

## ☀️ Table of Contents

<details>
<summary>Table of Contents</summary>

- [☀ ️ Table of Contents](#-table-of-contents)
- [🌞 Overview](#-overview)
- [🔥 Features](#-features)
- [🌅 Project Structure](#-project-structure)
    - [🌄 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
    - [🌟 Prerequisites](#-prerequisites)
    - [⚡ Installation](#-installation)
    - [🔆 Usage](#-usage)
    - [🌠 Testing](#-testing)
- [🌻 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [✨ Acknowledgments](#-acknowledgments)

</details>

---

## 🌞 Overview



---

## 🔥 Features

|      | Component       | Details                              |
| :--- | :-------------- | :----------------------------------- |
| ⚙️  | **Architecture**  | <ul><li>Python-based application</li><li>Utilizes FastAPI for web routing</li><li>Makefile for build and management</li><li>Containerized with Docker</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Uses Pydantic for data validation</li><li>Loguru for logging</li><li>Consistent use of Google GenAI and Langchain libraries indicating code reusability</li></ul> |
| 📄 | **Documentation** | <ul><li>Docker setup documented in Dockerfile and docker-compose.yaml</li><li>Test workflow documented in '.github/workflows/test.yml'</li></ul> |
| 🔌 | **Integrations**  | <ul><li>GitHub Actions for CI/CD</li><li>Python and Docker for runtime</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Codebase appears to be modular with usage of various libraries</li><li>Makefile for task segmentation</li></ul> |
| 🧪 | **Testing**       | <ul><li>Uses pytest for testing</li><li>CI/CD pipeline in GitHub Actions</li><li>Test workflow documented in '.github/workflows/test.yml'</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Performance details not available in the provided context</li></ul> |
| 🛡️ | **Security**      | <ul><li>Security details not available in the provided context</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Managed using Python's built-in package manager</li><li>Dependencies include FastAPI, Google GenAI, Loguru, Langchain, and more</li></ul> |
| 🚀 | **Scalability**   | <ul><li>Scalability details not available in the provided context</li></ul> |
```

**Notes:**
- This table summarizes the technical features of the "agent-api" project based on the provided context. Some sections might not be fully detailed due to the lack of information.
- Make sure to update the table as the project evolves and more information becomes available.

---

## 🌅 Project Structure

```sh
└── agent-api/
    ├── .github
    │   └── workflows
    │       └── test.yml
    ├── AGENTS.md
    ├── Dockerfile
    ├── Makefile
    ├── data
    │   └── .gitkeep
    ├── docker-compose.yaml
    ├── examples
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-312.pyc
    │   │   ├── gemini_image_test.cpython-312.pyc
    │   │   ├── generate_image_test.cpython-312.pyc
    │   │   ├── mcp_test.cpython-312.pyc
    │   │   ├── mcp_use_example.cpython-312.pyc
    │   │   ├── muti_mcp_test.cpython-312.pyc
    │   │   ├── supervisor_service_demo.cpython-312.pyc
    │   │   ├── test.cpython-312.pyc
    │   │   └── test_handoff.cpython-312.pyc
    │   ├── gemini_image_test.py
    │   ├── generate_image_test.py
    │   ├── mcp_test.py
    │   ├── mcp_use_example.py
    │   ├── muti_mcp_test.py
    │   ├── supervisor_service_demo.py
    │   ├── test.py
    │   └── test_handoff.py
    ├── folde_structure.py
    ├── interface
    │   ├── __pycache__
    │   │   └── chainlit_ui.cpython-312.pyc
    │   └── chainlit_ui.py
    ├── notebooks
    │   └── .gitkeep
    ├── pyproject.toml
    ├── src
    │   ├── agent_api.egg-info
    │   │   ├── PKG-INFO
    │   │   ├── SOURCES.txt
    │   │   ├── dependency_links.txt
    │   │   ├── entry_points.txt
    │   │   ├── requires.txt
    │   │   └── top_level.txt
    │   └── lastminute_api
    │       ├── __init__.py
    │       ├── __pycache__
    │       ├── application
    │       ├── config.py
    │       ├── domain
    │       └── infrastructure
    ├── static
    │   └── .gitkeep
    ├── tests
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-311.pyc
    │   │   ├── __init__.cpython-312.pyc
    │   │   ├── conftest.cpython-311-pytest-8.4.2.pyc
    │   │   ├── conftest.cpython-312-pytest-7.4.4.pyc
    │   │   ├── conftest.cpython-312-pytest-8.4.2.pyc
    │   │   ├── test_lastminute_api.cpython-311-pytest-8.4.2.pyc
    │   │   ├── test_lastminute_api.cpython-312-pytest-7.4.4.pyc
    │   │   ├── test_llm_provider_prompts.cpython-311-pytest-8.4.2.pyc
    │   │   ├── test_llm_provider_prompts.cpython-312-pytest-7.4.4.pyc
    │   │   ├── test_llm_providers.cpython-311-pytest-8.4.2.pyc
    │   │   ├── test_llm_providers.cpython-312-pytest-7.4.4.pyc
    │   │   ├── test_nano_banana.cpython-311-pytest-8.4.2.pyc
    │   │   ├── test_nano_banana.cpython-312-pytest-7.4.4.pyc
    │   │   └── test_nanobanana_openai.cpython-312-pytest-8.4.2.pyc
    │   ├── conftest.py
    │   ├── test_lastminute_api.py
    │   ├── test_llm_provider_prompts.py
    │   ├── test_llm_providers.py
    │   ├── test_nano_banana.py
    │   └── test_nanobanana_openai.py
    ├── todo.txt
    └── uv.lock
```

### 🌄 Project Index

<details open>
	<summary><b><code>AGENT-API/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/todo.txt'>todo.txt</a></b></td>
					<td style='padding: 8px;'>- In the context of the entire codebase, todo.txt serves as a central task management hub<br>- It outlines the progress of various development tasks such as image and supervisor node enhancement, minimap and flashcard integration, and improvements to the mind map agent<br>- The file also mentions upcoming deadlines.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/Dockerfile'>Dockerfile</a></b></td>
					<td style='padding: 8px;'>- The Dockerfile prepares a Python 3.12 environment and installs the necessary dependencies for the application<br>- It sets the working directory, synchronizes dependencies using uv, and copies the application into the container<br>- The file also specifies that the application is to be run on port 8000, accessible from any host.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/pyproject.toml'>pyproject.toml</a></b></td>
					<td style='padding: 8px;'>- The pyproject.toml serves as the project configuration center for the agent-api project<br>- It outlines the project's software dependencies, python version requirements, development tools, linting rules, and scripts<br>- Primarily, it's instrumental in setting up the environment for the AI agent to generate meaningful mind maps using web data.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/folde_structure.py'>folde_structure.py</a></b></td>
					<td style='padding: 8px;'>- FolderStructure.py generates a visual representation of the directory tree for the agent-api personal project<br>- Leveraging the DisplayTree function from the directory_tree module, it prints the structure from the specified path, aiding in understanding the project's organization and file hierarchy.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/Makefile'>Makefile</a></b></td>
					<td style='padding: 8px;'>- Makefile orchestrates the projects lifecycle operations using Docker Compose<br>- Specifically, it defines tasks for building the projects Docker images, initiating the project in a detached state, and halting the project<br>- Its role within the codebase architecture is to streamline the development workflow and ensure consistent build and runtime environments.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/docker-compose.yaml'>docker-compose.yaml</a></b></td>
					<td style='padding: 8px;'>- Docker-compose.yaml orchestrates the deployment of the agent-api service in a Docker environment<br>- It specifies container build information, port mappings, environment variables, and network configurations<br>- The file ensures that the agent-api service is appropriately isolated and can communicate with other services within the agent-api-network.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- .github Submodule -->
	<details>
		<summary><b>.github</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ .github</b></code>
			<!-- workflows Submodule -->
			<details>
				<summary><b>workflows</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ .github.workflows</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/.github/workflows/test.yml'>test.yml</a></b></td>
							<td style='padding: 8px;'>- Test Agent API, found in the GitHub workflow directory, manages the continuous integration process of the codebase<br>- It triggers on push or pull requests to the main branches, verifying the compatibility on different Python versions and Ubuntu OS<br>- This file also governs the linting, installation, and testing of the project using uv, ruff, and pytest tools respectively.</td>
						</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- examples Submodule -->
	<details>
		<summary><b>examples</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ examples</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/examples/gemini_image_test.py'>gemini_image_test.py</a></b></td>
					<td style='padding: 8px;'>- Gemini_image_test.py is a standalone smoke test script that verifies the functionality of the Gemini client within the Nano Banana API infrastructure<br>- It generates an image based on a specific prompt and saves it locally, ensuring that the Gemini API key, prompt scaffolding, and binary decoding are functioning as expected.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/examples/muti_mcp_test.py'>muti_mcp_test.py</a></b></td>
					<td style='padding: 8px;'>- Utilizing the lastminute_api, muti_mcp_test.py implements an agent to search for recent clinical trials involving dental implants<br>- The agent is created and executed in an asynchronous manner, producing results that are printed to the console<br>- The code forms a crucial part of the broader projects functionality related to clinical trials data retrieval.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/examples/test_handoff.py'>test_handoff.py</a></b></td>
					<td style='padding: 8px;'>- In the broader context of the codebase, test_handoff.py serves as a demonstrative tool for validating the custom_handoff_tool invocation<br>- It provides an example of how to utilize the tool with an agent, named tavily_agent in this case, and passes a test task to it<br>- The script prints the result, as well as the subsequent goto and update commands.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/examples/test.py'>test.py</a></b></td>
					<td style='padding: 8px;'>- The code in examples/test.py provides multifold functionality around the creation, display, and management of mindmaps<br>- It is designed to create mindmaps both directly and through an agent-based system, extract reference IDs from the results, display the created mindmaps, and manage batch creation of multiple mindmaps<br>- It also offers an interactive session for mindmap creation.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/examples/supervisor_service_demo.py'>supervisor_service_demo.py</a></b></td>
					<td style='padding: 8px;'>- Running the supervisory agent service end-to-end, the script in examples/supervisor_service_demo.py manages the revision agent process<br>- It initializes the environment, configures logging, and executes the revision agent with a specific query<br>- Results are then summarized and presented, with a focus on the query type and an optional image URL.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/examples/mcp_use_example.py'>mcp_use_example.py</a></b></td>
					<td style='padding: 8px;'>- The code connects ArXiv and PubMed search functionality to the Model Context Protocol (MCP) tools in the mcp_use SDK<br>- It creates lightweight connectors that translate tool calls into HTTP requests<br>- The connectors are consumed by an MCPAgent, demonstrating how to perform interactive search queries across both providers.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/examples/mcp_test.py'>mcp_test.py</a></b></td>
					<td style='padding: 8px;'>- Leveraging the ChatOpenAI model from the langchain_openai library, mcp_test.py initiates a Tavily search for AI news<br>- After loading environment variables and configuring the MCPClient, it deploys an MCPAgent to run the query<br>- The resulting data is then printed to the console for user review.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/examples/generate_image_test.py'>generate_image_test.py</a></b></td>
					<td style='padding: 8px;'>- Generate_image_test.py is a standalone utility in the project that leverages the OpenAI API to create an image based on a specified prompt<br>- The image is then saved to a specified file location<br>- This script validates the availability of necessary environment variables, initializes the OpenAI client, sends the image generation request, and handles potential errors throughout the process.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- interface Submodule -->
	<details>
		<summary><b>interface</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ interface</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/interface/chainlit_ui.py'>chainlit_ui.py</a></b></td>
					<td style='padding: 8px;'>- Chainlit_ui.py serves as the entry point for the LastMinute supervisory agent service in the Chainlit application<br>- It facilitates user interaction, handling chat initiation and message reception<br>- The script also manages the execution of the revision agent, the rendering of generated images, and the maintenance of chat history.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- src Submodule -->
	<details>
		<summary><b>src</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ src</b></code>
			<!-- agent_api.egg-info Submodule -->
			<details>
				<summary><b>agent_api.egg-info</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.agent_api.egg-info</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/agent_api.egg-info/SOURCES.txt'>SOURCES.txt</a></b></td>
							<td style='padding: 8px;'>- The <code>SOURCES.txt</code> in the <code>agent_api.egg-info</code> directory lists all the resources in the project<br>- Its a detailed inventory of files required for the project, from configuration and setup files to the main application code, tests, and dependencies<br>- Organised by the project structure, it aids in understanding the architecture and navigation of the lastminute_api' project.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/agent_api.egg-info/top_level.txt'>top_level.txt</a></b></td>
							<td style='padding: 8px;'>- Serving as a pivotal component of the overall codebase, src/agent_api.egg-info/top_level.txt directs the Python interpreter to the lastminute_api package<br>- This navigation aid significantly streamlines the importation process, ensuring seamless and efficient interactions within the overarching project structure.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/agent_api.egg-info/dependency_links.txt'>dependency_links.txt</a></b></td>
							<td style='padding: 8px;'>- Dependency_links.txt within src/agent_api.egg-info serves as a registry for external dependencies of the agent_api module<br>- It maintains the integrity and functionality of the module by ensuring all required external resources are correctly referenced<br>- This contributes to the overall projects robustness, reliability, and ease of maintenance.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/agent_api.egg-info/requires.txt'>requires.txt</a></b></td>
							<td style='padding: 8px;'>- Requires.txt in the src/agent_api.egg-info directory specifies necessary dependencies for the Agent API service, facilitating interaction with various tools like FastAPI, Loguru, and Pydantic<br>- It also includes libraries for language processing (langchain, langgraph), AI models (langchain-openai, langchain-groq), and data visualization (matplotlib)<br>- This ensures a smooth runtime environment for the service.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/agent_api.egg-info/entry_points.txt'>entry_points.txt</a></b></td>
							<td style='padding: 8px;'>- Entry_points.txt under src/agent_api.egg-info establishes a command-line interface for the LastMinute API<br>- It links the CLI command lastminute_api to the application defined in lastminute_api.cli<br>- This enables users to interact with the LastMinute API directly from their console, enhancing the accessibility of the application within the entire codebase architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/agent_api.egg-info/PKG-INFO'>PKG-INFO</a></b></td>
							<td style='padding: 8px;'>- Agent-API, an AI component of the LastMinuteAI project, leverages web resources to generate meaningful mind maps<br>- Equipped with features like multiple MCP support and multimodel agents, it uses Python 3.12 or higher and requires several dependencies for optimal functionality<br>- This package was crafted using Cookiecutter and the agent-api-cookiecutter project template.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- lastminute_api Submodule -->
			<details>
				<summary><b>lastminute_api</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.lastminute_api</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/config.py'>config.py</a></b></td>
							<td style='padding: 8px;'>- Configuring application settings, src/lastminute_api/config.py leverages the functionalities of pydantic_settings<br>- It establishes a Settings class, incorporating model configurations from an environment file<br>- This ensures proper loading and utilization of these settings throughout the application, contributing to an efficient and customizable runtime environment.</td>
						</tr>
					</table>
					<!-- infrastructure Submodule -->
					<details>
						<summary><b>infrastructure</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.lastminute_api.infrastructure</b></code>
							<!-- api Submodule -->
							<details>
								<summary><b>api</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.lastminute_api.infrastructure.api</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/api/main.py'>main.py</a></b></td>
											<td style='padding: 8px;'>- Main.py orchestrates the Agent API, which serves as a skeleton for creating robust APIs<br>- It manages startup and shutdown events, and provides endpoints for handling chat, evaluation, document ingestion, and memory reset requests<br>- The root endpoint redirects to API documentation<br>- Should the script be run directly, it uses uvicorn to serve the API.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/api/models.py'>models.py</a></b></td>
											<td style='padding: 8px;'>- Models.py in src/lastminute_api/infrastructure/api serves as a crucial interface for defining request payloads<br>- It contains Pydantic BaseModel subclasses representing various requests such as ChatRequest, EvalRequest, IngestDocumentsRequest, and ResetMemoryRequest, enabling structured data exchange across the application.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- nano_bannana Submodule -->
							<details>
								<summary><b>nano_bannana</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.lastminute_api.infrastructure.nano_bannana</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/nano_bannana/openai.py'>openai.py</a></b></td>
											<td style='padding: 8px;'>- OpenAI Client Implementation, located in the nano_banana module of the lastminute_api, facilitates interaction with the OpenAI service<br>- It defines methods for generating text and images and ensures appropriate configurations for API calls<br>- The client handles environment-specific defaults, validates input options and formats the responses, making it a crucial component in the projects infrastructure.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/nano_bannana/base.py'>base.py</a></b></td>
											<td style='padding: 8px;'>- The Nano Banana base in the LastMinute API provides an interface to multimodal inference providers for Swift STEM revision features<br>- It offers a structured response from a Nano Banana generation request, and supports multiple providers including gemini and together<br>- The base further enables caching instances per provider key.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/nano_bannana/together.py'>together.py</a></b></td>
											<td style='padding: 8px;'>- TogetherNanoBanana serves as a minimal interface between the official Together SDK and the projects Nano Banana protocol<br>- It primarily enables generation of Together responses, text streaming from Together chat completions, and Together client creation from configuration and environment settings<br>- Currently, it only supports the TEXT modality.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/nano_bannana/gemini.py'>gemini.py</a></b></td>
											<td style='padding: 8px;'>- GeminiNanoBanana serves as a client for Googles GenAI (Gemini) within the Lastminute API infrastructure<br>- It provides a streamlined interface for text and optional image responses, with support for both streaming and non-streaming interactions<br>- The module allows interactions with Gemini such as generating text responses, streaming events, and image creation, using a configuration-driven approach.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- llm_providers Submodule -->
							<details>
								<summary><b>llm_providers</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.lastminute_api.infrastructure.llm_providers</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/llm_providers/chat_groq.py'>chat_groq.py</a></b></td>
											<td style='padding: 8px;'>- ChatGroq provider in the LastMinute API infrastructure initializes a ChatGroq instance, leveraging LangChains library<br>- It manages the configuration setup involving API keys, model selection, and optional parameters like temperature, maximum tokens, and timeout<br>- This setup tailors the APIs behavior to suit various application requirements, enhancing the chat functionality.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/llm_providers/chat_openai.py'>chat_openai.py</a></b></td>
											<td style='padding: 8px;'>- The <code>chat_openai.py</code> within the <code>src/lastminute_api/infrastructure/llm_providers</code> directory establishes an interface to the OpenAI chat service<br>- It leverages the LangChains ChatOpenAI, facilitating the creation of an OpenAI chat instance by coercing configuration parameters and setting up necessary environment variables.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/llm_providers/base.py'>base.py</a></b></td>
											<td style='padding: 8px;'>- The base.py in the lastminute_api infrastructure creates chat Language Learning Model (LLM) instances using LangChain integrations with OpenAI and Groq<br>- It utilizes an environment-driven factory and configuration provided via environment variables<br>- This module also includes a simple in-process cache and the ability to collect configurations for different LLM types, providing a key component of the projects chat functionality.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- tools Submodule -->
							<details>
								<summary><b>tools</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.lastminute_api.infrastructure.tools</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/tools/tavily_serch.py'>tavily_serch.py</a></b></td>
											<td style='padding: 8px;'>- Acting as a bridge, tavily_serch.py enables interaction with the Tavily Search API within the Lastminute API infrastructure<br>- It provides a singular function that executes web searches using Tavilys search engine, returning the results for further application use<br>- This tool is a crucial component in query handling and response processes.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- mcp Submodule -->
							<details>
								<summary><b>mcp</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.lastminute_api.infrastructure.mcp</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/mcp/mcp_agent.py'>mcp_agent.py</a></b></td>
											<td style='padding: 8px;'>- Mcp_agent.py in the infrastructure layer of the lastminute_api application is responsible for creating an agent using the MCPAgent class<br>- This agent utilizes an MCP client and an LLM sourced from OpenAI, configured with a maximum step limit<br>- The agents creation is a crucial component in managing interactions with the MCP server.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/mcp/config.py'>config.py</a></b></td>
											<td style='padding: 8px;'>- Building the configuration for multiple MCP servers, the module in the src/lastminute_api/infrastructure/mcp/ directory extracts an API key from the environment and uses it to set up server details<br>- It supports the addition of new servers and forms an integral part of the Lastminute APIs infrastructure.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/infrastructure/mcp/mcp_client.py'>mcp_client.py</a></b></td>
											<td style='padding: 8px;'>- Leveraging the MCPClient class, the module at src/lastminute_api/infrastructure/mcp/mcp_client.py establishes a client for the MCP (Master Control Program) using a configuration built by the build_mcp_config function<br>- This ensures seamless integration with the MCP in the overall LastMinute API infrastructure.</td>
										</tr>
									</table>
								</blockquote>
							</details>
						</blockquote>
					</details>
					<!-- domain Submodule -->
					<details>
						<summary><b>domain</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.lastminute_api.domain</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/domain/exceptions.py'>exceptions.py</a></b></td>
									<td style='padding: 8px;'>- Serving as an essential component in the lastminute_api domain, exceptions.py defines custom exceptions for robust error handling<br>- These exceptions enable precise tracking and management of runtime anomalies, contributing to the stability and reliability of the overall codebase architecture.</td>
								</tr>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/domain/utils.py'>utils.py</a></b></td>
									<td style='padding: 8px;'>- Utilizing a range of utility functions, <code>src/lastminute_api/domain/utils.py</code> lends support to the broader LastMinute API project<br>- Through these utilities, it enhances code readability, maintainability, and reusability across the codebase, serving as a key pillar in the overall software architecture.</td>
								</tr>
							</table>
							<!-- prompts Submodule -->
							<details>
								<summary><b>prompts</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.lastminute_api.domain.prompts</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/domain/prompts/prompts.py'>prompts.py</a></b></td>
											<td style='padding: 8px;'>- Prompts.py in the lastminute_api domain serves as the central hub for managing interactive user prompts<br>- It is integral to the codebase, facilitating user interaction and input collection across the application<br>- Its strategic location within the domain directory highlights its importance in the overall project architecture.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/domain/prompts/get_prompts.py'>get_prompts.py</a></b></td>
											<td style='padding: 8px;'>- Get_prompts.py within the prompts domain of the lastminute_api forms a critical component of the overall architecture<br>- It serves to retrieve and manage user prompts, playing a key role in user interaction and data collection<br>- Through this functionality, it significantly contributes to the application's usability and data-driven operations.</td>
										</tr>
									</table>
								</blockquote>
							</details>
							<!-- tools Submodule -->
							<details>
								<summary><b>tools</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.lastminute_api.domain.tools</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/domain/tools/registry.py'>registry.py</a></b></td>
											<td style='padding: 8px;'>- The <code>registry.py</code> module in the <code>lastminute_api</code> domain handles the creation, storage, and retrieval of mindmaps<br>- It uses the <code>DynamicGraph</code> class to generate mindmaps from provided topics and subtopics, assigns a unique reference ID to each mindmap, and stores the mindmaps in a global cache<br>- It also provides functions to retrieve and display mindmaps using their reference IDs.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/domain/tools/graph_tool.py'>graph_tool.py</a></b></td>
											<td style='padding: 8px;'>- DynamicGraph in graph_tool.py provides a utility to build directed graphs, generate mind maps as PIL Images, and retrieve graph information<br>- Images can be returned as base64 encoded strings or saved locally<br>- It also includes a demo function for creating mind maps and a quick helper function to generate mind maps with default styling.</td>
										</tr>
									</table>
								</blockquote>
							</details>
						</blockquote>
					</details>
					<!-- application Submodule -->
					<details>
						<summary><b>application</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.lastminute_api.application</b></code>
							<!-- agent_service Submodule -->
							<details>
								<summary><b>agent_service</b></summary>
								<blockquote>
									<div class='directory-path' style='padding: 8px 0; color: #666;'>
										<code><b>⦿ src.lastminute_api.application.agent_service</b></code>
									<table style='width: 100%; border-collapse: collapse;'>
									<thead>
										<tr style='background-color: #f8f9fa;'>
											<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
											<th style='text-align: left; padding: 8px;'>Summary</th>
										</tr>
									</thead>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/application/agent_service/prompts.py'>prompts.py</a></b></td>
											<td style='padding: 8px;'>- Prompts.py in the LastMinute API serves as the messaging blueprint for different roles within the agent service<br>- It provides a comprehensive set of guidelines and prompt messages that facilitate inter-role communication, including supervisors, router and human router, and various types of agents like simple answer, deep research, quick search, and image generation<br>- It also includes a function to fetch specific prompts based on their names.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/application/agent_service/state.py'>state.py</a></b></td>
											<td style='padding: 8px;'>- AgentState, a class in the <code>agent_service</code> module, extends MessagesState to manage conversation history and session-specific data for the LangGraph MCP Revision Agent<br>- It retains the most recent user query, categorizes queries for routing, and stores the last generated answer, mind map data, generated images, and chat responses<br>- This class also tracks the status of subagent interactions and query completions.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/application/agent_service/service.py'>service.py</a></b></td>
											<td style='padding: 8px;'>- The service.py in the agent_service module of the lastminute_api application provides a high-level interface for managing agent services<br>- It facilitates the execution of a supervisory graph for a single user query, normalizes output into an AgentState instance, and offers both synchronous and asynchronous operation modes<br>- Its functionality also includes summarizing agent results for presentation layers and configuring agent service logging with a unique emoji formatter.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/application/agent_service/node.py'>node.py</a></b></td>
											<td style='padding: 8px;'>- Node in Agent ServiceThis Python file, <code>node.py</code>, is a key part of the Agent Service within the LastMinute API<br>- It operates as both a supervisory and worker node for orchestration purposes<br>- The file is primarily concerned with facilitating communication, data transformation, and command execution related to the functioning of the Agent Service.Its role within the broader LastMinute API is to interact with a variety of other services and tools, such as <code>langchain_core</code>, <code>pydantic</code>, <code>lastminute_api.application.agent_service</code>, <code>lastminute_api.domain.tools</code>, <code>lastminute_api.infrastructure.llm_providers</code>, <code>lastminute_api.infrastructure.mcp</code>, and <code>lastminute_api.infrastructure.nano_bannana</code>.The code in <code>node.py</code> is responsible for leveraging these various tools and services to help coordinate the operation of the agent service<br>- It manages the creation and display of mindmaps, the generation of prompts, the handling of different message types (such as AI, Human, and System messages), and the execution of specific commands.In addition to this, it interfaces with different providers to produce results based on different types of data<br>- This includes creating and managing agents, generating image results, and retrieving specific types of data.In summary, <code>node.py</code> serves as a critical orchestrator within the Agent Service, handling a variety of responsibilities to ensure the smooth and efficient operation of the service within the larger LastMinute API ecosystem.</td>
										</tr>
										<tr style='border-bottom: 1px solid #eee;'>
											<td style='padding: 8px;'><b><a href='/home/delen014/Desktop/persnol_projects/agent-api/blob/master/src/lastminute_api/application/agent_service/graph.py'>graph.py</a></b></td>
											<td style='padding: 8px;'>- Building a comprehensive supervision graph for the agent service, graph.py establishes the necessary interconnections among different nodes including supervisor, tavily_agent, mcp_agent, mind_map_agent, and image_agent<br>- It serves as the backbone of the agent service, managing the workflow and coordination among various agents by defining the relationships between them.</td>
										</tr>
									</table>
								</blockquote>
							</details>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---

## 🚀 Getting Started

### 🌟 Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Uv
- **Container Runtime:** Docker

### ⚡ Installation

Build agent-api from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone ../agent-api
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd agent-api
    ```

3. **Install the dependencies:**

<!-- SHIELDS BADGE CURRENTLY DISABLED -->
	<!-- [![docker][docker-shield]][docker-link] -->
	<!-- REFERENCE LINKS -->
	<!-- [docker-shield]: https://img.shields.io/badge/Docker-2CA5E0.svg?style={badge_style}&logo=docker&logoColor=white -->
	<!-- [docker-link]: https://www.docker.com/ -->

	**Using [docker](https://www.docker.com/):**

	```sh
	❯ docker build -t persnol_projects/agent-api .
	```
<!-- SHIELDS BADGE CURRENTLY DISABLED -->
	<!-- [![uv][uv-shield]][uv-link] -->
	<!-- REFERENCE LINKS -->
	<!-- [uv-shield]: https://img.shields.io/badge/uv-DE5FE9.svg?style=for-the-badge&logo=uv&logoColor=white -->
	<!-- [uv-link]: https://docs.astral.sh/uv/ -->

	**Using [uv](https://docs.astral.sh/uv/):**

	```sh
	❯ uv sync --all-extras --dev
	```

### 🔆 Usage

Run the project with:

**Using [docker](https://www.docker.com/):**
```sh
docker run -it {image_name}
```
**Using [uv](https://docs.astral.sh/uv/):**
```sh
uv run python {entrypoint}
```

### 🌠 Testing

Agent-api uses the {__test_framework__} test framework. Run the test suite with:

**Using [uv](https://docs.astral.sh/uv/):**
```sh
uv run pytest tests/
```

---

## 🌻 Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## 🤝 Contributing

- **💬 [Join the Discussions](https://LOCAL/persnol_projects/agent-api/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL/persnol_projects/agent-api/issues)**: Submit bugs found or log feature requests for the `agent-api` project.
- **💡 [Submit Pull Requests](https://LOCAL/persnol_projects/agent-api/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone /home/delen014/Desktop/persnol_projects/agent-api/
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to LOCAL**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://LOCAL{/persnol_projects/agent-api/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=persnol_projects/agent-api">
   </a>
</p>
</details>

---

## 📜 License

Agent-api is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## ✨ Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
