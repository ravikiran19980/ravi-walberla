@page contributor-guide Contributors Guide

This page contains instructions on how to get started with developing waLBerla.

## Prepare the Git Repository

The official waLBerla Git repository is hosted at [i10git.cs.fau.de], the GitLab instance of the [Chair for Systems Simulation](https://www.cs10.tf.fau.de/) at [FAU Erlangen-Nürnberg](https://fau.de/).
In order to contribute code to waLBerla, you will need to acquire an account there; to do so, please follow the instructions on the GitLab landing page.

## Create a Fork

Only the core developers of waLBerla have write-access to the primary repository.
To contribute, you will therefore have to create a fork of that repository by navigating to the [repository page](https://i10git.cs.fau.de/walberla/walberla) and selecting _Fork_ there.
In this fork, you may freely create branches and develop code, which may later be merged to a primary branch via [merge requests](#create-merge-request).

## Set Up your developer environment

To develop waLBerla, you will need at least the following software installed on your machine:

- An up-to-date C++ compiler
- `cmake` 3.26 or later and `make` or `ninja`: To invoke the build process.
- `python` 3.10 or later: To use our supplementary developer tools and codegeneration.
- Optionally, a MPI library for multi process development.
- Optionally, for GPU development:
  - At least CUDA 12.8 for Nvidia GPUs, or
  - At least ROCm/HIP 6.1 for AMD GPUs.

### Virtual Python Environment

Create a local virtual environment and install the required Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r utilities/developer-tools/requirements.txt
```

## Merge Request Guidelines {#create-merge-request}

To integrate your changes into the official waLBerla repository, create a [merge request](https://docs.gitlab.com/user/project/merge_requests/) from your development fork.

- **Source**: your branch with the intended changes, e.g. `development/walberla` (repository) and `featureX` (branch).
- **Target**: `walberla/walberla` (repository) and `master` (branch).

Leave the merge request as `draft` until you are done with all intended changes and have fulfilled our quality [Quality Requirements](#contribution-quality-requirements).

Once your merge request is ready, add the label `waiting-for-review` to it.
This will notify the waLBerla maintainers to initiate the review process.

> [!note]
> The person assigned is responsible for the next step.
> Assign yourself to the merge request when progress depends on you.

> [!important] Copyright
> We require the use of [spdx](https://spdx.dev/) machine-readable license and copyright text markers.
> Every source file must define a header containing at least `SPDX-License-Identifier` and `SPDX-FileCopyrightText`.

### Quality Requirements {#contribution-quality-requirements}

- [ ] The CI pipeline must pass.
- [ ] Your implementation must be tested. (See [V8-Contributors Guide -- Testing](#v8-contrib-testing))
- [ ] Your implementation needs good coverage (metric tested by our CI to identify untested code areas) [*Note: currently not available*].
- [ ] Every changed file should be passed to an [automatic code formatter](#automatic-formatting):
  - `CXX`: format with [clang-format](#formatting-clang-format).
  - `python`: format with [black](#formatting-black).
  - `cmake`: format with [cmake-format](#formatting-cmake-format).
- [ ] Every new or changed feature must be documented. (See the [V8-Contributors Guide -- Documentation](#v8core-contrib-documentation)).
- [ ] Annotate the use of generative AI according to waLBerla's [AI Guideline](#ai-guideline).
- [ ] Every file must include a spdx-compliant license header and and mark deviations from the overarching license where applicable.

## Review Guidelines {#review-guidelines}

- Reviewers are assigned by a maintainer or may assign themselves.
- A proper review includes:
  - viewing the code changes and commenting on correctness and quality.
  - checking that provided documentation is clear and complete.
  - inspecting included tests and examples.
- The reviewer has to approve the merge request (button in the merge request page) once they consider the request ready to merge.

Once all review threads are resolved, the responsible maintainer is to be assigned to the merge request to perform the actual merge.

## Guideline for AI-generated contributions {#ai-guideline}

> [!note]
> This guideline and it's phrasing is inspired by the AI contribution guidelines of the following groups:
>
> - [Python Developer's Guide](https://devguide.python.org/getting-started/ai-tools/index.htm)
> - [LLVM](https://llvm.org/docs/AIToolPolicy.html)
> - [numpy](https://numpy.org/devdocs/dev/ai_policy.html)
> - [The Linux Kernel](https://kernel.org/doc/html/next/process/coding-assistants.html)
>
> and the scientific-python blog post of Stéfan van der Walt, [Community Considerations Around AI Contributions](https://blog.scientific-python.org/scientific-python/community-considerations-around-ai/).

### Policy

- Contributors must fully understand and verify all LLM-generated code or text in detail before they ask other project members to review it.
- Contributors are fully responsible for all their contributions to the project, whether they were created using AI tooling or not.
- We expect MR authors and those filing issues to be able to explain their proposed changes in their own words.
- Disclosure of the use of AI tools in the MR is required.
- AI generated code must be labeled as such and published as public domain under the `CC-O` license.

### Copyright

The OSS project waLBerla is protected under the `GPLv3` license. We therefore exclude the use LLM-generated code within the core of waLBerla.
Contributors are free to use AI-tooling for *limited refactoring* on the framework code base
that does not change the structure of the code
(like renaming, bug fixes, localized modernizations, etc.).

Outside of the core libraries, e.g. in writing tests and example apps,
use of AI tooling is permitted, but must be properly disclosed.

Every source file must have the following header:

\verbatim
/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: [Year of creation] [Author Name] <author@mail.xx>
 */
\endverbatim

If a file was fully generated by LLM, or parts of it where generated and its authors agree to place the entire file in the public domain, this is replaced by:

\verbatim
/**
 * SPDX-License-Identifier: CC-0
 * SPDX-FileCreatedBy: [Human Name]
 * SPDX-FileCreatedUsing: [LLM/Agent Name]
 */
\endverbatim

If only part of a file is created by LLM and placed in the public domain, that part must be marked using SPDX snippet markers:

\verbatim
/**
 * SPDX-SnippetBegin
 * SPDX-License-Identifier: CC-0
 * SPDX-SnippetCreatedBy: [Human Name]
 * SPDX-SnippetCreatedUsing: [LLM/Agent Name]
 */

 ... code ...

 /* SPDX-SnippetEnd */
\endverbatim

Maintainers should do their best to check and enforce this.

> [!caution]
> Contributors are supposed to independently assess whether their contributed code is considered _LLM-generated_ or not. Every user of AI tooling should be aware the risk of contravening licenses. AI tooling will not reliably provide license information of their results and you as a contributor take full responsibility not to violate copyrights.

---

## Automatic Code Formatter {#automatic-formatting}

### CMake {#formatting-cmake-format}

> [!caution] **If you touch a CMake file, format it.**
> To keep the `git-blame` output somewhat reasonable, waLBerla's legacy code was not auto formatted in one go, but will gradually be formatted by it's current developers.

- Based on [`cmake-format`](https://cmake-format.readthedocs.io/en/latest/).
- Configuration can be found in `.cmake-format.yaml`.
- Relevant files are `*/CMakeLists.txt` and `cmake/*.cmake`.

To format your CMake files simply run:

```bash
cmake-format -i <files...>
```

> [!tip]
> Since `cmake-format` is far from perfect, you will encounter situations where you don't want cmake-format to break the readability for certain sections.
> To locally disable cmake-format, simple wrap `# cmake-format: off` and `# cmake-format: on` around your codeblock.

### C++ {#formatting-clang-format}

- Based on [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html).
- Configuration can be found in `.clang-format`.
- Relevant files are `*/*.{cpp,c,tpp,cu,hip,hpp,h}`.
  - `.cpp`: For C++ translation units.
  - `.c`: For C translation units.
  - `.tpp`: For template implementation translation units of header only files. Sometimes also called `.impl.hpp`
  - `.cu`: For CUDA translation units.
  - `.hip`: For HIP translation units.
  - `.hpp`: For C++ header files.
  - `.h`: For C header files.

To format your C++ files simply run:

```bash
clang-format -i <files...>
```

> [!note]
> There are great IDE integrations for C++ linting that help you correctly format and write better C++ code.

### Python {#formatting-black}

- Based on [`black`](https://black.readthedocs.io/en/stable/).
- We use the black default configuration; No config file.
- Relevant files are `*/*.py`.

#### Formatting files

To format your python files simply run:

```bash
black <files...>
```

> [!note]
> There are great IDE integrations for python linting that help you correctly format and write better python code.