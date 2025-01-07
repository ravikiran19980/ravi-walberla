# Using clang-tidy on waLBerla

See also the [Clang-Tidy Documentation](https://clang.llvm.org/extra/clang-tidy/).

This document briefly explains how to run a clang-tidy analysis and (auto-)fixing pass on the waLBerla code base.

## Set up CMake

To run clang-tidy, CMake needs to be configured to emit a compile command data base which clang-tidy
reads to figure out which files, with which compilation options, it should analyze.
Also, the build system should be set up in Debug mode with additional debugging features enabled
such that all debug code paths can be covered.

Here is a sample `CMakeUserPresets.json` with a configure preset called `clang-tidy`, which defines
a build system that can be used for clang-tidy analysis:

```JSON
{
  "version": 6,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 23,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "clang-tidy",
      "generator": "Unix Makefiles",
      "binaryDir": "${sourceDir}/build/clang-tidy",
      "cacheVariables": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": true,
        "WALBERLA_BUFFER_DEBUG": true,
        "WALBERLA_BUILD_TESTS": true,
        "WALBERLA_BUILD_BENCHMARKS": true,
        "WALBERLA_BUILD_TUTORIALS": true,
        "WALBERLA_BUILD_TOOLS": true,
        "WALBERLA_BUILD_WITH_MPI": true,
        "WALBERLA_BUILD_WITH_OPENMP": true,
        "CMAKE_BUILD_TYPE": "Debug",
        "WALBERLA_BUILD_WITH_METIS": true,
        "WALBERLA_BUILD_WITH_PARMETIS": true,
        "WALBERLA_BUILD_WITH_OPENMESH": true,
        "WALBERLA_DOUBLE_ACCURACY": true,
        "WALBERLA_LOGLEVEL": "DETAIL"
      }
    },
  ]
}
```

The above configuration:
 - requires that *OpenMesh* is installed - if you do not plan on analyzing the `mesh_common` and `mesh` modules,
   you can set `WALBERLA_BUILD_WITH_OPENMESH` to `false`.
 - requires that *Metis* and *ParMetis* are installed - if you do not plan to analyze the Metis-dependent code
   in `core` and `blockforest`, you can set `WALBERLA_BUILD_WITH_METIS` and `WALBERLA_BUILD_WITH_PARMETIS` to `false`.
 - prepares the build system also for tests, benchmarks and tutorials - if you do not wish to analyze those, you can exlude them.

Copy the preset JSON code to the `CMakeUserPresets.json` file in your waLBerla root directory and run `cmake --preset clang-tidy` to set up the build system.

Next, navigate to the `build/clang-tidy` subdirectory.
There, you will find the compile command data base `compile_commands.json`.
It will include all compile commands CMake would execute when building all targets enabled by your configure preset.
That list will include a number of commands that are not relevant to our analysis.

To limit the analysis only to a subset of waLBerla's modules,
we use the `filterCompileCommands` script in the `utilities/clang-tidy` subfolder.
That subfolder has been symlinked into the build directory, so we can use it directly from there.
For example, to analyze only the `core` and `blockforest` modules, run the filter script from within the build directory like this:

```bash
python utilities/clang-tidy/filterCompileCommands.py \
  -f compile_commands.json \
  --exclude * \
  --include core blockforest
```

Here, we pass the following options:
 - the `-f` flag locates the command database.
 - The first `--exclude` tells the script to exclude all files from the analysis.
 - The following `--include` flag re-includes a number of folders into the analysis;
   in this case, it includes any files within directories called `core` or `blockforest`.
   This includes not only `src/core` and `src/blockforest`, but also the respective directories in `tests`.
   To restrict to `src`, write `--include src/core src/blockforest` instead.

You may freely alternate `--exclude` and `--include` flags; each new flag will override any previous ones it overlaps with.

When running the above command, you should see an output similar to this,
indicating that of the original `N` compile commands, `M` were retained and the rest deleted:

```
loading compile commands file: build/clang-tidy/compile_commands.json
compile commands read: <N>
compile commands filtered: <M>
```

## Run the Analysis

After setting up the build system and filtering the compile command data base, we can now run clang-tidy.
To perform the analysis and write the output to a file `tidy.txt`, execute the following command:

```bash
run-clang-tidy -quiet > tidy.txt
```

In order to also see the output on the console, you may alternatively use this:

```bash
run-clang-tidy -quiet | tee tidy.txt
```

Depending on the number of files you are analyzing, this might take a few minutes.

### Filtering Headers

When we filtered the compile command data base, we only restricted the set of translation units that should
be analyzed. `clang-tidy` will still analyze any non-system header files included into those translation units.
If you want to only analyze and clean up one module, but want to ignore headers included from different modules,
you have to explicitly exclude those by specifying the `-header-filter` parameter to `run-clang-tidy`.

`-header-filter` takes a regular expression against which all included headers are checked,
and `clang-tidy` will only analyze those headers that match the regex.
For instance, to restrict yourself to headers from the `core` and `blockforest` modules,
set `-header-filter=".*/(core|blockforest)/.*"`.

### Filtering Warnings

`clang-tidy` will filter its checks according to the settings in the `.clang-tidy` file in the `waLBerla` project root.
You may additionally enable and disable checks on the command line;
e.g. the following call overrides the `.clang-tidy` settings by enabling the `misc-misplaced-const` check
and disabling the `modernize-use-emplace` check:

```bash
run-clang-tidy -checks=misc-misplaced-const,-modernize-use-emplace -quiet > tidy.txt
```

To only enable a few specific checks, and disable all else, start your `-checks` list with the exclude-all clause `-*`,
and then list all checks that should be included.

## Display a Summary

After the analysis, the `tidy.txt` output file might contain a very long list of warnings and errors.
To display a summary of the warnings that occured, and their frequency, run `python utilities/clang-tidy/clangTidySummary.py tidy.txt`.
This will display a table of warning identifiers
(see the [List of Clang-Tidy Checks](https://clang.llvm.org/extra/clang-tidy/checks/list.html)),
each with a number indicating how often it occured.

## Fix the Errors

Now that we know which style errors occur, we need to fix them.
For a wide range of errors, `clang-tidy` knows how to fix them automatically, and we can make use of this feature.
In other cases, we need to fix style errors by hand.

### Auto-Fixing

For many of its checks, especially in the `modernize` category, `clang-tidy` offers to auto-fix your code.
Before you start auto-fixing, to keep track of any changes and avoid unrecoverable errors, you should:
 - Only perform extensive auto-fixes with a *clean git working tree*;
 - Only perform one type of fix at a time;
 - `git commit` any significant set of fixes immediately.

For instance, to auto-fix all occurences of `modernize-use-using` within the `core` and `blockforest` modules
(i.e. automatically replace any `typedef` declarations with a modern `using` declaration),
run the following command:

```bash
run-clang-tidy -header-filter=".*/(core|blockforest)/.*"  -checks=-*,modernize-use-using -fix
```

Here,
 - the `-header-filter` makes sure that the fixes don't spread to any header files in other modules;
 - the `-checks` disables all checks except for the one we want to auto-fix;
 - and finally, the `-fix` flag instructs `clang-tidy` to perform the auto-fixing.


### Manual Fixing

To manually track down and fix occurences of `clang-tidy` errors,
re-run `run-clang-tidy` with only the specific check enabled, and with the necessary header filter.
Then, read through the output to locate the errors in the code files, and fix them one by one.
Commit your results afterward.

### Don't Forget To Test

After a session of code style fixes, make sure to run the test suite (e.g. using the GitLab CI) to see
if your changes have introduced any bugs. If they did, fix them.
